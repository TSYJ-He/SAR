
import json
import os
import sys
from typing import TYPE_CHECKING, List, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llamafactory.data.loader import get_dataset
from llamafactory.eval.evaluator import Evaluator
from llamafactory.model.loader import load_model, load_tokenizer
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments,
    GeneratingArguments,
    get_eval_args,
)
from llamafactory.extras import logging
from llamafactory.analysis.focus_metrics import FocusMetricsEvaluator
from llamafactory.analysis.layerwise_probe import LayerwiseProbe
if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = logging.get_logger(__name__)


def run_sar_analysis(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        eval_args: "EvaluationArguments",
        finetuning_args: "FinetuningArguments",
) -> None:
    """
    Main entry point for running SAR-specific analyses, including SFS/FI metrics
    and layer-wise probing, as described in the paper.
    """
    # Load model and tokenizer using the corrected, separate functions
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
    # Load the specific dataset required for the analysis
    dataset = get_dataset(model_args, data_args, training_args=eval_args, stage="eval", tokenizer=tokenizer)
    # Branch logic to determine which analysis to run
    if eval_args.do_sfs_fi_eval:
        logger.info_rank0("Running Semantic Focus Score (SFS) and Focus Instability (FI) evaluation...")

        # Validate that necessary paths for grounding models are provided
        if not all([
            eval_args.gd_config_path,
            eval_args.gd_checkpoint_path,
            eval_args.sam_checkpoint_path
        ]):
            raise ValueError(
                "For SFS/FI evaluation, paths to GroundingDINO and SAM are required. "
                "Please provide --gd_config_path, --gd_checkpoint_path, and --sam_checkpoint_path."
            )

        focus_evaluator = FocusMetricsEvaluator(
            gd_config_path=eval_args.gd_config_path,
            gd_checkpoint_path=eval_args.gd_checkpoint_path,
            sam_checkpoint_path=eval_args.sam_checkpoint_path,
            sam_model_type=eval_args.sam_model_type,
        )

        results = focus_evaluator.evaluate(model, tokenizer, dataset)
        logger.info_rank0("SFS/FI Evaluation Results:")
        logger.info_rank0(json.dumps(results, indent=4))
        output_file = os.path.join(eval_args.output_dir, "focus_metrics_results.json")
        os.makedirs(eval_args.output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info_rank0(f"SFS/FI results saved to {output_file}")

    elif eval_args.do_layerwise_probe:
        logger.info_rank0("Running Layer-wise Probing analysis...")

        probe = LayerwiseProbe(
            model=model,
            tokenizer=tokenizer,
            task_type=eval_args.layerwise_probe_task,
            probe_lr=eval_args.probe_lr,
            probe_epochs=eval_args.probe_epochs,
            probe_batch_size=eval_args.probe_batch_size,
        )

        layer_performance = probe.run_probing(dataset)
        logger.info_rank0("Layer-wise Probing Results:")
        logger.info_rank0(json.dumps(layer_performance, indent=4))
        output_file = os.path.join(eval_args.output_dir, "layerwise_probe_results.json")
        os.makedirs(eval_args.output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(layer_performance, f, ensure_ascii=False, indent=4)
        logger.info_rank0(f"Layer-wise probing results saved to {output_file}")

def main():
    """
    Main function to parse arguments and launch either standard evaluation or SAR analysis.
    """
    # get_eval_args will be modified to include our new analysis arguments
    (
        model_args,
        data_args,
        eval_args,
        finetuning_args,
        generating_args,
    ) = get_eval_args()
    # Check for custom analysis flags
    is_sar_analysis = eval_args.do_sfs_fi_eval or eval_args.do_layerwise_probe

    if is_sar_analysis:
        run_sar_analysis(
            model_args=model_args,
            data_args=data_args,
            eval_args=eval_args,
            finetuning_args=finetuning_args,
        )
    else:
        # If no specific analysis flag is set, run the standard LlamaFactory evaluation workflow
        logger.info_rank0("No SAR-specific analysis task specified. Running standard LlamaFactory evaluation.")
        evaluator = Evaluator(
            model_args=model_args,
            data_args=data_args,
            eval_args=eval_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
        )
        evaluator.run()


if __name__ == "__main__":
    main()