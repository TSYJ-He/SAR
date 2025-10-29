
import json
import os
import sys
from typing import TYPE_CHECKING, List, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llamafactory.data.loader import get_dataset
from llamafactory.eval.evaluator import Evaluator
from llamafactory.model.loader import load_model, load_tokenizer
from datetime import datetime
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
import torch
import numpy as np
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
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


def run_mme_eval(
    model_args: "ModelArguments",
    eval_args: "EvaluationArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """
    Run MME (Multimodal Evaluation) benchmark evaluation.
    MME is a yes/no VQA benchmark with 14 perception and cognition categories.
    """
    logger.info_rank0("Running MME evaluation...")

    # Load model, tokenizer and processor
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    if processor is None:
        raise ValueError("MME evaluation requires a processor for handling images. Please ensure your model supports vision inputs.")

    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
    model.eval()

    # Load MME dataset
    mme_data_path = os.path.join(eval_args.task_dir, "mme/data")
    if not os.path.exists(mme_data_path):
        raise ValueError(f"MME data not found at {mme_data_path}")

    logger.info_rank0(f"Loading MME dataset from {mme_data_path}")
    dataset = hf_load_dataset(mme_data_path, split="test")

    # Get unique categories
    categories = sorted(set(dataset["category"]))
    logger.info_rank0(f"Found {len(categories)} categories: {categories}")

    # Initialize results storage
    category_results = {cat: {"correct": 0, "total": 0} for cat in categories}
    all_predictions = []

    # Process each example
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc="Evaluating MME"):
        # for idx in range(10):
            example = dataset[idx]
            image = example["image"]
            question = example["question"]
            answer = example["answer"]
            category = example["category"]

            # Create messages in chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            # Process inputs
            try:
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
            except Exception as e:
                # Fallback for processors that don't have apply_chat_template
                logger.warning_rank0(f"apply_chat_template failed, using simple text format: {e}")
                text = f"{question}"
                inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate prediction using model.generate (better for LLaVA-OneVision)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,  # Short answer: Yes or No
                do_sample=False,    # Deterministic generation
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode the generated text (only new tokens)
            generated_text = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Parse the answer - check for Yes/No in the generated text
            generated_lower = generated_text.lower()
            if "yes" in generated_lower:
                predicted_answer = "Yes"
            elif "no" in generated_lower:
                predicted_answer = "No"
            else:
                # If unclear, check which word comes first or use the first word
                if generated_text:
                    first_word = generated_text.split()[0].strip('.,!?').capitalize()
                    if first_word in ["Yes", "No"]:
                        predicted_answer = first_word
                    else:
                        # Default to No if unclear (conservative)
                        predicted_answer = "No"
                        if idx < 10:  # Only log first few warnings
                            logger.warning_rank0(f"Unclear answer: '{generated_text}'")
                else:
                    predicted_answer = "No"

            # Check if correct
            is_correct = (predicted_answer == answer)
            category_results[category]["correct"] += int(is_correct)
            category_results[category]["total"] += 1

            all_predictions.append({
                "question_id": example["question_id"],
                "category": category,
                "question": question,
                "predicted": predicted_answer,
                "predicted_raw": generated_text,
                "ground_truth": answer,
                "correct": is_correct
            })
            
    
    # Calculate and display results
    logger.info_rank0("\n" + "="*60)
    logger.info_rank0("MME Evaluation Results")
    logger.info_rank0("="*60)

    total_correct = 0
    total_count = 0

    for category in categories:
        correct = category_results[category]["correct"]
        total = category_results[category]["total"]
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        # MME uses a scoring system where each correct answer gets points
        score = correct * 100.0 / total * 2 if total > 0 else 0.0
        logger.info_rank0(f"{category:>25}: {accuracy:>6.2f}% ({correct}/{total}) | Score: {score:>7.2f}")
        total_correct += correct
        total_count += total

    overall_accuracy = 100.0 * total_correct / total_count
    overall_score = total_correct * 100.0 / total_count * 2

    logger.info_rank0("="*60)
    logger.info_rank0(f"{'Overall':>25}: {overall_accuracy:>6.2f}% ({total_correct}/{total_count}) | Score: {overall_score:>7.2f}")
    logger.info_rank0("="*60)

    # Save results
    if eval_args.save_dir is not None:
        os.makedirs(eval_args.save_dir, exist_ok=True)

        # Save detailed predictions
        predictions_file = os.path.join(eval_args.save_dir, "mme_predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)

        # Save summary results
        summary = {
            "overall": {
                "accuracy": overall_accuracy,
                "score": overall_score,
                "correct": total_correct,
                "total": total_count
            },
            "categories": {
                cat: {
                    "accuracy": 100.0 * category_results[cat]["correct"] / category_results[cat]["total"] if category_results[cat]["total"] > 0 else None,
                    "score": category_results[cat]["correct"] * 100.0 / category_results[cat]["total"] * 2 if category_results[cat]["total"] > 0 else None,
                    "correct": category_results[cat]["correct"] if category_results[cat]["total"] > 0 else None,
                    "total": category_results[cat]["total"] if category_results[cat]["total"] > 0 else None
                }
                for cat in categories
            }
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(eval_args.save_dir, f"mme_results_{timestamp}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info_rank0(f"\nResults saved to {eval_args.save_dir}")
        logger.info_rank0(f"  - Predictions: {predictions_file}")
        logger.info_rank0(f"  - Summary: {summary_file}")


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
    
    print("Current SAR Status:")
    print(f"  - SAR is enabled: {finetuning_args.use_sar}")
    print(f"  - Activation layers (K): {finetuning_args.sar_activation_layer_k}")
    print(f"  - Beta (temperature): {finetuning_args.sar_beta}")
    print()
    
    # Check if this is MME evaluation
    is_mme = eval_args.task.lower() == "mme"

    # Check if this is sar 

    if is_mme:
        run_mme_eval(
            model_args=model_args,
            eval_args=eval_args,
            finetuning_args=finetuning_args,
        )
    elif is_sar_analysis:
        run_sar_analysis(
            model_args=model_args,
            data_args=data_args,
            eval_args=eval_args,
            finetuning_args=finetuning_args,
        )
    else:
        # If no specific analysis flag is set, run the standard LlamaFactory evaluation workflow
        logger.info_rank0("No SAR-specific analysis task specified. Running standard LlamaFactory evaluation.")

        evaluator = Evaluator()
        evaluator.eval()


if __name__ == "__main__":
    main()