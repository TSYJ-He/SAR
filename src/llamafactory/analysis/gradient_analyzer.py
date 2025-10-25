import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from transformers import TrainerCallback, TrainerState, TrainingArguments, PreTrainedTokenizer
from transformers.integrations import is_wandb_available
from ..extras import logging
if TYPE_CHECKING:
    from transformers import TrainerControl, PreTrainedModel
logger = logging.get_logger(__name__)
if is_wandb_available():
    import wandb

def _get_token_indices(
        input_ids: torch.LongTensor,
        image_token_id: int,
        tokenizer: "PreTrainedTokenizer"
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Identifies and returns the indices for visual and text tokens within a sequence.
    This function is enhanced to robustly distinguish between visual tokens, actual text tokens,
    and special tokens (e.g., padding, BOS, EOS) by using the tokenizer.
    """
    if input_ids is None or image_token_id is None or tokenizer is None:
        return None, None

    # Identify all special tokens
    special_token_ids = set(tokenizer.all_special_ids)
    is_special_token = torch.tensor([token_id in special_token_ids for token_id in input_ids], device=input_ids.device)
    #  Identify visual tokens
    is_visual_token = (input_ids == image_token_id)
    if not torch.any(is_visual_token):
        return None, None
    # Identify text tokens: must not be visual AND must not be special
    is_text_token = ~is_visual_token & ~is_special_token

    visual_indices = is_visual_token.nonzero(as_tuple=True)[0]
    text_indices = is_text_token.nonzero(as_tuple=True)[0]
    if visual_indices.numel() == 0 or text_indices.numel() == 0:
        return None, None  # Ensure both types of tokens are present

    return visual_indices, text_indices


def _gini_coefficient(data: torch.Tensor) -> float:
    """
    Calculates the Gini coefficient of a PyTorch tensor. A value of 1 represents
    maximum sparsity (inequality), while 0 represents perfect density (equality).
    """
    if data is None or data.numel() < 2:
        return 0.0

    # Flatten, convert to float, and ensure non-negativity
    data = torch.abs(data.flatten().float())

    # Sort the data in ascending order
    sorted_data = torch.sort(data)[0]

    n = len(sorted_data)
    # Calculate the cumulative sum
    cum_data = torch.cumsum(sorted_data, dim=0)
    # Gini formula: G = 1 - 2 * (Area under Lorenz Curve)
    # Area under Lorenz Curve = sum of cumulative proportions / n
    lorenz_area = cum_data.sum() / (n * cum_data[-1])
    gini = 1 - 2 * lorenz_area.item()

    return gini


class GradientSparsityCallback(TrainerCallback):
    """
    HINTS: A TrainerCallback to analyze and log gradient sparsity and magnitude during training.
    This callback is designed to provide empirical evidence for the claims in
    Section 3 of the SAR paper. It logs two key types of metrics to WandB:
    1.  Gradient Magnitude Disparity [Exp 3.1]: Compares the average L1 norm of gradients for
        visual vs. text token embeddings at the input layer.
    2.  Attention Gradient Sparsity [Exp 3.2]: Measures the Gini coefficient of the gradients
        of the query and key projection weights in each attention layer to track sparsity evolution.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizer", log_every_n_steps: int = 50):
        super().__init__()
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        if not is_wandb_available():
            logger.warning("WandB is not installed. GradientSparsityCallback will be disabled.")

    def on_step_end(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            control: "TrainerControl",
            model: "PreTrainedModel",
            **kwargs,
    ):
        """
        Hook called at the end of a training step, after the backward pass and before the optimizer step.
        """
        if not is_wandb_available() or state.global_step % self.log_every_n_steps != 0:
            return

        log_data = {}
        try:
            input_ids = kwargs.get("inputs", {}).get("input_ids")
            image_token_id = getattr(model.config, "image_token_index", None)
            embedding_layer = model.get_input_embeddings()
            if input_ids is not None and image_token_id is not None and hasattr(embedding_layer.weight, "grad"):
                grad_data = embedding_layer.weight.grad
                if grad_data is not None:
                    grad_data = grad_data.detach()
                    visual_grads_list, text_grads_list = [], []
                    for i in range(input_ids.shape[0]):  # Iterate over batch
                        visual_indices, text_indices = _get_token_indices(
                            input_ids[i], image_token_id, self.tokenizer
                        )

                        if visual_indices is not None and text_indices is not None:
                            visual_grads_list.append(grad_data[input_ids[i][visual_indices]])
                            text_grads_list.append(grad_data[input_ids[i][text_indices]])
                    if visual_grads_list and text_grads_list:
                        all_visual_grads = torch.cat(visual_grads_list, dim=0)
                        all_text_grads = torch.cat(text_grads_list, dim=0)

                        log_data["gradient_analysis/avg_visual_token_grad_L1"] = torch.mean(
                            torch.abs(all_visual_grads)).item()
                        log_data["gradient_analysis/avg_text_token_grad_L1"] = torch.mean(
                            torch.abs(all_text_grads)).item()
        except Exception as e:
            logger.warning(f"Could not compute gradient magnitude disparity: {e}", exc_info=True)

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                decoder_layers = model.model.layers
            elif hasattr(model, "layers"):
                decoder_layers = model.layers
            else:
                decoder_layers = []
            num_layers = len(decoder_layers)
            q_gini_scores, k_gini_scores = [], []

            for i, layer in enumerate(decoder_layers):
                if hasattr(layer, "self_attn"):
                    q_grad = getattr(layer.self_attn.q_proj.weight, "grad", None)
                    k_grad = getattr(layer.self_attn.k_proj.weight, "grad", None)

                    q_gini = _gini_coefficient(q_grad)
                    k_gini = _gini_coefficient(k_grad)

                    log_data[f"attention_grad_sparsity_per_layer/layer_{i}_q_gini"] = q_gini
                    log_data[f"attention_grad_sparsity_per_layer/layer_{i}_k_gini"] = k_gini
                    q_gini_scores.append(q_gini)
                    k_gini_scores.append(k_gini)

            # Log aggregated stats for easier plotting of shallow, middle, and deep layers
            if num_layers > 0:
                third = num_layers // 3
                log_data["attention_grad_sparsity_agg/avg_shallow_q_gini"] = sum(q_gini_scores[:third]) / max(1, third)
                log_data["attention_grad_sparsity_agg/avg_middle_q_gini"] = sum(q_gini_scores[third:2 * third]) / max(1,
                                                                                                                      third)
                log_data["attention_grad_sparsity_agg/avg_deep_q_gini"] = sum(q_gini_scores[2 * third:]) / max(1,
                                                                                                               num_layers - 2 * third)
                log_data["attention_grad_sparsity_agg/avg_total_q_gini"] = sum(q_gini_scores) / num_layers

        except Exception as e:
            logger.warning(f"Could not compute attention gradient sparsity: {e}", exc_info=True)

        if log_data:
            wandb.log(log_data, step=state.global_step)