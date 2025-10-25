import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from transformers import TrainerCallback, TrainerState, TrainingArguments
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
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Identifies the indices for visual and text tokens.
    """
    if input_ids is None or image_token_id is None:
        return None, None
    is_visual_token = (input_ids == image_token_id)
    # Ensure we handle the case where there are no visual tokens
    if not torch.any(is_visual_token):
        return None, None
    visual_indices = is_visual_token.nonzero(as_tuple=True)[1]

    # Create a mask for all tokens and set visual token positions to False
    all_indices_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    all_indices_mask.scatter_(1, visual_indices.unsqueeze(0), False)

    # Text tokens are non-special, non-visual tokens
    # This is a simplification; a more robust version would exclude padding, bos, eos etc.
    text_indices = all_indices_mask.nonzero(as_tuple=True)[1]

    return visual_indices, text_indices


def _gini_coefficient(data: torch.Tensor) -> float:
    """
    Calculates the Gini coefficient of a PyTorch tensor.
    A value of 1 represents maximum sparsity (inequality), 0 represents perfect density (equality).
    """
    if data is None or data.numel() == 0:
        return 0.0

    # Flatten the tensor and ensure it's float
    data = data.flatten().float()

    # The tensor must be non-negative
    if torch.any(data < 0):
        data = torch.abs(data)

    # Sort the data
    sorted_data, _ = torch.sort(data)

    n = len(sorted_data)
    if n < 2:
        return 0.0

    # Calculate the cumulative sum
    cum_data = torch.cumsum(sorted_data, dim=0)

    # Gini formula
    lorenz_area = cum_data.sum() / (n * cum_data[-1])
    return (0.5 - lorenz_area.item()) / 0.5


class GradientSparsityCallback(TrainerCallback):
    """
    A TrainerCallback to analyze and log gradient sparsity and magnitude during training.

    This callback is designed to provide empirical evidence for the claims in
    Section 3 of the SAR paper. It logs two key types of metrics:
    1. Gradient Magnitude Disparity: Compares the avg. L1 norm of gradients for
       visual vs. text token embeddings.
    2. Attention Gradient Sparsity: Measures the Gini coefficient of the gradients
       of the query and key projection weights in each attention layer.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        if not is_wandb_available():
            logger.warning("WandB is not installed. GradientSparsityCallback will not log anything.")

    def on_step_end(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            control: "TrainerControl",
            model: "PreTrainedModel",
            **kwargs,
    ):
        """
        Hook called at the end of a training step, after backward pass.
        """
        if not is_wandb_available() or state.global_step % self.log_every_n_steps != 0:
            return

        log_data = {}

        # --- 1. Gradient Magnitude Disparity Analysis ---
        try:
            input_ids = kwargs.get("inputs", {}).get("input_ids")
            image_token_id = getattr(model.config, "image_token_index", None)

            embedding_layer = model.get_input_embeddings()
            if input_ids is not None and image_token_id is not None and embedding_layer.weight.grad is not None:
                grad_data = embedding_layer.weight.grad.detach()

                # Get gradients for each token in the current batch's input_ids
                batch_grads = grad_data[input_ids]  # Shape: (batch_size, seq_len, hidden_dim)

                visual_indices, text_indices = _get_token_indices(input_ids[0], image_token_id)

                if visual_indices is not None and text_indices is not None:
                    visual_grads = batch_grads[:, visual_indices, :]
                    text_grads = batch_grads[:, text_indices, :]

                    avg_visual_grad_norm = torch.mean(torch.abs(visual_grads)).item()
                    avg_text_grad_norm = torch.mean(torch.abs(text_grads)).item()

                    log_data["gradient_analysis/avg_visual_token_grad_L1"] = avg_visual_grad_norm
                    log_data["gradient_analysis/avg_text_token_grad_L1"] = avg_text_grad_norm
        except Exception as e:
            logger.warning(f"Could not compute gradient magnitude disparity: {e}")

        # --- 2. Attention Gradient Sparsity Analysis ---
        try:
            # Identify decoder layers
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
                    q_gini = _gini_coefficient(q_grad) if q_grad is not None else 0.0
                    k_gini = _gini_coefficient(k_grad) if k_grad is not None else 0.0
                    log_data[f"attention_grad_sparsity/layer_{i}_q_gini"] = q_gini
                    log_data[f"attention_grad_sparsity/layer_{i}_k_gini"] = k_gini
                    q_gini_scores.append(q_gini)
                    k_gini_scores.append(k_gini)

            # Log aggregated stats for easier plotting
            if num_layers > 0:
                third = num_layers // 3
                log_data["attention_grad_sparsity/avg_shallow_q_gini"] = sum(q_gini_scores[:third]) / max(1, third)
                log_data["attention_grad_sparsity/avg_middle_q_gini"] = sum(q_gini_scores[third:2 * third]) / max(1,
                                                                                                                  third)
                log_data["attention_grad_sparsity/avg_deep_q_gini"] = sum(q_gini_scores[2 * third:]) / max(1,
                                                                                                           num_layers - 2 * third)

        except Exception as e:
            logger.warning(f"Could not compute attention gradient sparsity: {e}")

        if log_data:
            wandb.log(log_data, step=state.global_step)