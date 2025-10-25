"""
sar_core.py is designed to be a pure, self-contained,
and model-agnostic library for the SAR algorithm.
Its only responsibility is to perform the mathematical computations for
SAV and the re-weighting logic.

It does not know or care about specific model architectures,
caching strategies, or attention implementations.

All the model-specific integration and boilerplate handling is delegated
to src/llamafactory/model/patcher.py, which will dynamically inject the
logic from sar_core.py into any given MLLM.

 """
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

_semantic_prior_cache: Dict[str, torch.Tensor] = {}


class SARCore:
    """
    Encapsulates the core logic for Semantic Attention Re-weighting (SAR).
    This class is designed to be model-agnostic. It operates on tensor inputs
    (hidden states, attention scores) and returns computed weights. The integration
    into a specific model is handled by a separate patching mechanism.
    """
    def __init__(
            self,
            model: PreTrainedModel,
            finetuning_args: "FinetuningArguments",
            layer_idx: int
    ):
        self.config = model.config
        self.finetuning_args = finetuning_args
        self.layer_idx = layer_idx
        self.device = model.device
        self.dtype = model.dtype
        self.is_active = self._is_sar_active()

    def _is_sar_active(self) -> bool:
        """Checks if SAR should be applied to the current layer."""
        if not self.finetuning_args.use_sar:
            return False

        total_layers = getattr(self.config, "num_hidden_layers", 0)
        if total_layers == 0:
            return False
        # SAR only for the final K layers
        activation_start_layer = total_layers - self.finetuning_args.sar_activation_layer_k
        return self.layer_idx >= activation_start_layer

    def _get_token_indices(
            self,
            input_ids: torch.LongTensor
    ) -> Tuple[Optional[int], Optional[int]]:
        # A common convention is to use a specific placeholder ID. We generalize it.
        # This will be refined based on the specific tokenizer/model.
        # TODO: Replace with robust token identification logic based on model type.
        image_token_id = getattr(self.config, "image_token_index", -1)
        if image_token_id == -1:
            return None, None
        vision_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        if len(vision_token_indices[1]) == 0:
            return None, None

        vision_start_index = vision_token_indices[1].min().item()
        num_vision_tokens = len(vision_token_indices[1])

        return vision_start_index, num_vision_tokens

    def _compute_semantic_prior(
            self,
            vision_encoder_hidden_states: torch.Tensor,
            num_vision_tokens: int,
            start_index: int
    ) -> torch.Tensor:
        """
        Computes or retrieves from cache the query-agnostic semantic prior.
        """
        cache_key = f"{vision_encoder_hidden_states.device}_{vision_encoder_hidden_states.shape}"
        if cache_key in _semantic_prior_cache:
            return _semantic_prior_cache[cache_key]

        visual_features = vision_encoder_hidden_states[:, start_index: start_index + num_vision_tokens, :]
        l2_norms = torch.linalg.vector_norm(visual_features.to(self.device, dtype=torch.float32), ord=2, dim=-1)
        semantic_prior = F.normalize(l2_norms, p=1, dim=-1, eps=1e-6)

        _semantic_prior_cache[cache_key] = semantic_prior
        return semantic_prior.to(self.dtype)

    def _compute_sav_scores(
            self,
            pre_softmax_scores: torch.Tensor,
            attention_probs: torch.Tensor,
            semantic_prior: torch.Tensor,
            vision_start_index: int,
            num_vision_tokens: int,
            num_text_tokens: int
    ) -> torch.Tensor:

        cross_attn_scores = pre_softmax_scores[
                            :, :, -num_text_tokens:, vision_start_index: vision_start_index + num_vision_tokens
                            ]
        cross_attn_probs = attention_probs[
                           :, :, -num_text_tokens:, vision_start_index: vision_start_index + num_vision_tokens
                           ]
        # Alignment Variance (AV)
        alignment_variance = torch.mean(torch.var(cross_attn_scores, dim=-1, unbiased=True), dim=-1)
        #SA
        semantic_prior_reshaped = semantic_prior.unsqueeze(1).unsqueeze(2)
        semantic_alignment_per_query = torch.sum(cross_attn_probs * semantic_prior_reshaped, dim=-1)
        semantic_alignment = torch.mean(semantic_alignment_per_query, dim=-1)
        # SAV
        sav_scores = alignment_variance * semantic_alignment
        return sav_scores

    def _get_sar_weights(
            self,
            sav_scores: torch.Tensor
    ) -> torch.Tensor:
        beta = self.finetuning_args.sar_beta
        if beta <= 0:
            raise ValueError("Temperature beta must be positive.")
        scaled_sav = sav_scores / beta
        head_weights = F.softmax(scaled_sav, dim=-1)
        max_weights, _ = torch.max(head_weights, dim=-1, keepdim=True)
        head_weights[:, 0] = max_weights.squeeze(-1)  # Protect the first head
        return head_weights.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def apply_reweighting(
            attention_output_heads: torch.Tensor,
            head_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the re-weighting to the outputs of the attention heads.
        """
        num_heads = attention_output_heads.shape[1]
        reweighted_output = attention_output_heads * head_weights * num_heads

        return reweighted_output

    def run(
            self,
            pre_softmax_scores: torch.Tensor,
            attention_probs: torch.Tensor,
            input_ids: torch.LongTensor,
            vision_encoder_hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        The main execution flow for SAR. Computes head weights if active.
        """
        if not self.is_active or input_ids is None:
            return None

        with torch.no_grad():
            seq_len = input_ids.shape[-1]
            vision_start_index, num_vision_tokens = self._get_token_indices(input_ids)

            if vision_start_index is None:
                return None
            num_text_tokens = seq_len - (vision_start_index + num_vision_tokens)
            if num_text_tokens <= 0:
                return None
            try:
                semantic_prior = self._compute_semantic_prior(
                    vision_encoder_hidden_states, num_vision_tokens, vision_start_index
                )

                sav_scores = self._compute_sav_scores(
                    pre_softmax_scores,
                    attention_probs,
                    semantic_prior,
                    vision_start_index,
                    num_vision_tokens,
                    num_text_tokens
                )
                head_weights = self._get_sar_weights(sav_scores)
                return head_weights.to(self.dtype)
            except Exception:
                return None