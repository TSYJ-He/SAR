import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from .trainer import CustomSeq2SeqTrainer
from ...extras import logging
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from ...hparams import FinetuningArguments
logger = logging.get_logger(__name__)
class ControlSFTTrainer(CustomSeq2SeqTrainer):
    r"""
    A specialized SFT Trainer for the control group experiment (Section 3.3).

    This trainer overrides the `compute_loss` method to add an auxiliary
    visual supervision loss (CLIP-style contrastive loss) to the standard
    language modeling loss. This is used to empirically validate the effect
    of direct visual supervision on mitigating attention gradient sparsity.
    """
    def __init__(self, finetuning_args: "FinetuningArguments", *args, **kwargs):
        # The base class is CustomSeq2SeqTrainer, not a generic Trainer.
        super().__init__(finetuning_args=finetuning_args, *args, **kwargs)
        # Store the weight for the auxiliary loss from finetuning arguments
        # This argument will need to be added to FinetuningArguments
        self.visual_contrastive_loss_weight = getattr(finetuning_args, "visual_contrastive_loss_weight", 0.0)
        if self.is_in_train and self.visual_contrastive_loss_weight <= 0:
            logger.warning(
                "`visual_contrastive_loss_weight` is not set or non-positive. "
                "ControlSFTTrainer will behave like the standard SFT trainer."
            )
    def _compute_contrastive_loss(
            self,
            model: "PreTrainedModel",
            inputs: Dict[str, Union[torch.Tensor, int]],
            model_outputs
    ) -> torch.Tensor:
        r"""
        Computes the CLIP-style contrastive loss between global image and text representations.
        This implementation robustly handles batch processing and token type identification.
        """
        # The last hidden state contains the final representations for all tokens.
        last_hidden_state = model_outputs.hidden_states[-1]

        input_ids = inputs["input_ids"]
        image_token_id = getattr(self.model.config, "image_token_index", None)

        if image_token_id is None:
            logger.warning_once("`image_token_index` not found in model config. Cannot compute contrastive loss.")
            return torch.tensor(0.0, device=last_hidden_state.device)

        batch_size = last_hidden_state.shape[0]
        visual_features_list = []
        text_features_list = []
        special_token_ids = set(self.tokenizer.all_special_ids)
        for i in range(batch_size):
            current_input_ids = input_ids[i]



            # --- Robust Token Identification ---
            is_visual_token = (current_input_ids == image_token_id)
            if not torch.any(is_visual_token):
                continue  # Skip if no visual tokens are present in this sequence
            is_special_token = torch.tensor(
                [token_id in special_token_ids for token_id in current_input_ids],
                device=current_input_ids.device
            )

            # Text tokens are those that are NOT visual and NOT special tokens.
            is_text_token = ~is_visual_token & ~is_special_token

            visual_indices = is_visual_token.nonzero(as_tuple=True)[0]
            text_indices = is_text_token.nonzero(as_tuple=True)[0]

            if visual_indices.numel() == 0 or text_indices.numel() == 0:
                continue  # Skip if either modality is missing
            # --- Feature Pooling ---
            # Pool visual features using mean pooling.
            visual_features = last_hidden_state[i, visual_indices, :].mean(dim=0)
            visual_features_list.append(visual_features)

            # Pool text features (instruction part) using mean pooling.
            text_features = last_hidden_state[i, text_indices, :].mean(dim=0)
            text_features_list.append(text_features)

        if len(visual_features_list) < 2:  # Contrastive loss requires at least 2 samples
            logger.warning_once(
                "Batch size for contrastive loss is less than 2. Cannot compute loss for this step. "
                "This can happen with gradient accumulation and dropping samples."
            )
            return torch.tensor(0.0, device=last_hidden_state.device)

        # Stack and normalize features from the batch
        visual_embeds = F.normalize(torch.stack(visual_features_list), p=2, dim=-1)
        text_embeds = F.normalize(torch.stack(text_features_list), p=2, dim=-1)

        # --- Symmetric Contrastive Loss Calculation ---
        # Get the logit scale, which may be a learnable parameter.
        logit_scale = getattr(model, "logit_scale", None)
        if logit_scale is not None:
            temperature = logit_scale.exp()
        else:
            temperature = 1.0  # Fallback if no logit_scale is defined

        # logits_per_image: (num_valid_samples, num_valid_samples)
        logits_per_image = torch.matmul(visual_embeds, text_embeds.t()) * temperature
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(logits_per_image), device=logits_per_image.device, dtype=torch.long)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        contrastive_loss = (loss_img + loss_txt) / 2.0

        return contrastive_loss

    def compute_loss(
            self,
            model: "PreTrainedModel",
            inputs: Dict[str, Union[torch.Tensor, int]],
            return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""
        Overrides the default loss computation to include the auxiliary visual contrastive loss.
        The method signature is aligned with the base `transformers.Trainer`.
        """
        # Compute the standard supervised fine-tuning (SFT) loss
        # We need hidden states for the contrastive loss, so we ensure they are returned.
        model_kwargs = {
            "output_hidden_states": True,
            # Pass all inputs to the model
            **inputs,
        }

        outputs = model(**model_kwargs)

        # Standard language modeling loss
        if self.args.ignore_pad_token_for_loss:
            labels = inputs["labels"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = inputs["labels"]

        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs[0]

        # This logic is adapted from `transformers.Trainer._compute_loss` for decoder-only models
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        text_gen_loss = loss_fct(shift_logits, shift_labels)

        # Compute the auxiliary visual contrastive loss (only during training)
        if self.is_in_train and self.visual_contrastive_loss_weight > 0:
            contrastive_loss = self._compute_contrastive_loss(model, inputs, outputs)
            total_loss = text_gen_loss + self.visual_contrastive_loss_weight * contrastive_loss
            # Log individual losses for monitoring
            self.log({
                "loss": total_loss.item(),  # The main loss for the optimizer
                "loss_text_gen": text_gen_loss.item(),
                "loss_contrastive": contrastive_loss.item(),
            })
        else:
            # If not in training or weight is zero, just use the standard SFT loss
            total_loss = text_gen_loss

        return (total_loss, outputs) if return_outputs else total_loss