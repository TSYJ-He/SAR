# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than

from .sar_core import SARCore


from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead
    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    if "InternVLChatModel" in getattr(config, "architectures", []):
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model_for_sar(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    r"""
    Patches the model's attention layers to apply Semantic Attention Re-weighting (SAR) at inference time.
    This function dynamically replaces the forward method of each attention block.
    """
    if not finetuning_args.use_sar:
        return
    logger.info_rank0("Applying SAR patch to the model.")
    # Identify the decoder layers of the model. This is model-specific.
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # LLaMA, Qwen, etc.
        decoder_layers = model.model.layers
    elif hasattr(model, "layers"):  # Some models have layers directly
        decoder_layers = model.layers
    else:
        logger.warning_rank0("Could not find decoder layers to patch for SAR. Skipping.")
        return
    # To get the semantic prior, we need access to the vision encoder's intermediate features.
    # We will patch the vision encoder to store its last hidden state.
    if not hasattr(model, "vision_tower") or not hasattr(model.vision_tower, "vision_tower"):
        logger.warning_rank0("Could not find vision tower for SAR semantic prior. Skipping.")
        return
    vision_encoder = model.vision_tower.vision_tower
    original_vision_forward = vision_encoder.forward

    def patched_vision_forward(self, *args, **kwargs):
        outputs = original_vision_forward(*args, **kwargs)
        # Store the intermediate features for the semantic prior.
        # Let's use 2/3 depth as a heuristic.
        intermediate_layer_idx = (len(self.encoder.layers) * 2) // 3
        if kwargs.get("output_hidden_states", False):
            model._sar_vision_hidden_states = outputs.hidden_states[intermediate_layer_idx]
        else:
            # Re-run with output_hidden_states=True if not already available
            kwargs["output_hidden_states"] = True
            outputs = original_vision_forward(*args, **kwargs)
            model._sar_vision_hidden_states = outputs.hidden_states[intermediate_layer_idx]
        return outputs
    vision_encoder.forward = MethodType(patched_vision_forward, vision_encoder)

    for layer_idx, layer in enumerate(decoder_layers):
        # Access the attention module, which can be nested differently
        if hasattr(layer, "self_attn"):
            attention_module = layer.self_attn
        else:
            logger.warning_rank0(f"Layer {layer_idx} has no 'self_attn' module. Skipping SAR patch.")
            continue
        original_attention_forward = attention_module.forward
        sar_instance = SARCore(model, finetuning_args, layer_idx)

        def sar_forward(self, *args, **kwargs):
            # We need to pass `output_attentions=True` to get the scores
            original_output_attentions = kwargs.get("output_attentions", False)
            kwargs["output_attentions"] = True
            # The original forward call
            outputs = original_attention_forward(*args, **kwargs)
            # `outputs` is a tuple: (attn_output, attn_weights, past_key_value)
            attn_output, attn_weights, past_key_value = outputs

            if sar_instance.is_active and self.training is False:
                # During inference, run SAR
                input_ids = kwargs.get("input_ids", None)
                # pre_softmax_scores are not always returned. We recalculate them if needed.
                # This is a simplification; a more robust way would be to patch the score calculation itself.
                query_states = kwargs.get("hidden_states")  # Assuming hidden_states are passed
                # This part is highly model-dependent, so we will operate on the returned attention_probs
                # and assume pre_softmax_scores are similar in distribution for variance calculation.
                # For simplicity, we use attention_probs as a proxy for pre_softmax_scores for variance.
                head_weights = sar_instance.run(
                    pre_softmax_scores=attn_weights,  # Using probs as a proxy
                    attention_probs=attn_weights,
                    input_ids=input_ids,
                    vision_encoder_hidden_states=getattr(model, "_sar_vision_hidden_states", None),
                )

                if head_weights is not None:
                    # Reshape attn_output to be per-head before projection
                    bsz, q_len, _ = query_states.size()
                    head_dim = attn_output.size(-1) // self.num_heads

                    # Shape (bsz, q_len, num_heads, head_dim) -> (bsz, num_heads, q_len, head_dim)
                    attn_output_heads = attn_output.view(bsz, q_len, self.num_heads, head_dim).transpose(1, 2)
                    reweighted_heads = sar_instance.apply_reweighting(attn_output_heads, head_weights)
                    attn_output = reweighted_heads.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

            # Restore original `output_attentions` behavior for the final return value
            if not original_output_attentions:
                return attn_output, None, past_key_value
            else:
                return attn_output, attn_weights, past_key_value

        attention_module.forward = MethodType(sar_forward, attention_module)









def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
finetuning_args: "FinetuningArguments", #  Add
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)



#
    if finetuning_args.use_sar:
        patch_model_for_sar(model, finetuning_args)







    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
