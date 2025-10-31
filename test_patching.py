"""
Test script to verify if SAR is actually being applied during model inference.

This script:
1. Loads a model with use_sar=False and use_sar=True
2. Performs generation with the same input
3. Checks if the outputs differ
4. Inspects the model architecture to verify SAR patching
"""

import os
import sys
import torch
from PIL import Image
from types import MethodType

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.hparams import ModelArguments, FinetuningArguments
from llamafactory.model.loader import load_tokenizer, load_model


def get_decoder_layers(model):
    """Get decoder layers from the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    elif hasattr(model, "layers"):
        return model.layers
    return None


def get_vision_encoder(model):
    """Get vision encoder from the model."""
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_tower"):
        return model.vision_tower.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    return None


def verify_sar_patching(model, finetuning_args=None):
    """Check if SAR has been applied to the model."""
    print("\n" + "="*80)
    print("VERIFYING SAR PATCHING")
    print("="*80)

    # Check vision encoder patching
    vision_encoder = get_vision_encoder(model)
    if vision_encoder:
        is_patched = isinstance(vision_encoder.forward, MethodType)
        print(f"✓ Vision encoder: {vision_encoder.__class__.__name__} (patched: {is_patched})")
    else:
        print("✗ No vision encoder found")

    # Check decoder layers
    decoder_layers = get_decoder_layers(model)
    if decoder_layers is None:
        print("✗ Could not find decoder layers")
        print("="*80 + "\n")
        return False
    
    num_layers = len(decoder_layers)
    print(f"✓ Found {num_layers} decoder layers")
    
    # Check SAR activation layers
    if finetuning_args and finetuning_args.use_sar:
        total_layers = getattr(model.config, "num_hidden_layers", num_layers)
        activation_start = total_layers - finetuning_args.sar_activation_layer_k
        print(f"  - SAR active in layers [{activation_start}, {total_layers-1}] (K={finetuning_args.sar_activation_layer_k})")
    
    # Check attention patching
    patched_count = sum(1 for layer in decoder_layers if hasattr(layer, "self_attn") and isinstance(layer.self_attn.forward, MethodType))
    print(f"✓ {patched_count}/{num_layers} attention layers patched")
    
    # Check generate method patching
    generate_is_patched = isinstance(model.generate, MethodType)
    print(f"✓ Model.generate patched: {generate_is_patched}")
    
    print("="*80 + "\n")
    return patched_count > 0


def load_model_variant(model_name, use_sar, sar_k=10, sar_beta=0.5):
    """Load a model variant with or without SAR."""
    model_args = ModelArguments(model_name_or_path=model_name, trust_remote_code=True)
    finetuning_args = FinetuningArguments(
        stage="sft",
        use_sar=use_sar,
        sar_activation_layer_k=sar_k if use_sar else None,
        sar_beta=sar_beta if use_sar else None,
    )
    
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(
        tokenizer=tokenizer_module["tokenizer"],
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=False,
    )
    model.eval()
    model.generation_config.use_cache = False
    
    return model, tokenizer_module["tokenizer"], tokenizer_module["processor"]


def extract_generated_text(outputs, inputs, tokenizer):
    """Extract generated text from GenerateDecoderOnlyOutput."""
    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs[0]
    generated_ids = sequences[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, text


def test_sar_effectiveness():
    """Test if SAR actually changes model outputs."""
    print("\n" + "="*80)
    print("TESTING SAR EFFECTIVENESS")
    print("="*80)

    # Model configuration
    # model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # model_name = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load models
    print("\n[1/3] Loading models...")
    model_no_sar, tokenizer, processor = load_model_variant(model_name, use_sar=False)
    print("✓ Model without SAR loaded")
    verify_sar_patching(model_no_sar)
    
    model_with_sar, _, _ = load_model_variant(model_name, use_sar=True)
    print("✓ Model with SAR loaded")
    sar_patched = verify_sar_patching(model_with_sar, FinetuningArguments(use_sar=True, sar_activation_layer_k=10))
    
    if not sar_patched:
        print("⚠ WARNING: SAR patches may not be correctly applied!\n")

    # Create test input
    print("\n[2/3] Creating test input...")
    dummy_image = Image.new('RGB', (336, 336), color='red')
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    
    if processor:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)
    else:
        inputs = tokenizer("Describe this image.", return_tensors="pt", padding=True)
    
    device = model_no_sar.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"✓ Input shape: {inputs['input_ids'].shape}, Image shape: {inputs.get('pixel_values', 'N/A').shape if 'pixel_values' in inputs else 'N/A'}")

    # Run inference
    print("\n[3/3] Running inference and comparing outputs...")
    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": False,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():
        print("  - Generating WITHOUT SAR...")
        outputs_no_sar = model_no_sar.generate(**inputs, **generation_kwargs)
        generated_ids_no_sar, text_no_sar = extract_generated_text(outputs_no_sar, inputs, tokenizer)
        print(f"    Output: {text_no_sar}")

        print("  - Generating WITH SAR...")
        outputs_with_sar = model_with_sar.generate(**inputs, **generation_kwargs)
        generated_ids_with_sar, text_with_sar = extract_generated_text(outputs_with_sar, inputs, tokenizer)
        print(f"    Output: {text_with_sar}")

    # Compare outputs
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    sequences_identical = torch.equal(
        outputs_no_sar.sequences if hasattr(outputs_no_sar, 'sequences') else outputs_no_sar[0],
        outputs_with_sar.sequences if hasattr(outputs_with_sar, 'sequences') else outputs_with_sar[0]
    )
    texts_identical = text_no_sar == text_with_sar
    
    if generated_ids_no_sar.shape == generated_ids_with_sar.shape:
        num_different = (generated_ids_no_sar != generated_ids_with_sar).sum().item()
        total_tokens = generated_ids_no_sar.numel()
        similarity = 1.0 - (num_different / total_tokens) if total_tokens > 0 else 0.0
        print(f"Token differences: {num_different}/{total_tokens} (similarity: {similarity:.2%})")
    else:
        print(f"Token differences: Different lengths ({len(generated_ids_no_sar)} vs {len(generated_ids_with_sar)})")

    print(f"Sequences identical: {sequences_identical}")
    print(f"Texts identical: {texts_identical}")

    if sequences_identical:
        print("\n⚠ WARNING: SAR is NOT affecting model outputs!")
        print("  Possible reasons:")
        print("    1. Input IDs not being passed to SAR context")
        print("    2. Vision encoder hidden states not being captured")
        print("    3. image_token_index not configured in model config")
        print("    4. SAR only active in final layers, may not affect short generations")
        print("    5. Image too simple - similar attention patterns regardless of SAR")
        sar_is_working = False
    else:
        print("\n✓ SUCCESS: SAR is changing model outputs!")
        sar_is_working = True

    print("="*80 + "\n")
    return sar_is_working


if __name__ == "__main__":
    sar_is_working = test_sar_effectiveness()
    sys.exit(0 if sar_is_working else 1)
