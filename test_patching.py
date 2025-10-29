"""
Test script to verify if SAR is actually being applied during model inference.

This script:
1. Loads a model with use_sar=False and use_sar=True
2. Performs a simple forward pass with the same input
3. Checks if the outputs differ
4. Inspects the model architecture to verify SAR patching
"""

import os
import sys
import torch
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.hparams import ModelArguments, FinetuningArguments
from llamafactory.model.loader import load_tokenizer, load_model


def verify_sar_patching(model):
    """Check if SAR has been applied to the model."""
    print("\n" + "="*80)
    print("VERIFYING SAR PATCHING")
    print("="*80)

    # Check 1: Vision encoder hidden state storage
    has_vision_storage = hasattr(model, "_sar_vision_hidden_states")
    print(f" Model has _sar_vision_hidden_states attribute: {has_vision_storage}")

    # Check 2: Decoder layers
    # DEBUG @shulin16
    # import pdb; pdb.set_trace()
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        decoder_layers = model.model.language_model.layers
        print(f" Found {len(decoder_layers)} decoder layers")

        # Check first attention layer
        if len(decoder_layers) > 0:
            first_layer = decoder_layers[0]
            if hasattr(first_layer, "self_attn"):
                attn_module = first_layer.self_attn
                forward_method = attn_module.forward

                # Check if forward method is a bound MethodType (indicating patching)
                from types import MethodType
                is_method_type = isinstance(forward_method, MethodType)
                print(f" Attention forward is MethodType: {is_method_type}")

                # Check the actual function name/signature
                forward_name = getattr(forward_method, "__name__", "unknown")
                print(f"  - Forward method name: {forward_name}")

                # Check if function contains our SAR code
                if hasattr(forward_method, "__code__"):
                    code_vars = forward_method.__code__.co_names
                    has_sar_vars = any("sar" in str(v).lower() for v in code_vars)
                    print(f"  - Contains SAR-related variables: {has_sar_vars}")
            else:
                print(" First layer has no self_attn attribute")
    else:
        print(" Could not find decoder layers")

    # Check 3: Vision tower patching
    if hasattr(model.model, "visual"):
        vision_tower = model.model.visual
        if hasattr(vision_tower, "vision_tower"):
            vision_encoder = visitower.forward
            from types import MethodType
            is_patched = isinstance(vision_encoder, MethodType)
            print(f" Vision encoder forward is patched (MethodType): {is_patched}")
    else:
        print(" Model has no visual attribute")

    print("="*80 + "\n")


def test_sar_effectiveness():
    """Test if SAR actually changes model outputs."""
    print("\n" + "="*80)
    print("TESTING SAR EFFECTIVENESS")
    print("="*80)

    # Model configuration
    # model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # model_name = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load model WITHOUT SAR
    print("\n[1/4] Loading model WITHOUT SAR...")
    model_args_no_sar = ModelArguments(
        model_name_or_path=model_name,
        trust_remote_code=True,
    )
    finetuning_args_no_sar = FinetuningArguments(
        stage="sft",
        use_sar=False,
    )

    tokenizer_module = load_tokenizer(model_args_no_sar)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    model_no_sar = load_model(
        tokenizer=tokenizer,
        model_args=model_args_no_sar,
        finetuning_args=finetuning_args_no_sar,
        is_trainable=False,
    )

    print(" Model without SAR loaded")
    verify_sar_patching(model_no_sar)

    # Load model WITH SAR
    print("\n[2/4] Loading model WITH SAR...")
    model_args_with_sar = ModelArguments(
        model_name_or_path=model_name,
        trust_remote_code=True,
    )
    finetuning_args_with_sar = FinetuningArguments(
        stage="sft",
        use_sar=True,
        sar_activation_layer_k=10,
        sar_beta=0.5,
    )

    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args_with_sar,
        finetuning_args=finetuning_args_with_sar,
        is_trainable=False,
    )
    model.generation_config.use_cache = False
    model.config.output_hidden_states = True
    model.visual.config.output_hidden_states = True
    
    import pdb; pdb.set_trace()
    print(" Model with SAR loaded")
    verify_sar_patching(model)

    # Create a simple test input
    print("\n[3/4] Creating test input...")
    # Create a dummy image
    dummy_image = Image.new('RGB', (336, 336), color='red')

    # Create a simple prompt
    prompt = "Describe this image."

    # Process inputs
    if processor is not None:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)
    else:
        # Fallback for models without processor
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Move to device
    device = model_no_sar.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(" Test input created")

    # Run inference
    print("\n[4/4] Running inference and comparing outputs...")

    with torch.no_grad():
        # Generate with model WITHOUT SAR
        print("  - Generating with model WITHOUT SAR...")
        outputs_no_sar = model_no_sar.generate(
            **inputs,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            max_new_tokens=20,
        )
        import pdb; pdb.set_trace()
        text_no_sar = tokenizer.decode(outputs_no_sar[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    Output: {text_no_sar}")

        # Generate with model WITH SAR
        print("  - Generating with model WITH SAR...")
        outputs_with_sar = model.generate(
            **inputs,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            max_new_tokens=20,
        )
        text_with_sar = tokenizer.decode(outputs_with_sar[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    Output: {text_with_sar}")

    # Compare outputs
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    outputs_identical = torch.equal(outputs_no_sar, outputs_with_sar)
    texts_identical = text_no_sar == text_with_sar

    print(f"Generated token IDs identical: {outputs_identical}")
    print(f"Generated text identical: {texts_identical}")

    if outputs_identical:
        print("\n� WARNING: SAR is NOT affecting model outputs!")
        print("  This indicates that SAR is not being properly applied during generation.")
    else:
        print("\n SUCCESS: SAR is changing model outputs!")
        print("  The outputs differ, indicating SAR is working.")

    print("="*80 + "\n")

    return not outputs_identical


if __name__ == "__main__":
    try:
        sar_is_working = test_sar_effectiveness()
        if sar_is_working:
            print(" SAR test PASSED - SAR is affecting outputs")
            sys.exit(0)
        else:
            print(" SAR test FAILED - SAR is NOT affecting outputs")
            sys.exit(1)
    except Exception as e:
        print(f"\n ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
