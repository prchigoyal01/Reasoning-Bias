"""
Merge DPO LoRA adapter into base model for vLLM compatibility.

This script merges the DPO LoRA weights into the base model, creating
a single merged model that can be used directly with vLLM for efficient inference.
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from config_mbbq import RL_MODEL_PATH, BASE_MODEL_NAME

# Output path for merged model (relative to script location)
MERGED_MODEL_PATH = "mbbq_rl_model_merged"

print("="*60)
print("Merging DPO Model for vLLM Compatibility")
print("="*60)
print(f"DPO Model: {RL_MODEL_PATH}")
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"Output Path: {MERGED_MODEL_PATH}")
print("="*60)

# Check if DPO model exists
if not os.path.exists(RL_MODEL_PATH):
    raise FileNotFoundError(f"DPO model not found at {RL_MODEL_PATH}. Please run dpo_training_mbbq.py first.")

# ----------------------------------------------------------
# 1. Load DPO model (with LoRA adapter)
# ----------------------------------------------------------
print("\n" + "="*60)
print("Loading DPO model with LoRA adapter...")
print("="*60)

dpo_model = AutoPeftModelForCausalLM.from_pretrained(
    RL_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(RL_MODEL_PATH)

print("✓ DPO model loaded")


# ----------------------------------------------------------
# 2. Merge LoRA into base model
# ----------------------------------------------------------
print("\n" + "="*60)
print("Merging LoRA adapter into base model...")
print("="*60)
print("This will create a single merged model that can be used with vLLM.")

merged_model = dpo_model.merge_and_unload()

print("✓ LoRA merged into base model")


# ----------------------------------------------------------
# 3. Save merged model
# ----------------------------------------------------------
print("\n" + "="*60)
print("Saving merged model...")
print("="*60)

# Create output directory
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)

# Save model and tokenizer
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"✓ Merged model saved to {MERGED_MODEL_PATH}")


# ----------------------------------------------------------
# 4. Verify merged model can be loaded
# ----------------------------------------------------------
print("\n" + "="*60)
print("Verifying merged model...")
print("="*60)

try:
    # Try loading with AutoModelForCausalLM (required for vLLM)
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    test_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    
    print("✓ Merged model can be loaded with AutoModelForCausalLM")
    print("✓ Model is ready for vLLM inference")
    
    # Check model size
    total_params = sum(p.numel() for p in test_model.parameters())
    trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params / 1e9:.2f}B")
    print(f"✓ Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Clean up test model
    del test_model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"⚠ Warning: Could not verify merged model: {e}")
    raise


# ----------------------------------------------------------
# 5. Print usage instructions
# ----------------------------------------------------------
print("\n" + "="*60)
print("Merge Complete!")
print("="*60)
print(f"\nMerged model saved to: {MERGED_MODEL_PATH}")
print(f"\nTo use with vLLM:")
print(f"  from vllm import LLM, SamplingParams")
print(f"  ")
print(f"  llm = LLM(")
print(f"      model='{os.path.abspath(MERGED_MODEL_PATH)}',")
print(f"      tensor_parallel_size=1,")
print(f"      dtype='float16',")
print(f"      trust_remote_code=True")
print(f"  )")
print(f"  ")
print(f"  sampling_params = SamplingParams(")
print(f"      temperature=0.7,")
print(f"      max_tokens=512")
print(f"  )")
print(f"  ")
print(f"  outputs = llm.generate(prompts, sampling_params)")
print(f"\nTo use with transformers:")
print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"  ")
print(f"  model = AutoModelForCausalLM.from_pretrained('{MERGED_MODEL_PATH}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{MERGED_MODEL_PATH}')")
print("="*60)

