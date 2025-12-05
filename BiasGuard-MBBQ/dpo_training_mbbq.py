"""
DPO Training script for BiasGuard-MBBQ.

Adapted from BiasGuard/rl_training.py for MBBQ dataset.
Trains DPO model on preference pairs generated from SFT model.
"""

import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import DPOConfig, DPOTrainer
from config_mbbq import (
    RL_DATA_PATH, RL_MODEL_PATH, SFT_MODEL_PATH, BASE_MODEL_NAME,
    PROMPT_TEMPLATE_PATH, BIAS_TYPES_PATH, STANDARDS_PATH
)
from generate_mbbq_sft_data import get_mbbq_system_instruction, format_prompt

torch.cuda.empty_cache()

print("="*60)
print("BiasGuard-MBBQ DPO Training")
print("="*60)
print(f"RL Data: {RL_DATA_PATH}")
print(f"SFT Model: {SFT_MODEL_PATH}")
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"Output Path: {RL_MODEL_PATH}")
print("="*60)

# --------------------------
# 1. Preprocess dataset
# --------------------------
# Load system instruction for formatting
sys_inst = get_mbbq_system_instruction()

def preprocess_function(example):
    """Convert to DPO format with formatted prompts (BiasGuard format)."""
    # Format the prompt with system instruction (same as in SFT training)
    formatted_prompt = format_prompt({"prompt": example["prompt"]}, sys_inst)
    prompt_text = formatted_prompt["prompt_text"]
    
    # DPO expects prompt/response pairs, not chat template format
    # The prompt_text already includes [SYSTEM]: ... [USER]: ... [ASSISTANT]: 
    return {
        "prompt": prompt_text,
        "chosen": example["chosen"].strip(),
        "rejected": example["rejected"].strip(),
    }

print("\nLoading RL data...")
dataset = load_dataset("json", data_files=RL_DATA_PATH, split="train")
print(f"Total preference pairs: {len(dataset)}")

print("\nPreprocessing data for DPO...")
dataset = dataset.map(preprocess_function, remove_columns=["label"])

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
test_data = train_test_split["test"]
print(f"Train examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")


# ----------------------------------------------------------
# 2. LOAD SFT LoRA + BASE MODEL
# ----------------------------------------------------------
print("\n" + "="*60)
print("Loading SFT model + adapter...")
print("="*60)
sft_model = AutoPeftModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("SFT model loaded")


print("\n" + "="*60)
print("Merging LoRA into base model...")
print("="*60)
print("This is critical: DPO needs to train on the merged base model, not just the adapter.")
sft_model = sft_model.merge_and_unload()   # <--- CRITICAL STEP
print("✓ LoRA merged into base model")
print("\n" + "="*60)
print("Applying NEW LoRA for DPO...")
print("="*60)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # All attention layers for Llama
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none",
)

model = get_peft_model(sft_model, lora_config)
model.print_trainable_parameters()


print("\n" + "="*60)
print("Loading reference model (frozen)...")
print("="*60)

ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,  # same base model
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("Reference model loaded")

print("\n" + "="*60)
print("Setting up DPO training...")
print("="*60)

training_args = DPOConfig(
    output_dir=RL_MODEL_PATH,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="paged_adamw_8bit",
    fp16=True,
    save_total_limit=3,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,  # Use tokenizer instead of processing_class
    args=training_args,
    max_length=2048,  # Set max length for tokenization
    max_prompt_length=1024,  # Max prompt length
    beta=0.1,  # DPO beta parameter
    loss_type="sigmoid",  # DPO loss type
)

print("\n" + "="*60)
print("Starting DPO training...")
print("="*60)
trainer.train()


print("\n" + "="*60)
print("Saving DPO LoRA adapter...")
print("="*60)
model.save_pretrained(RL_MODEL_PATH)
tokenizer.save_pretrained(RL_MODEL_PATH)

print(f"✓ Model saved to: {RL_MODEL_PATH}")

print("\n" + "="*60)
print("Verifying LoRA adapter file...")
print("="*60)
try:
    from safetensors import safe_open
    import os
    adapter_path = os.path.join(RL_MODEL_PATH, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        with safe_open(adapter_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"✓ Adapter contains {len(keys)} tensors.")
            print(f"✓ File size: {os.path.getsize(adapter_path) / (1024*1024):.1f} MB")
    else:
        print(f"⚠ Warning: adapter_model.safetensors not found at {adapter_path}")
except Exception as e:
    print(f"⚠ Warning: Could not verify saved model: {e}")

print("\n" + "="*60)
print("DPO Training Complete!")
print("="*60)
print(f"\nTo load the model for inference:")
print(f"  from peft import AutoPeftModelForCausalLM")
print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{RL_MODEL_PATH}')")
print(f"\nNext step: Run merge_dpo_model_for_vllm.py to create vLLM-compatible merged model")
print("="*60)

