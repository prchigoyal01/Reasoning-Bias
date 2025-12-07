"""
SFT Training script for BiasGuard-MBBQ.

Adapted from BiasGuard/sft_training.py for MBBQ dataset.
Uses LoRA for efficient fine-tuning, compatible with vLLM inference.
"""

import torch
import json
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from config_mbbq import (
    SFT_DATA_PATH, SFT_MODEL_PATH, BASE_MODEL_NAME,
    PROMPT_TEMPLATE_PATH, BIAS_TYPES_PATH, STANDARDS_PATH
)

from generate_mbbq_sft_data import get_mbbq_system_instruction, format_prompt

print("="*60)
print("BiasGuard-MBBQ SFT Training")
print("="*60)
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"SFT Data: {SFT_DATA_PATH}")
print(f"Output Path: {SFT_MODEL_PATH}")
print("="*60)


# Load SFT data
print("\nLoading SFT data...")
dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
print(f"Total examples: {len(dataset)}")

# Balance languages (en/tr) to equal counts
print("\nComputing language distribution...")
en_ds = dataset.filter(lambda ex: ex.get("lang", "en") == "en")
tr_ds = dataset.filter(lambda ex: ex.get("lang", "en") == "tr")
print(f"English (en): {len(en_ds)} | Turkish (tr): {len(tr_ds)}")

if len(en_ds) == 0 or len(tr_ds) == 0:
    print("⚠ Warning: One of the languages has zero samples; skipping balancing.")
else:
    min_count = min(len(en_ds), len(tr_ds))
    print(f"Balancing to {min_count} samples per language...")
    en_bal = en_ds.shuffle(seed=42).select(range(min_count))
    tr_bal = tr_ds.shuffle(seed=42).select(range(min_count))
    dataset = concatenate_datasets([en_bal, tr_bal]).shuffle(seed=42)
    print(f"Balanced total examples: {len(dataset)}")

# Split into train/test
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
test_data = train_test_split["test"]
print(f"Train examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

with open("test_samples_mbbq.jsonl", "w", encoding="utf-8") as f:
    for example in test_data:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

def preprocess_function(example):
    # Combine prompt and response into a single text field for SFTTrainer
    return {
        "text": example["prompt"].strip() + "\n\n" + example["response"].strip(),
    }

# Apply preprocessing
print("\nPreprocessing data...")
train_data = train_data.map(
    preprocess_function, 
    remove_columns=["prompt", "response", "conclusion", "label", "category", "lang"]
)
test_data = test_data.map(
    preprocess_function, 
    remove_columns=["prompt", "response", "conclusion", "label", "category", "lang"]
)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load base model with 8-bit quantization for memory efficiency
print(f"\nLoading base model: {BASE_MODEL_NAME}...")
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weights=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",  # Automatically uses CUDA if available
    quantization_config=bnb_config_8bit,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Align tokenizer BOS token ID with model config
if hasattr(model.config, 'bos_token_id') and model.config.bos_token_id is not None:
    tokenizer.bos_token_id = model.config.bos_token_id

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model config: {model.config.model_type}")

# Configure LoRA for efficient fine-tuning
# Using attention modules compatible with Llama architecture
print("\nConfiguring LoRA...")
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # All attention layers for Llama
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    output_dir=SFT_MODEL_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for memory efficiency
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    torch_empty_cache_steps=10,  # Clear cache periodically
    optim="paged_adamw_8bit",  # Use 8-bit optimizer to save memory
    fp16=True,  # Use mixed precision training
    save_total_limit=3,  # Keep only last 3 checkpoints
    report_to="none",  # Disable wandb/tensorboard
)

def formatting_func(example):
    """Format function for SFTTrainer - just return the text."""
    return example["text"]

# Create trainer
print("\nInitializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    formatting_func=formatting_func,
    args=training_args,
)

# Train
print("\n" + "="*60)
print("Starting training...")
print("="*60)
trainer.train()

# Evaluate
print("\n" + "="*60)
print("Evaluating...")
print("="*60)
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Save the LoRA adapter
print("\n" + "="*60)
print("Saving model...")
print("="*60)
model.save_pretrained(SFT_MODEL_PATH)
tokenizer.save_pretrained(SFT_MODEL_PATH)

print(f"\n✓ Model saved to {SFT_MODEL_PATH}")

# Verify the saved model
print("\nVerifying saved model integrity...")
try:
    from safetensors import safe_open
    import os
    adapter_path = os.path.join(SFT_MODEL_PATH, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        with safe_open(adapter_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"✓ Successfully saved {len(keys)} adapter weights")
            print(f"✓ File size: {os.path.getsize(adapter_path) / (1024*1024):.1f} MB")
    else:
        print(f"⚠ Warning: adapter_model.safetensors not found at {adapter_path}")
except Exception as e:
    print(f"⚠ Warning: Could not verify saved model: {e}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nTo load the model for inference:")
print(f"  from peft import AutoPeftModelForCausalLM")
print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{SFT_MODEL_PATH}')")
print(f"\nTo use with vLLM:")
print(f"  from vllm import LLM")
print(f"  # vLLM supports PEFT/LoRA adapters")
print(f"  llm = LLM(model='{BASE_MODEL_NAME}', tensor_parallel_size=1)")
print(f"  # Then load adapter weights separately if needed")
print(f"  # Or merge LoRA weights first for better vLLM compatibility:")
print(f"  # merged_model = model.merge_and_unload()")
print(f"  # merged_model.save_pretrained('merged_model')")
print("="*60)

