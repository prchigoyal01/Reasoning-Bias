import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from config import *

# Load sft_data.jsonl (your generated data with conclusions)
dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

def preprocess_function(example):
    # Combine prompt and response into a single text field for SFTTrainer
    return {
        "text": example["prompt"].strip() + "\n\n" + example["response"].strip(),
    }

# Apply preprocessing and keep only the text field
train_data = train_data.map(preprocess_function, remove_columns=["prompt", "response", "conclusion", "label"])
test_data = test_data.map(preprocess_function, remove_columns=["prompt", "response", "conclusion", "label"])

bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weights=True,
    bnb_8bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config_8bit,
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Align tokenizer BOS token ID with model config
tokenizer.bos_token_id = model.config.bos_token_id

# Configure LoRA for efficient fine-tuning on quantized model
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced from 4 to fit in memory
    per_device_eval_batch_size=1,   # Reduced from 4
    gradient_accumulation_steps=4,  # Accumulate gradients (effective batch size = 4)
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    torch_empty_cache_steps=10,  # Clear cache periodically
    optim="paged_adamw_8bit",  # Use 8-bit optimizer to save memory
)

def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    args=training_args,
)
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Save the LoRA adapter (not the full model since it's quantized)
print("Saving LoRA adapter...")
model.save_pretrained(SFT_MODEL_PATH)
tokenizer.save_pretrained(SFT_MODEL_PATH)

print(f"Model saved to {SFT_MODEL_PATH}")

# Verify the saved model can be loaded
print("Verifying saved model integrity...")
try:
    from safetensors import safe_open
    import os
    adapter_path = os.path.join(SFT_MODEL_PATH, "adapter_model.safetensors")
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"✓ Successfully saved {len(keys)} adapter weights")
        print(f"✓ File size: {os.path.getsize(adapter_path) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"⚠ Warning: Could not verify saved model: {e}")

print("To load: use AutoPeftModelForCausalLM.from_pretrained('" + SFT_MODEL_PATH + "')")