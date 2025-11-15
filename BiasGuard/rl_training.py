import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from config import *
from trl import DPOConfig, DPOTrainer

torch.cuda.empty_cache()

def preprocess_function(example):
    return {
        "chosen": [{
            "content": example["prompt"].strip(),
            "role": "user"
        }, {
            "content": example["chosen"].strip(),
            "role": "assistant"
        }],
        "rejected": [{
            "content": example["prompt"].strip(),
            "role": "user"
        }, {
            "content": example["rejected"].strip(),
            "role": "assistant"
        }]
    }

dataset = load_dataset("json", data_files=RL_DATA_PATH, split="train")
dataset = dataset.map(preprocess_function, remove_columns=["prompt", "label"])

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

model = AutoPeftModelForCausalLM.from_pretrained(
    'ineedausername101/ANLP-BiasGuard-lora-adapter',
    device_map='auto',
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained('ineedausername101/ANLP-BiasGuard-lora-adapter')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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

training_args = DPOConfig(
    output_dir=RL_MODEL_PATH,
    num_train_epochs=30,
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

trainer = DPOTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Save the LoRA adapter (not the full model since it's quantized)
print("Saving LoRA adapter...")
model.save_pretrained(RL_MODEL_PATH)
tokenizer.save_pretrained(RL_MODEL_PATH)

print(f"Model saved to {RL_MODEL_PATH}")

# Verify the saved model can be loaded
print("Verifying saved model integrity...")
try:
    from safetensors import safe_open
    import os
    adapter_path = os.path.join(RL_MODEL_PATH, "adapter_model.safetensors")
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"✓ Successfully saved {len(keys)} adapter weights")
        print(f"✓ File size: {os.path.getsize(adapter_path) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"⚠ Warning: Could not verify saved model: {e}")

print("To load: use AutoPeftModelForCausalLM.from_pretrained('" + RL_MODEL_PATH + "')")