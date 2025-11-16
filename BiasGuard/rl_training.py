import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from config import *
from trl import DPOConfig, DPOTrainer

torch.cuda.empty_cache()

# --------------------------
# 1. Preprocess dataset
# --------------------------
def preprocess_function(example):
    # Keep your original preprocessing EXACTLY the same
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


# ----------------------------------------------------------
# 2. LOAD SFT LoRA + BASE MODEL (correctly)
# ----------------------------------------------------------
# IMPORTANT: AutoPeftModelForCausalLM loads base model automatically 
# if your adapter_config.json has base_model_name_or_path set correctly.

print("Loading SFT model + adapter...")
sft_model = AutoPeftModelForCausalLM.from_pretrained(
    "ineedausername101/ANLP-BiasGuard-lora-adapter",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("ineedausername101/ANLP-BiasGuard-lora-adapter")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ----------------------------------------------------------
# 3. MERGE LoRA → base model (critical fix)
# ----------------------------------------------------------
print("Merging LoRA into base model...")
sft_model = sft_model.merge_and_unload()   # <--- FIX


# ----------------------------------------------------------
# 4. Create DPO LoRA on top of merged base model
# ----------------------------------------------------------
print("Applying NEW LoRA for DPO...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none",
)

model = get_peft_model(sft_model, lora_config)
model.print_trainable_parameters()


# ----------------------------------------------------------
# 5. Create frozen reference model  (Required for DPO)
# ----------------------------------------------------------
print("Loading reference model (frozen)...")

ref_model = AutoModelForCausalLM.from_pretrained(
    tokenizer.name_or_path,  # same base model
    device_map="auto",
    torch_dtype=torch.float16
)


# ----------------------------------------------------------
# 6. DPO Training
# ----------------------------------------------------------
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
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train()


# ----------------------------------------------------------
# 7. Save DPO LoRA adapter
# ----------------------------------------------------------
print("Saving DPO LoRA adapter...")
model.save_pretrained(RL_MODEL_PATH)
tokenizer.save_pretrained(RL_MODEL_PATH)

print(f"Model saved to: {RL_MODEL_PATH}")

# ----------------------------------------------------------
# 8. Validate saved adapter
# ----------------------------------------------------------
print("Verifying LoRA adapter file...")
try:
    from safetensors import safe_open
    import os
    adapter_path = os.path.join(RL_MODEL_PATH, "adapter_model.safetensors")
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"✓ Adapter contains {len(keys)} tensors.")
except Exception as e:
    print(f"Warning: {e}")

print(f"To load later: AutoPeftModelForCausalLM.from_pretrained('{RL_MODEL_PATH}')")