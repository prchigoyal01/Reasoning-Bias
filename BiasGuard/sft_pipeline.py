import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from config import *

# Load finetune_data.jsonl (your generated data with conclusions)
dataset = load_dataset("json", data_files=FINETUNE_DATA_PATH, split="train")
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

def preprocess_function(example):
    # Keep prompt and completion as separate fields for SFTTrainer
    # SFTTrainer will concatenate them internally
    return {
        "prompt": example["prompt"].strip(),
        "completion": example["response"].strip(),
    }

# Apply preprocessing and remove unused columns for BOTH train and eval
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

trainer = SFTTrainer(
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

# Save the trained model
trainer.save_model("sft_model")