from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *

model = AutoModelForCausalLM.from_pretrained(DPO_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(DPO_MODEL_NAME)
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")
print(dataset)

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()