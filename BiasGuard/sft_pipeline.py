import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import *

dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")
train_test_split = dataset.train_test_split(test_size=0.1)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"].strip()}],
        "completion": [
            {"role": "assistant", "content": example["response"].strip()}
        ],
    }

train_data = train_data.map(preprocess_function, remove_columns=["response", "conclusion", "label"])

bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weights=True,
    bnb_8bit_compute_dtype=torch.float16,  # or torch.bfloat16 if supported
)
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_NAME,
    device_map="auto", # Automatically distributes the model across available GPUs
    quantization_config = bnb_config_8bit,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
)
trainer.train()

metrics = trainer.evaluate(eval_dataset=test_data)
print(metrics)

trainer.save_model("sft_model")