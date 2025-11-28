# valeriobasile/HatEval

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from config import *
from tqdm.auto import tqdm
from generate_sft_data import format_prompt, get_system_instruction, batch_generate_responses, BIASED, UNBIASED
from sklearn.metrics import accuracy_score, confusion_matrix


print("Loading dataset...")

ds = load_dataset("valeriobasile/HatEval")["test"]

ds = ds.filter(lambda example: example['language'] == 'es')  # Filter to only English examples

ds = ds.shuffle(seed=42)

dataset = Dataset.from_list(ds)
sys_inst = get_system_instruction()  # Evaluate sys_inst only once
dataset = dataset.rename_column("text", "prompt")
dataset = dataset.rename_column("HS", "prompt_label")

dataset = dataset.map(lambda example: format_prompt(example, sys_inst))

print("Loading BiasGuard model...")

# SFT
# model_name = "ineedausername101/ANLP-BiasGuard-lora-adapter"
# DPO
model_name="/home/rajmahes/ANLP/Reasoning-Bias/BiasGuard/rl_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_4bit=True,
)
model.eval()

batch_size = BATCH_SIZE

y_preds = []
y_labels = []

for i in range(0, len(dataset), batch_size):
    print(f"Progess: {i}/{len(dataset)}")

    batch_results = batch_generate_responses(model, tokenizer, dataset[i:i+batch_size])

    for row in batch_results:
        conclusion_text = row["conclusion"]
        prompt_label = row["label"]

        pred = 0.0 if ((UNBIASED in conclusion_text) and (BIASED not in conclusion_text)) else 1.0 if (BIASED in conclusion_text) and (UNBIASED not in conclusion_text) else None

        if pred is None or prompt_label is None:
            continue
            
        if pred != prompt_label:
            print("Mismatch found!")
            print(f"Prompt: {row['prompt']}")
            print(f"Conclusion: {conclusion_text}")
            print(f"Response: {row['response']}")
            print(f"Predicted: {pred}, Actual: {prompt_label}")

        y_preds += [pred]
        y_labels += [prompt_label]

    print("Calculating metrics...")

    cm = confusion_matrix(y_labels, y_preds)
    acc = accuracy_score(y_labels, y_preds)

    print(f"Evaluations so far: {len(y_labels)}")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)