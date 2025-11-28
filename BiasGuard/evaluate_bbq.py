# evaluate_toxicity.py

import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.auto import tqdm

from config import *
from generate_sft_data import format_prompt, get_system_instruction, batch_generate_responses, BIASED, UNBIASED


print("Loading dataset...")

ds_age = load_dataset("walledai/BBQ")["age"]
ds_gender = load_dataset("walledai/BBQ")["genderIdentity"]
ds_religion = load_dataset("walledai/BBQ")["religion"]
ds = concatenate_datasets([ds_age, ds_gender, ds_religion])
ds = ds.shuffle(seed=42)
dataset = Dataset.from_list(ds)

sys_inst = get_system_instruction()  # Evaluate sys_inst only once

def expand_to_three_choices(example):
    """
    Expands each example into 3 rows - one for each choice.
    The correct answer gets label 0 (unbiased).
    The other 2 choices get label 1 (biased).
    """
    expanded_examples = []
    answer_key = example['answer']
    
    for i, choice in enumerate(example['choices']):
        # Find the other choice (assuming 2 choices)
        other_choice = example['choices'][1 - i] if i <= 1 else None
        
        prompt = (
            f"{example['context']}\n"
            f"{choice} is the person who {example['question'][3:-1]}\n"
        )
        
        if other_choice:
            prompt += f"{other_choice} is not the person who {example['question'][3:-1]}\n"
        
        label = 0 if i == answer_key else 1
        expanded_examples.append({
            "prompt": prompt,
            "prompt_label": label,
        })
    
    return expanded_examples

# Expand dataset to include all 3 choices per example
expanded_data = []
for example in dataset:
    expanded_data.extend(expand_to_three_choices(example))
dataset = Dataset.from_list(expanded_data)

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
    torch_dtype=torch.float16
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