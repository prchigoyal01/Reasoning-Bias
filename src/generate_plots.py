import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Base directory
BASE = "results_mbbq"

# Models you want to show
MODELS = {
    "Llama-2 SA": "Llama-2-7b-chat-hf",
    "Llama-3.1 SA": "Llama-3.1-8B-Instruct",
    "Llama-3.1 CoT": "Llama-3.1-8B-Instruct",
    "DeepSeek R1": "DeepSeek-R1-Distill-Llama-8B",
}

# File suffixes for each model-mode combination
FILE_SUFFIX = {
    "Llama-2 SA": "short_answer",
    "Llama-3.1 SA": "short_answer",
    "Llama-3.1 CoT": "cot",
    "DeepSeek R1": "reasoning",
}

LANGS = ["en", "es", "nl", "tr"]
LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "nl": "Dutch",
    "tr": "Turkish"
}

COLORS = {
    "Llama-2 SA": "#1f77b4",   # blue
    "Llama-3.1 SA": "#ff7f0e", # orange
    "Llama-3.1 CoT": "#2ca02c",# green
    "DeepSeek R1": "#d62728"   # red
}

def load_acc(model_key, lang):
    model_id = MODELS[model_key]
    suffix = FILE_SUFFIX[model_key]

    # Build file name
    fname = f"results_{model_id}_{lang}_{suffix}.json"
    path = os.path.join(BASE, fname)

    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)
        acc = (data["accD"] + data["accA"])/2 
        print(acc * 100)
        return acc * 100   # convert to percentage like in paper

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, lang in enumerate(LANGS):
    ax = axes[i]

    values = []
    labels = list(MODELS.keys())

    for key in labels:
        acc = load_acc(key, lang)
        if acc is None:
            acc = 0
        values.append(acc)

    colors = [COLORS[m] for m in MODELS.keys()]


    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_title(LANG_NAMES[lang])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(labels, rotation=45, ha="right")

plt.tight_layout()
plt.show()
