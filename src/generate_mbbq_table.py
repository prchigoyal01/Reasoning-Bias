import json
import os
import numpy as np

BASE = "results_mbbq"

MODELS = {
    "Llama-2 SA": ("Llama-2-7b-chat-hf", "short_answer"),
    "Llama-3.1 SA": ("Llama-3.1-8B-Instruct", "short_answer"),
    "Llama-3.1 CoT": ("Llama-3.1-8B-Instruct", "cot"),
    "DeepSeek R1": ("DeepSeek-R1-Distill-Llama-8B", "reasoning"),
}

LANGS = ["en", "es", "nl", "tr"]

def load_metrics(model_id, suffix, lang):
    fname = f"results_{model_id}_{lang}_{suffix}.json"
    path = os.path.join(BASE, fname)

    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None

    with open(path, "r") as f:
        d = json.load(f)

    return {
        "accA": d["accA"] * 100,
        "accD": d["accD"] * 100,
        "overall": (d["accA"] + d["accD"]) / 2 * 100
    }

# Collect metrics
table = {}

for model_name, (mid, suffix) in MODELS.items():
    accA_list, accD_list, overall_list = [], [], []

    for lang in LANGS:
        m = load_metrics(mid, suffix, lang)
        if m is None:
            continue
        accA_list.append(m["accA"])
        accD_list.append(m["accD"])
        overall_list.append(m["overall"])

    # Convert to arrays
    accA_arr = np.array(accA_list)
    accD_arr = np.array(accD_list)
    overall_arr = np.array(overall_list)

    # Format: mean ± std
    table[model_name] = {
        "ACC_D": f"{accD_arr.mean():.1f} ± {accD_arr.std():.2f}",
        "ACC_A": f"{accA_arr.mean():.1f} ± {accA_arr.std():.2f}",
        "Overall": f"{overall_arr.mean():.1f} ± {overall_arr.std():.2f}",
    }

# Pretty print
print("\n" + "-"*60)
print(f"{'Model':<20} {'ACC_D':<18} {'ACC_A':<18} Overall")
print("-"*60)

for model, vals in table.items():
    print(f"{model:<20} {vals['ACC_D']:<18} {vals['ACC_A']:<18} {vals['Overall']}")

print("-"*60)
