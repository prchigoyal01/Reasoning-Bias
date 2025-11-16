import os
import json
import pandas as pd

MBBQ_DIR = "../MBBQ_data"
RESULTS_DIR = "results_new"

categories = [
    "Age", "Age_control",
    "Disability_status", "Disability_status_control",
    "Gender_identity", "Gender_identity_control",
    "Physical_appearance", "Physical_appearance_control",
    "SES", "SES_control",
    "Sexual_orientation", "Sexual_orientation_control"
]

eval_modes = ["short_answer", "cot", "reasoning"]
languages = ["en", "es", "tr", "nl"]

# Mapping models to eval modes:
model_for_mode = {
    "short_answer": ["Llama-3.1-8B-Instruct"],
    "cot":          ["Llama-3.1-8B-Instruct"],
    "reasoning":    ["DeepSeek-R1-Distill-Llama-8B"]
}

# Restrict tests:
LLAMA31_SA = "Llama-3.1-8B-Instruct"
LLAMA31_COT = "Llama-3.1-8B-Instruct"
LLAMA2_SA = "Llama-2-7b-chat-hf"
DEEPSEEK_R1 = "DeepSeek-R1-Distill-Llama-8B"

cross_lingual_models = [LLAMA31_SA]     # only SA for cross-lingual
cross_category_models = [LLAMA31_SA]    # only SA for cross-category

# Final rows for CSV
rows = []

for category in categories:
    category_type = "control" if "control" in category.lower() else "non-control"

    for lang in languages:

        mbbq_path = os.path.join(MBBQ_DIR, f"{category}_{lang}.jsonl")
        if not os.path.exists(mbbq_path):
            print(f"[WARN] Missing MBBQ file: {mbbq_path}")
            continue

        # Load MBBQ
        ground_truth = []
        contexts = []

        with open(mbbq_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                ground_truth.append(ex["label"])
                contexts.append(ex["context_condition"])

        total_dataset_size = len(ground_truth)

        # Load predictions
        for eval_mode in eval_modes:
            for model in model_for_mode[eval_mode]:

                # Filtering based on the experimental plan
                include = False

                if eval_mode == "short_answer" and model == LLAMA31_SA:
                    include = True  # for cross-lingual / cross-category

                if not include:
                    continue

                result_filename = f"results_{model}_{lang}_{category}_{eval_mode}.json"
                result_path = os.path.join(RESULTS_DIR, result_filename)

                if not os.path.exists(result_path):
                    print(f"[WARN] Missing results file: {result_path}")
                    continue

                with open(result_path, "r", encoding="utf-8") as f:
                    result_data = json.load(f)

                predictions = result_data["predictions"]

                if len(predictions) != total_dataset_size:
                    print(f"[ERROR] Size mismatch: {result_path}")
                    continue

                # Add unified rows
                for idx, pred in enumerate(predictions):
                    rows.append({
                        "model": model,
                        "eval_mode": eval_mode,
                        "language": lang,
                        "category": category,
                        "category_type": category_type,
                        "context": contexts[idx],  # ambig or disambig
                        "gold": ground_truth[idx],
                        "pred": pred,
                        "item_id": idx
                    })

df = pd.DataFrame(rows)
df.to_csv("kruskal_input_predictions.csv", index=False)

print("\n=== SAVED: kruskal_input_predictions.csv ===")
print(df.head())
print(f"\nTotal rows: {len(df)}")
