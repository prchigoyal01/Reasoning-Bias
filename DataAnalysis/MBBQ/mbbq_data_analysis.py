# data_analysis.py
import os
import json
import pandas as pd

def load_mbbq_data(data_dir="data"):
    data_dir = os.path.abspath(data_dir)
    print(f"Reading from: {data_dir}")
    records = []

    for fname in os.listdir(data_dir):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue

        fpath = os.path.join(data_dir, fname)
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        lang = parts[-1]                # 'en', 'es', 'tr', 'nl
        subset = "_".join(parts[:-1])   # eg: 'Disability_status' / 'Age'

        try:
            n_items = 0
            if fname.endswith(".json"):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_items = len(data)
            elif fname.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    n_items = sum(1 for _ in f)  # each line is an item

            records.append({
                "subset": subset,
                "language": lang,
                "n_items": n_items,
                "file": fname
            })
        except Exception as e:
            print(f"Could not read {fpath}: {e}")

    return pd.DataFrame(records)

def summarize(df):
    print("\n===== MBBQ Dataset Statistics =====")
    print(f"Total subsets: {df['subset'].nunique()}")
    print(f"Total examples: {df['n_items'].sum():,}")

    pivot = df.pivot_table(index="language", columns="subset", values="n_items", aggfunc="sum", fill_value=0)
    print("\nExamples per language × subset:")
    print(pivot)

    df.to_csv("mbbq_data_summary.csv", index=False)
    pivot.to_csv("mbbq_data_pivot.csv")
    print("\nSaved summary to 'mbbq_data_summary.csv' and pivot table to 'mbbq_data_pivot.csv'")


if __name__ == "__main__":
    df = load_mbbq_data("MBBQ/data")
    if df.empty:
        print("No JSON or JSONL files found!")
    else:
        summarize(df)