from datasets import load_dataset
import os

ds = load_dataset("walledai/BBQ")
os.makedirs("bbq_csv", exist_ok=True)

for subset_name, subset in ds.items():
    subset.to_csv(f"bbq_csv/{subset_name}.csv")