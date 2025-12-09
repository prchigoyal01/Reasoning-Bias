import pandas as pd
import re

# Path to your output CSV
csv_path = "/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/MBBQ_RL/adbp_out_rl.csv"  # Change if needed

df = pd.read_csv(csv_path)

def extract_answer(ans):
    if pd.isna(ans):
        return None
    # Try to extract from <answer>...</answer>
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(ans), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    # Fallback: look for yes/no/unknown in the answer string
    ans_lower = str(ans).strip().lower()
    if ans_lower.startswith("yes"):
        return "yes"
    if ans_lower.startswith("no"):
        return "no"
    if "unknown" in ans_lower:
        return "unknown"
    return ans_lower

df['predicted'] = df['answer'].apply(extract_answer)

# Map to binary: 1 = biased, 0 = unbiased, -1 = unknown
def map_to_binary(ans):
    if ans == "yes":
        return 1
    if ans == "no":
        return 0
    if ans == "unknown":
        return -1
    return -1

df['pred_binary'] = df['predicted'].apply(map_to_binary)

# Compare to label
df['correct'] = df['pred_binary'] == df['label']

# Print summary
print("Total examples:", len(df))
print("Mitigation Accuracy: {:.2%}".format(df['correct'].mean()))
print("\nConfusion Matrix:")
print(pd.crosstab(df['label'], df['pred_binary'], rownames=['True'], colnames=['Pred']))

# Show a few mismatches
print("\nSample mismatches:")
print(df[df['correct'] == False][['example_id', 'answer', 'label', 'pred_binary']].head())

# Optionally, save detailed results
df.to_csv("adbp_eval_detailed.csv", index=False)