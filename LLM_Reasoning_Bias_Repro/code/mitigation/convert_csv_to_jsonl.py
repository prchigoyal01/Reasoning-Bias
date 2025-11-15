#!/usr/bin/env python3
"""
Convert CoT CSV outputs to JSONL format for ADBP script.
Reconstructs the prompt field needed by ADBP.
"""

import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description="Convert CSV to JSONL for ADBP")
parser.add_argument("input_csv", type=str, help="Input CSV file path")
parser.add_argument("output_jsonl", type=str, help="Output JSONL file path")
args = parser.parse_args()

# Read CSV
df = pd.read_csv(args.input_csv)

# Convert to JSONL format
with open(args.output_jsonl, 'w') as f:
    for _, row in df.iterrows():
        # Reconstruct prompt in chat format
        prompt_content = f"""Context: {row['context']}
Question: {row['question']}
Choices: "ans0": {row['ans0']}, "ans1": {row['ans1']}, "ans2": {row['ans2']}
Provide your response with <answer>ansX</answer> where X is 0, 1, or 2."""
        
        prompt = [{"role": "user", "content": prompt_content}]
        
        # Create JSONL entry with required fields
        entry = {
            "example_id": row['example_id'],
            "label": int(row['label']),
            "cot_response_v1": row['deepseek_8b_cot_response_v1'],
            "final_answer": row['pred_ans_tag'],
            "prompt": prompt,
            # Keep extra fields for reference
            "context": row['context'],
            "question": row['question'],
            "ans0": row['ans0'],
            "ans1": row['ans1'],
            "ans2": row['ans2']
        }
        
        f.write(json.dumps(entry) + '\n')

print(f"Converted {len(df)} examples from {args.input_csv} to {args.output_jsonl}")
