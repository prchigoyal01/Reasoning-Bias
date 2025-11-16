#!/usr/bin/env python3
import json
import pandas as pd
import os

results_dir = 'llm_as_judge_results'
categories = ['Age', 'Disabilitystatus', 'Genderidentity', 'Nationality', 
              'Physicalappearance', 'Raceethnicity', 'Religion', 'Ses', 'Sexualorientation']

table_data = []

for cat in categories:
    row = {'Category': cat}
    
    for condition in ['equal_equal', 'equal_not_equal']:
        pattern = f'{cat}_llama3_8b_deepseek_8b_{condition}_llama_2_7b_chat_hf_single_eval_original.jsonl'
        filepath = os.path.join(results_dir, pattern)
        
        if not os.path.exists(filepath):
            continue
        
        ambig_scores = []
        disambig_scores = []
        
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                context_cond = data.get('context_condition', '')
                judge_responses = data.get('judge_responses', [])
                
                scores = [r['bias_score'] for r in judge_responses if 'bias_score' in r]
                if scores:
                    avg_bias = sum(scores) / len(scores)
                    
                    if context_cond == 'ambig':
                        ambig_scores.append(avg_bias)
                    elif context_cond == 'disambig':
                        disambig_scores.append(avg_bias)
        
        if condition == 'equal_equal':
            row['Ambig Correct'] = f'{sum(ambig_scores)/len(ambig_scores):.2f} (n={len(ambig_scores)})' if ambig_scores else 'N/A'
            row['Disambig Correct'] = f'{sum(disambig_scores)/len(disambig_scores):.2f} (n={len(disambig_scores)})' if disambig_scores else 'N/A'
        else:
            row['Ambig Incorrect'] = f'{sum(ambig_scores)/len(ambig_scores):.2f} (n={len(ambig_scores)})' if ambig_scores else 'N/A'
            row['Disambig Incorrect'] = f'{sum(disambig_scores)/len(disambig_scores):.2f} (n={len(disambig_scores)})' if disambig_scores else 'N/A'
    
    table_data.append(row)

df = pd.DataFrame(table_data)

for col in ['Ambig Correct', 'Ambig Incorrect', 'Disambig Correct', 'Disambig Incorrect']:
    if col not in df.columns:
        df[col] = 'N/A'

df = df[['Category', 'Ambig Correct', 'Ambig Incorrect', 'Disambig Correct', 'Disambig Incorrect']]

output_path = '../../results_and_analysis/bias_by_context_condition.csv'
df.to_csv(output_path, index=False)
print(f'Saved to: {output_path}\n')
print(df.to_string(index=False))
