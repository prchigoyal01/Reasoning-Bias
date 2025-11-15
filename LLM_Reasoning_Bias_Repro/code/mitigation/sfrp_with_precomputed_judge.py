import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import copy
import argparse
import re

parser = argparse.ArgumentParser(
    description="SFRP using pre-computed LLM judge scores"
)

parser.add_argument("cot_jsonfilename", type=str, help="CoT responses JSONL")
parser.add_argument("judge_jsonfilename", type=str, help="Pre-computed judge scores JSONL")
parser.add_argument("outputfilename", type=str, help="Output CSV file")
parser.add_argument("reasoning_modelid", type=str, help="Reasoning model ID")
parser.add_argument("gpu", type=int, help="GPU device ID")

args = parser.parse_args()

# Load data
print("Loading CoT responses...")
import json
cot_records = []
with open(args.cot_jsonfilename, 'r') as f:
    for line in f:
        if line.strip():
            cot_records.append(json.loads(line))
df_cot = pd.DataFrame(cot_records)

print("Loading judge scores...")
judge_records = []
with open(args.judge_jsonfilename, 'r') as f:
    for line in f:
        if line.strip():
            judge_records.append(json.loads(line))
df_judge = pd.DataFrame(judge_records)

# Merge on example_id
print("Merging datasets...")
df = pd.merge(df_cot, df_judge, on='example_id', how='inner', suffixes=('_cot', '_judge'))
print(f"Merged {len(df)} examples")

# Use label from CoT file (should be consistent)
if 'label_cot' in df.columns:
    df['label'] = df['label_cot']
elif 'label_judge' in df.columns:
    df['label'] = df['label_judge']

# Load reasoning model with 8-bit quantization
print("Loading reasoning model...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    args.reasoning_modelid,
    device_map=f'cuda:{args.gpu}',
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(args.reasoning_modelid)

def generate_response(messages, max_new_tokens=512):
    """Generate response from model."""
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_answer(text):
    """Extract answer from <answer>...</answer> tags."""
    match = re.findall(r"<answer>(.*?)</answer>", text)
    if match:
        return match[-1].strip()
    return ""

def filter_biased_paragraphs(cot_response, paragraph_scores):
    """
    Filter out biased paragraphs using pre-computed judge scores.
    paragraph_scores is a list of dicts with 'paragraph' and 'score' keys.
    Keep only paragraphs with score == 0 (unbiased).
    """
    filtered_paragraphs = []
    bias_scores = []
    
    for item in paragraph_scores:
        score = item.get('score', 0)
        paragraph = item.get('paragraph', '')
        
        bias_scores.append(score)
        
        # Keep only unbiased paragraphs (score == 0)
        if score == 0 and paragraph.strip():
            filtered_paragraphs.append(paragraph)
    
    return '\n\n'.join(filtered_paragraphs), bias_scores

# Open output file
import csv
output_file = open(args.outputfilename, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(output_file)
csv_writer.writerow(['example_id', 'original_answer', 'sfrp_answer', 'label', 
                     'num_original_paragraphs', 'num_filtered_paragraphs', 
                     'bias_scores', 'accuracy_improvement'])
output_file.flush()

# Process each example
correct_original = 0
correct_sfrp = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    example_id = row['example_id']
    label = row['label']
    
    # Handle potential column suffix from merge
    original_answer = row.get('final_answer', row.get('final_answer_cot', ''))
    
    # Get paragraph scores from judge results (should be in judge_responses)
    judge_responses = row.get('judge_responses', [])
    
    # Extract paragraph scores from judge_responses
    paragraph_scores = []
    if isinstance(judge_responses, list):
        for resp in judge_responses:
            if isinstance(resp, dict):
                paragraph_scores.append({
                    'paragraph': resp.get('reason_step', ''),
                    'score': resp.get('bias_score', 0)
                })
    
    # Filter biased paragraphs
    cot_response = row.get('cot_response_v1', row.get('cot_response_v1_cot', ''))
    filtered_reasoning, bias_scores = filter_biased_paragraphs(
        cot_response, 
        paragraph_scores
    )
    
    # Count paragraphs
    num_original = len(paragraph_scores)
    num_filtered = len([s for s in bias_scores if s == 0])
    
    # Generate answer using filtered reasoning (SFRP)
    if filtered_reasoning.strip():
        prompt = row.get('prompt', row.get('prompt_cot', []))
        if prompt:
            prompt = copy.deepcopy(prompt)
            prompt[0]['content'] = prompt[0]['content'] + "\n\n" + filtered_reasoning + "\n\nBased on the above unbiased reasoning, provide your final answer."
            
            response = generate_response(prompt)
            sfrp_answer = get_answer(response)
            
            if not sfrp_answer:
                sfrp_answer = original_answer
        else:
            sfrp_answer = original_answer
    else:
        # All paragraphs were biased, use original
        sfrp_answer = original_answer
    
    # Track accuracy
    if original_answer == f"ans{label}":
        correct_original += 1
    if sfrp_answer == f"ans{label}":
        correct_sfrp += 1
    
    accuracy_improvement = (sfrp_answer == f"ans{label}") and (original_answer != f"ans{label}")
    
    # Write results
    csv_writer.writerow([
        example_id,
        original_answer,
        sfrp_answer,
        label,
        num_original,
        num_filtered,
        str(bias_scores),
        accuracy_improvement
    ])
    output_file.flush()

output_file.close()

# Print summary
print(f"\n✅ SFRP results saved to {args.outputfilename}")
print(f"\n📊 Accuracy Summary:")
print(f"Original accuracy: {correct_original}/{len(df)} ({correct_original/len(df)*100:.2f}%)")
print(f"SFRP accuracy: {correct_sfrp}/{len(df)} ({correct_sfrp/len(df)*100:.2f}%)")
print(f"Improvement: {correct_sfrp - correct_original} examples")
