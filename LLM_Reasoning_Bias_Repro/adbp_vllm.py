import pandas as pd
import torch
from tqdm import tqdm
import copy 
import statistics
import argparse
import re
from vllm import LLM, SamplingParams

def complete_chat0_vllm(llm, messages, sampling_params):
    prompt_text = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt_text += f"{role}: {content}\n"
    outputs = llm.generate([prompt_text], sampling_params)
    out = next(iter(outputs))
    return out.text

parser = argparse.ArgumentParser(
    description="Script to mitigate social bias from LLM reasoning using ADBP"
)

# Path to the input JSON file containing data to process
parser.add_argument(
    "jsonfilename", 
    type=str, 
    help="Path to the input JSON file containing questions"
)

# Path where the output (e.g. model responses) will be saved
parser.add_argument(
    "outputfilename", 
    type=str, 
    help="Path to the output CSV file to store the new answers under ADBP"
)

# Identifier of the language model to use (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
parser.add_argument(
    "modelid", 
    type=str, 
    help="The model ID or path to the Reasoning-based LLM"
)

# GPU device ID to run the model on
parser.add_argument(
    "gpu", 
    type=int, 
    help="GPU device ID"
)

# Get input arguments
args = parser.parse_args()

import os
device = torch.device('cuda:'+ str(args.gpu))
# Check if input file is empty
if os.path.getsize(args.jsonfilename) == 0:
    raise ValueError(f"Input file {args.jsonfilename} is empty. Please check your data source.")
try:
    df = pd.read_json(args.jsonfilename, lines=True)
except ValueError as e:
    raise ValueError(f"Failed to read {args.jsonfilename} as JSONL. Ensure it is valid JSONL (one JSON object per line) and not empty. Original error: {e}")

model_name = args.modelid
llm = LLM(model=model_name)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
)

## Remove HuggingFace complete_chat0 (not needed)
def get_answer(text):
    """
    Extracts the last occurrence of text enclosed in <answer>...</answer> tags.

    Parameters:
    - text: A string potentially containing <answer>...</answer> tags.

    Returns:
    - The content inside the last <answer>...</answer> tag, or an empty string if none are found.
    """

    match = re.findall(r"<answer>(.*?)</answer>", text)
    if match:
        result = match[-1]
        return result
    else:
        return ""

# Start of ADBP Algorithm
example_ids = [] # Example ID from BBQ dataset
labels = [] # Ground-truth answers
answer_fix = [] # Answers under ADBP
logss = [] # Model's reasoning under various incremental steps
new_log = [] # Model's final reasoning between two choices


marker = "\nProvide explanation based on known facts or Stay unbiased if no facts are known.\n"

# Batching parameters
BATCH_SIZE = 16  # Suitable for 40GB GPU


# --- Batching Implementation ---
def batch_inference(prompts, llm, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    # Each output may have multiple completions; outputs[0].text for single completion
    return [out.outputs[0].text for out in outputs]

for start_idx in tqdm(range(0, df.shape[0], 1)):
    row = df.iloc[start_idx]
    example_ids.append(row['example_id'])
    thoughts = row['cot_response_v1'].split('\n\n')
    labels.append(row['label'])
    cands = []
    logs = []

    # --- Batch reasoning steps ---
    reasoning_prompts = []
    for i in range(len(thoughts)):
        prompt = copy.deepcopy(row['prompt'])
        prompt[0]['content'] = prompt[0]['content'] + " ".join(thoughts[:i]) + marker
        # Convert to prompt string for vLLM
        prompt_text = ""
        for m in prompt:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_text += f"{role}: {content}\n"
        reasoning_prompts.append(prompt_text)

    # Run all reasoning steps in a batch
    batch_size = BATCH_SIZE
    batch_outputs = []
    for batch_start in range(0, len(reasoning_prompts), batch_size):
        batch_prompts = reasoning_prompts[batch_start:batch_start+batch_size]
        batch_outputs.extend(batch_inference(batch_prompts, llm, sampling_params))

    for response in batch_outputs:
        answer = get_answer(response).strip()
        cands.append(answer)
        logs.append(response)

    logss.append(logs)
    new_cands = [x for x in cands if x != ""]

    if len(new_cands) == 0:
        finalanswer = row['final_answer']
    elif len(new_cands) == 1:
        finalanswer = new_cands[0]
    else:
        changes = 0
        for i in range(len(new_cands)-1):
            if new_cands[i] != new_cands[i+1]:
                changes += 1
        if changes == 0:
            finalanswer = new_cands[0]
        else:
            answer1 = cands[-1]
            new_cands2 = [x for x in new_cands if x != answer1]
            answer2 = statistics.mode(new_cands2)
            index1 = cands.index(answer1)
            if marker in logs[0]:
                split_log = logs[0].split(marker)
                firstreasoning = split_log[1] if len(split_log) > 1 else ""
            else:
                firstreasoning = ""
            if index1 == 0:
                reasoning1 = [firstreasoning]
            else:
                reasoning1 = [thoughts[index1-1]]
            index2 = cands.index(answer2)
            if index2 == 0:
                reasoning2 = [firstreasoning]
            else:
                reasoning2 = [thoughts[index2-1]]
            prompt = copy.deepcopy(row['prompt'])
            prompt[0]['content'] = prompt[0]['content'] + "\nPreviously you are hesitant between these two choices: " + answer1 + " and " + answer2 + " You picked " + answer1 + " becasue of " + str(reasoning1) + " You picked " + answer2 + " because of " + str(reasoning2) + ". Verify them to see if there is any bias and output the answer."
            # Convert to prompt string for vLLM
            prompt_text = ""
            for m in prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_text += f"{role}: {content}\n"
            # Run candidate comparison as a batch of 1 (could batch if multiple rows)
            response = batch_inference([prompt_text], llm, sampling_params)[0]
            finalanswer = get_answer(response).strip()
            new_log.append(response)
    answer_fix.append(finalanswer)
    if len(new_log) != len(example_ids):
        new_log.append('')
    if (start_idx + 1) % 10 == 0:
        results = pd.DataFrame({
            'example_id': example_ids,
            'answer': answer_fix,
            'label': labels,
            'logs': logss,
            'new_log': new_log
        })
        results.to_csv(args.outputfilename, index=False)

# Final save after all iterations
results = pd.DataFrame({
    'example_id': example_ids,
    'answer': answer_fix,
    'label': labels,
    'logs': logss,
    'new_log': new_log
})
results.to_csv(args.outputfilename, index=False)