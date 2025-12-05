"""
Generate RL training data for BiasGuard-MBBQ.

Uses the trained SFT model to generate preference pairs (chosen/rejected),
mirroring the original BiasGuard RL data script.
"""

import json
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from config_mbbq import (
    SFT_MODEL_PATH, RL_DATA_PATH, BATCH_SIZE,
    SFT_DATA_PATH
)
from generate_mbbq_sft_data import get_mbbq_system_instruction, format_prompt

BIASED = "Yes, the sentence is biased."
UNBIASED = "No, the sentence is unbiased."


def batch_generate_responses(model, tokenizer, batch):
    """Generate responses for a batch of prompts using transformers (fallback)."""
    results = []
    inputs = tokenizer(
        batch["prompt_text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ).to(model.device)

    input_lens = [len(prompt) for prompt in inputs['input_ids']]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Same as original BiasGuard
            temperature=0.9,  # Higher temperature for diversity in RL data
            do_sample=True,
        )
    decoded = tokenizer.batch_decode(
        [out[input_lens[i]:] for i, out in enumerate(outputs)],
        skip_special_tokens=True
    )

    for prompt, prompt_label, output in zip(batch['prompt'], batch['prompt_label'], decoded):
        # Extract the final conclusion in the output
        conclusion_matches = re.findall(r"## Conclusion ##:\s*(.*)", output)
        if conclusion_matches:
            conclusion = conclusion_matches[-1].strip()
        else:
            # Fallback: look for conclusion markers
            if BIASED in output:
                conclusion = BIASED
            elif UNBIASED in output:
                conclusion = UNBIASED
            else:
                conclusion = ""
        
        results.append({
            "prompt": prompt,
            "label": prompt_label,
            "response": output,
            "conclusion": conclusion
        })
    return results


def _is_inappropriate_conclusion(prompt_label, conclusion_text, output_text):
    """Check basic quality: require multi-step reasoning similar to BiasGuard toxigen."""
    required_steps = ["Step " + str(i) for i in range(1, 5)]
    
    # Only enforce minimum reasoning length/steps (BiasGuard toxigen heuristic)
    step_count = sum(1 for step in required_steps if step in output_text)
    if step_count < 2:
        return True  # Too few reasoning steps
    
    if len(output_text.strip()) < 100:
        return True  # Output too short to be useful
    
    return False


def _generate_preference_pairs(model, tokenizer, batch, batch_num=None):
    """Generate preference pairs (chosen/rejected) for a batch of prompts.
    
    We generate `per_try` samples for each input in the batch using a single model call.
    This is done by repeating each input `per_try` times and generating all at once.
    
    Args:
        model: AutoPeftModelForCausalLM used for sampling
        tokenizer: Tokenizer
        batch: Batch of prompts
        batch_num: Batch number for logging
    """
    per_try = 4  # samples to generate per input in a single call
    results = []
    
    batch_size = len(batch["prompt"])
    batch_label = f"Batch {batch_num}" if batch_num is not None else "Batch"
    
    print(f"\n{batch_label}: Processing {batch_size} prompts...")

    for idx, (prompt, prompt_text, label) in enumerate(zip(batch["prompt"], batch["prompt_text"], batch["prompt_label"])):
        minibatch = {
            "prompt": [prompt] * per_try,
            "prompt_label": [label] * per_try,
            "prompt_text": [prompt_text] * per_try,
        }
        
        outputs = batch_generate_responses(model, tokenizer, minibatch)
        
        chosen = None
        wrong_responses = []
        correct_count = 0
        
        for out in outputs:
            response = out["response"]
            conclusion = out["conclusion"]


            if not _is_inappropriate_conclusion(label, conclusion, response):
                correct_count += 1
                if chosen is None:  # Take first correct one as chosen
                    chosen = response
            else:
                wrong_responses.append(response)
        
        # Detailed logging per prompt
        print(f"  Prompt {idx+1}/{batch_size} (label={label}): {correct_count} correct, {len(wrong_responses)} wrong out of {per_try}")

        # Create preference pairs: one chosen vs each rejected
        if chosen is not None:
            pairs_for_prompt = len(wrong_responses)
            for rejected in wrong_responses:
                results.append({
                    "prompt": prompt,
                    "label": label,
                    "chosen": chosen,
                    "rejected": rejected
                })
            print(f"Created {pairs_for_prompt} preference pairs for this prompt")
        else:
            print(f"Warning: No correct response found, skipping this prompt")
    
    print(f"{batch_label}: Generated {len(results)} total preference pairs from {batch_size} prompts")
    return results


def generate_rl_data_mbbq():
    """Main function to generate RL data from MBBQ using SFT data + model."""
    
    print("="*60)
    print("Generating RL Data for BiasGuard-MBBQ")
    print("="*60)
    
    print(f"\nLoading SFT model from {SFT_MODEL_PATH}...")

    model = AutoPeftModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    sys_inst = get_mbbq_system_instruction()
    
    print("\nLoading prompts from SFT data...")
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    
    def prepare_prompt(example):
        formatted = format_prompt({"prompt": example["prompt"]}, sys_inst)
        example["prompt_text"] = formatted["prompt_text"]
        example["prompt_label"] = example["label"]
        return example
    
    dataset = dataset.map(prepare_prompt)
    
    # Generate preference pairs
    print(f"\nGenerating preference pairs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Samples per prompt: 4")
    
    batch_size = BATCH_SIZE
    out_path = RL_DATA_PATH
    
    # Clear output file if it exists
    with open(out_path, "w") as f:
        pass  # Create empty file
    
    total_pairs = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Total batches to process: {total_batches}")
    print(f"Total prompts: {len(dataset)}")
    print("="*60)
    
    with open(out_path, "a") as f:
        for batch_idx, i in enumerate(tqdm(range(0, len(dataset), batch_size), desc="Processing batches"), 1):
            batch = dataset[i : i + batch_size]
            pairs = _generate_preference_pairs(model, tokenizer, batch, batch_num=batch_idx)
            
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_pairs += 1
            
            # Progress summary every 10 batches
            if batch_idx % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {batch_idx}/{total_batches} batches completed")
                print(f"Total preference pairs so far: {total_pairs}")
                print(f"Average pairs per batch: {total_pairs / batch_idx:.1f}")
                print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"Generated {total_pairs} preference pairs total")
    print(f"Saved to {out_path}")
    print(f"Average pairs per batch: {total_pairs / total_batches:.1f}")
    print(f"Success rate: {total_pairs / (len(dataset) * 8) * 100:.2f}% (pairs / total_samples)")
    print(f"{'='*60}")
    
    print("="*60)


if __name__ == "__main__":
    generate_rl_data_mbbq()

