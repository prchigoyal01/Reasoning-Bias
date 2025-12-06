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
    SFT_DATA_PATH, BASE_MODEL_NAME
)
from generate_mbbq_sft_data import get_mbbq_system_instruction, format_prompt, init_vllm_model, load_biasguard_dataset, _is_wrong_conclusion, batch_generate_responses

STEP_NAME_BY_LANG = {
    "en": "Step",
    "tr": "Adım",
}

def _is_inappropriate_conclusion(output_text, lang="en"):
    """Check basic quality: require multi-step reasoning similar to BiasGuard toxigen."""
    required_steps = [STEP_NAME_BY_LANG[lang] + " " + str(i) for i in range(1, 5)]
    
    for step in required_steps:
        if step not in output_text:
            # print(f"Missing required step in conclusion: {step}")
            return True
    
    # Don't add more than 1 conclusion
    # Don't do analysis after conclusion

    if len(output_text.strip()) < 100:
        return True  # Model may have skipped reasoning steps, too short
    
    return False


def _generate_preference_pairs(llm, sampling_params, batch, batch_num=None):
    """Generate preference pairs (chosen/rejected) for a batch of prompts.
    
    We generate `per_try` samples for each input in the batch using a single model call.
    This is done by repeating each input `per_try` times and generating all at once.
    
    Args:
        model: AutoPeftModelForCausalLM used for sampling
        tokenizer: Tokenizer
        batch: Batch of prompts
        batch_num: Batch number for logging
    """
    per_try = 8  # samples to generate per input in a single call
    results = []
    
    batch_size = len(batch["prompt"])
    batch_label = f"Batch {batch_num}" if batch_num is not None else "Batch"
    
    print(f"\n{batch_label}: Processing {batch_size} prompts...")

    for idx, (prompt, prompt_text, label, category) in enumerate(zip(batch["prompt"], batch["prompt_text"], batch["prompt_label"], batch["category"])):
        lang = batch["lang"][idx]
        minibatch = {
            "prompt": [prompt] * per_try,
            "prompt_label": [label] * per_try,
            "prompt_text": [prompt_text] * per_try,
            "lang": [lang] * per_try,
            "category": [category] * per_try,
        }
        
        outputs = batch_generate_responses(llm, sampling_params, minibatch)
        
        chosen = None
        inappropriate_responses = []
        correct_count = 0
        wrong_count = 0
        
        for out in outputs:
            response = out["response"]
            conclusion = out["conclusion"]

            if _is_wrong_conclusion(label, conclusion, lang):
                wrong_count += 1
                continue  # Skip wrong conclusions

            if not _is_inappropriate_conclusion(conclusion, lang):
                correct_count += 1
                if chosen is None:  # Take first correct one as chosen
                    chosen = response
            else:
                inappropriate_responses.append(response)
        
        # Detailed logging per prompt
        print(f"  Prompt {idx+1}/{batch_size} (label={label}): {correct_count} correct, {wrong_count} wrong, {len(inappropriate_responses)} inappropriate out of {len(outputs)} samples.")

        # Create preference pairs: one chosen vs each rejected
        if chosen is not None:
            pairs_for_prompt = len(inappropriate_responses)
            for rejected in inappropriate_responses:
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
    
    print(f"\nLoading Base model from {BASE_MODEL_NAME}...")
    llm, sampling_params = init_vllm_model()

    dataset = load_biasguard_dataset(mixed=True)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     SFT_MODEL_PATH,
    #     device_map='auto',
    #     torch_dtype=torch.float16
    # )
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token


    # Generate preference pairs
    print(f"\nGenerating preference pairs...")
    print(f"Batch size: {BATCH_SIZE}")
    
    batch_size = BATCH_SIZE
    out_path = RL_DATA_PATH
    
    total_pairs = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Total batches to process: {total_batches}")
    print(f"Total prompts: {len(dataset)}")
    print("="*60)
    
    with open(out_path, "a") as f:
        for batch_idx, i in enumerate(tqdm(range(0, len(dataset), batch_size), desc="Processing batches"), 1):
            batch = dataset[i : i + batch_size]
            pairs = _generate_preference_pairs(llm, sampling_params, batch, batch_num=batch_idx)
            
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

