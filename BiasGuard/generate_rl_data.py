import json
import torch

from config import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
import regex as re
from trl import extract_prompt
from peft import AutoPeftModelForCausalLM

from generate_sft_data import setup_dataset, batch_generate_responses, BIASED, UNBIASED


def generate_rl_data():
    """Generate RL-style data where for each correct response we produce N wrong responses.

    Output format (one JSON object per line in `rl_data.jsonl`):
      {
        "prompt": <prompt string>,
        "label": <0|1>,
        "chosen": <correct response string>,
        "rejected": [<wrong1>, <wrong2>, ...]
      } 

    The function uses `batch_generate_responses` to obtain teacher / correct responses
    and then attempts to generate `n_wrong` distinct incorrect responses per prompt by
    sampling the model and selecting outputs whose conclusion indicates the opposite
    label (clear wrong answers). It writes results to `rl_data.jsonl`.
    """
    required_steps = ["Step " + str(i) for i in range(1, 5)]

    def _is_inappropriate_conclusion(prompt_label, conclusion_text, output_text):
        # Bias undetected
        if prompt_label == 1 and ((UNBIASED in conclusion_text) or (BIASED not in conclusion_text)):
            # print("Detected wrong conclusion for biased prompt.")
            return True
        # Bias detected incorrectly
        if prompt_label == 0 and ((UNBIASED not in conclusion_text) or (BIASED in conclusion_text)):
            # print("Detected wrong conclusion for unbiased prompt.")
            return True
        
        for step in required_steps:
            if step not in output_text:
                # print(f"Missing required step in conclusion: {step}")
                return True

        return False

    def _generate_preference_pairs(model, tokenizer, batch):
        """Attempt to collect up to n_wrong distinct wrong responses for a single prompt.

        We generate `per_try` samples for each input in the batch using a single model call.
        This is done by repeating each input `per_try` times and generating all at once.
        """
        per_try = 8  # samples to generate per input in a single call
        results = []

        for prompt, prompt_text, label in zip(batch["prompt"], batch["prompt_text"], batch["prompt_label"]):
            minibatch = {
                "prompt": [prompt] * per_try,
                "prompt_label": [label] * per_try,
                "prompt_text": [prompt_text] * per_try,
            }
            outputs = batch_generate_responses(model, tokenizer, minibatch)
            
            chosen = None
            wrong_responses = []
            for out in outputs:
                response = out["response"]
                conclusion = out["conclusion"]

                if not _is_inappropriate_conclusion(label, conclusion, response):
                    chosen = response
                else:
                    wrong_responses.append(response)
                
            print(f"{len(wrong_responses)} wrong responses of {per_try} generated.")

            if chosen is not None:
                for response in wrong_responses:
                    results.append({
                        "prompt": prompt,
                        "label": label,
                        "chosen": chosen,
                        "rejected": response
                    })
        return results

    # --- main logic ---

    model = AutoPeftModelForCausalLM.from_pretrained(
        'ineedausername101/ANLP-BiasGuard-lora-adapter',
        device_map='auto',
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained('ineedausername101/ANLP-BiasGuard-lora-adapter')
    dataset = setup_dataset()

    batch_size = BATCH_SIZE
    out_path = "rl_data.jsonl"

    with open(out_path, "a") as f:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            pairs = _generate_preference_pairs(model, tokenizer, batch)
            for row in pairs:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    generate_rl_data()
