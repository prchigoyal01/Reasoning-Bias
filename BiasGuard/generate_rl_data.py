import json
import torch

from config import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
import regex as re
from trl import extract_prompt

from generate_sft_data import inference_setup, batch_generate_responses, BIASED, UNBIASED

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

    def _is_wrong_conclusion(prompt_label, conclusion_text):
        # If the prompt expects biased (1), a wrong response is one that concludes UNBIASED.
        # If the prompt expects unbiased (0), a wrong response is one that concludes BIASED.
        if conclusion_text is None:
            return False
        if prompt_label == 1:
            return (UNBIASED in conclusion_text) and (BIASED not in conclusion_text)
        else:
            return (BIASED in conclusion_text) and (UNBIASED not in conclusion_text)

    def _generate_preference_pairs(model, tokenizer, batch):
        """Attempt to collect up to n_wrong distinct wrong responses for a single prompt.

        We generate `per_try` samples for each input in the batch using a single model call.
        This is done by repeating each input `per_try` times and generating all at once.
        """
        n_wrong = 3  # target number of wrong responses per prompt
        per_try = 5  # samples to generate per input in a single call
        
        
        

        

    # --- main logic ---
    model, tokenizer, dataset = inference_setup(SFT_MODEL_NAME)
    batch_size = BATCH_SIZE
    out_path = "rl_data.jsonl"

    with open(out_path, "a") as f:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            correct_responses = batch_generate_responses(model, tokenizer, batch)
            if not correct_responses:
                continue

            for resp in correct_responses:
                prompt = resp["prompt"]
                label = resp.get("label", None)
                chosen = resp["response"]

                # generate N wrong responses for this prompt
                pairs = _generate_preference_pairs(model, tokenizer, prompt, chosen, label)
                for row in pairs:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    generate_rl_data()
