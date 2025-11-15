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

    def _generate_preference_pairs(model, tokenizer, prompt_text, prompt_label, max_tries=8, per_try=4):
        """Attempt to collect up to n_wrong distinct wrong responses for a single prompt.

        We batch `per_try` samples per generate call by repeating the input tensors.
        """
        n_wrong = 8  # specified in the paper
        collected = []
        seen = set()
        input_enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(model.device)

        input_ids = input_enc["input_ids"]

        for _ in range(max_tries):
            # repeat inputs so generate returns multiple samples in one call
            rep_input_ids = input_ids.repeat_interleave(per_try, dim=0)
            rep_attention_mask = None
            if "attention_mask" in input_enc:
                rep_attention_mask = input_enc["attention_mask"].repeat_interleave(per_try, dim=0)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=rep_input_ids,
                    attention_mask=rep_attention_mask,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=1.2,
                    num_return_sequences=per_try,
                )

            # decode and split per sample
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # each decoded includes the prompt + generated continuation; strip prompt prefix
            for seq in decoded:
                # remove the prompt_text prefix if present to get the generated continuation
                gen = seq
                if seq.startswith(prompt_text):
                    gen = seq[len(prompt_text):].strip()

                # extract conclusion (same logic as batch_generate_responses)
                m = re.search(r"## Conclusion ##:\s*(.*)", seq)
                if m:
                    conclusion = m.group(1).strip()
                    if _is_wrong_conclusion(prompt_label, conclusion):
                        # deduplicate by generated text
                        if gen not in seen and gen and gen != "":
                            seen.add(gen)
                            collected.append({"response": seq, "conclusion": conclusion})
                            if len(collected) >= n_wrong:
                                return collected
        return collected

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
