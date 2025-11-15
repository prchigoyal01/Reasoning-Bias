import json
import torch

from config import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
import regex as re

BIASED = "Yes, the sentence is biased."
UNBIASED = "No, the sentence is unbiased."

def get_system_instruction():
    print("Loading prompt template from:", PROMPT_TEMPLATE_PATH)

    with open("prompt_templates/bias_types.json", 'r') as f:
        bias_dict = json.load(f)
        bias_types = ""
        for bias_name, bias in bias_dict.items():
            bias_types += "\n" + bias_name + ":\n"
            for k, v in bias.items():
                bias_types += f"{k}: {v}\n"
    
    with open("prompt_templates/standards.json", 'r') as f:
        standards_dict = json.load(f)
        standards = ""
        for _, standard in standards_dict.items():
            standards += "\n"
            for k, v in standard.items():
                standards += f"{k}: {v}\n"

    with open(PROMPT_TEMPLATE_PATH, 'r') as f:
        prompt_template = json.load(f)
        prompt = ""
        for key, value in prompt_template.items():
            value = value.replace("{standards}", standards)
            value = value.replace("{bias_types}", bias_types)
            prompt += f"{key}\n{value}\n\n"

    return prompt


def format_prompt(example, sys_inst):
    messages = [
        {"role": "SYSTEM", "content": sys_inst},
        {"role": "USER", "content": example["prompt"]},
    ]
    text = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])
    text += "\n[ASSISTANT]: "
    return {"prompt_text": text}


def setup_dataset():
    sys_inst = get_system_instruction()  # Evaluate sys_inst only once
    ds = load_dataset("toxigen/toxigen-data", "train")
    ds = ds['train'].shuffle(seed=42).select(range(LAST_CHKPT, TOTAL_SAMPLES))
    dataset = Dataset.from_list(ds)
    dataset = dataset.map(lambda example: format_prompt(example, sys_inst))
    return dataset

def _is_wrong_conclusion(prompt_label, conclusion_text, output_text):
    # Bias undetected
    if prompt_label == 1 and ((UNBIASED in conclusion_text) or (BIASED not in conclusion_text)):
        # print("Detected wrong conclusion for biased prompt.")
        return True
    # Bias detected incorrectly
    if prompt_label == 0 and ((UNBIASED not in conclusion_text) or (BIASED in conclusion_text)):
        # print("Detected wrong conclusion for unbiased prompt.")
        return True
    if len(output_text) < len(conclusion_text) + len(prompt_label): # not enough thinking steps
        # print("Too short output, likely missing steps.")
        return True
    return False

def batch_generate_responses(model, tokenizer, batch):
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
            max_new_tokens=2048, # specified in the paper
            temperature=0.7,
            do_sample=True,
        )
    decoded = tokenizer.batch_decode(
        [out[input_lens[i]:] for i, out in enumerate(outputs)],
        skip_special_tokens=True
    )

    for prompt, prompt_label, output in zip(batch['prompt'], batch['prompt_label'], decoded):
        conclusion = re.search(r"## Conclusion ##:\s*(.*)", output)
        if conclusion:
            conclusion = conclusion.group(1).strip()
            # print(f"Prompt Label: {prompt_label}, Conclusion: {conclusion}")
            finetune_row = {
                "prompt": prompt,
                "label": prompt_label,
                "response": output,
                "conclusion": conclusion
            }
            results.append(finetune_row)
    return results

def generate_sft_data():
    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",          # most memory-efficient format
        bnb_4bit_compute_dtype=torch.float16  # use bfloat16 if your GPU supports it
    )
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weights=True,
        bnb_8bit_compute_dtype=torch.float16,  # or torch.bfloat16 if supported
    )

    # Load the model in 8-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        device_map="auto", # Automatically distributes the model across available GPUs
        quantization_config = bnb_config_8bit,
        # offload_folder="./offload",       # folder for CPU offload weights
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    dataset = setup_dataset()
    
    batch_size = BATCH_SIZE
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_results = batch_generate_responses(model, tokenizer, dataset[i:i+batch_size])
        batch_results = [res for res in batch_results if not _is_wrong_conclusion(res["prompt"], res["conclusion"], res["response"])]

        with open(SFT_DATA_PATH, "a") as f:
            for row in batch_results:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    generate_sft_data()
