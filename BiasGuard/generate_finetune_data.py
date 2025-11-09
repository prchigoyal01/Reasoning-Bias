import json
import torch

from config import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
import regex as re


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


def generate_finetune_data():
    ds = load_dataset("toxigen/toxigen-data", "train")
    ds = ds['train'].shuffle(seed=37).select(range(500))
    
    biased = "Yes, the sentence is biased."
    unbiased = "No, the sentence is unbiased."

    sys_inst = get_system_instruction()

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
        quantization_config = bnb_config_4bit,
        # offload_folder="./offload",       # folder for CPU offload weights
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # paper specified 2048, but too slow.
        temperature=1.2, # specified in the paper
    )

    finetune_rows = []
    for data in tqdm(ds):
        # torch.cuda.empty_cache()
        query = data['prompt']
        label = data['prompt_label']

        messages = [
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": query},
        ]

        resp = pipe(messages)[-1]
        output = resp["generated_text"][-1]["content"]
        conclusion = re.search(r"## Conclusion ##:\s*(.*)", output)
        if not conclusion:
            continue        
        conclusion = conclusion.group(1).strip()

        if (label == 1 and biased in conclusion and not (unbiased in conclusion)) or (label == 0 and unbiased in conclusion and not (biased in conclusion)):
            row = {
                'prompt': query,
                'label': label,
                'response': output,
                'conclusion': conclusion
            }
            finetune_rows.append(row)

    with open(FINETUNE_DATA_PATH, 'w') as f:
        for row in finetune_rows:
            f.write(json.dumps(row) + '\n')


if __name__ == "__main__":
    generate_finetune_data()
