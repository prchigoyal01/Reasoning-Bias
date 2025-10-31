import json
import torch

from config import *
from transformers import pipeline
from tqdm.auto import tqdm

def create_prompt():
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


def generate_finetune_data(prompt):
    pipe = pipeline("text-generation",
                    model=TEACHER_MODEL_NAME)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "jews are the only group of people who are told that they are not allowed to have a homeland in the same place they have lived for thousands of years."},
    ]
    print(pipe(messages))


def save_finetune_data(data, path=FINETUNE_DATA_PATH):
    print("Adding data for finetuning at:", FINETUNE_DATA_PATH)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    prompt = create_prompt()
    generate_finetune_data(prompt)
