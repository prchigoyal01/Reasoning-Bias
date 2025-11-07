import json
import torch

from config import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset


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


def response(query):
    prompt = create_prompt()

    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",          # most memory-efficient format
        bnb_4bit_compute_dtype=torch.float16  # use bfloat16 if your GPU supports it
    )
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True,
        bnb_8bit_compute_dtype=torch.float16,  # or torch.bfloat16 if supported
    )

    # Load the model in 8-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        device_map="auto", # Automatically distributes the model across available GPUs
        # load_in_4bit=True, # Load the model in 8-bit precision to save memory
        quantization_config = bnb_config_4bit
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048, # specified in the paper
        temperature=1.2, # specified in the paper
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    return pipe(messages)[-1]["content"]

def generate_finetune_data():
    # ds = load_dataset("toxigen/toxigen-data", "annotations")
    # for data in tqdm(ds['train']):
    #     query = data['text']
    #     print("Query:", query)

    query = "jews are the only group of people who are told that they are not allowed to have a homeland in the same place they have lived for thousands of years."
    response_text = response(query)
    print(response_text)


def save_finetune_data(data, path=FINETUNE_DATA_PATH):
    print("Adding data for finetuning at:", FINETUNE_DATA_PATH)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())
    generate_finetune_data()
