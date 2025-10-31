import json

from config import FINETUNE_DATA_PATH, PROMPT_TEMPLATE_PATH
from transformers import pipeline
from tqdm.auto import tqdm


def create_prompt():
    print("Loading prompt template from:", PROMPT_TEMPLATE_PATH)

    with open(PROMPT_TEMPLATE_PATH, 'r') as f:
        prompt_template = json.load(f)

    prompt = ""
    for key, value in prompt_template.items():
        prompt += f"{key}\n{value}\n\n"
    return prompt


def generate_finetune_data(prompt):
    pipe = pipeline("text-generation",
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    print(pipe(messages))


def save_finetune_data(data, path=FINETUNE_DATA_PATH):
    print("Adding data for finetuning at:", FINETUNE_DATA_PATH)


if __name__ == "__main__":
    prompt = create_prompt()
    generate_finetune_data(prompt)
