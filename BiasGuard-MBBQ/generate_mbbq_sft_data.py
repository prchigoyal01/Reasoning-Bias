"""
Generate SFT training data for BiasGuard-MBBQ using vLLM.

Reads BiasGuard-formatted MBBQ prompts and produces SFT rows by sampling
reasoning traces from the teacher model.
"""

import json
import os
import regex as re
import torch

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams

from config_mbbq import (
    SFT_DATA_PATH,
    PROMPT_TEMPLATE_PATH,
    BIAS_TYPES_PATH,
    STANDARDS_PATH,
    TEACHER_MODEL_NAME,
    BATCH_SIZE,
    MBBQ_BIASGUARD_PATH,
)

# Language-specific verdict phrases used inside the conclusion text
BIASED_BY_LANGS = {
    "en": "Yes, the sentence is biased.",
    "tr": "Evet, cümle önyargılıdır.",
}
UNBIASED_BY_LANGS = {
    "en": "No, the sentence is unbiased.",
    "tr": "Hayır, cümle önyargısızdır.",
}

CONCLUSION_BY_LANGS = {
    "en": "## Conclusion ##:",
    "tr": "## Sonuç ##:",
}


def get_mbbq_system_instruction(lang):
    """Load MBBQ-specific bias types and standards."""

    with open(BIAS_TYPES_PATH + f"/{lang}.json", "r", encoding="utf-8") as f:
        bias_dict = json.load(f)
        bias_types = ""
        for bias_name, bias in bias_dict.items():
            bias_types += "\n" + bias_name + ":\n"
            for k, v in bias.items():
                bias_types += f"{k}: {v}\n"

    with open(STANDARDS_PATH + f"/{lang}.json", "r", encoding="utf-8") as f:
        standards_dict = json.load(f)
        standards = ""
        for _, standard in standards_dict.items():
            standards += "\n"
            for k, v in standard.items():
                standards += f"{k}: {v}\n"

    with open(PROMPT_TEMPLATE_PATH + f"/{lang}.json", "r", encoding="utf-8") as f:
        prompt_template = json.load(f)
        prompt = ""
        for key, value in prompt_template.items():
            value = value.replace("{standards}", standards)
            value = value.replace("{bias_types}", bias_types)
            prompt += f"{key}\n{value}\n\n"

    return prompt


def format_prompt(example, sys_inst):
    """Format example with system instruction."""
    messages = [
        {"role": "SYSTEM", "content": sys_inst},
        {"role": "USER", "content": example["prompt"]},
    ]
    text = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])
    text += "\n[ASSISTANT]: "
    return {"prompt_text": text}


def load_biasguard_dataset_by_lang(lang):
    file_path = MBBQ_BIASGUARD_PATH + f"/{lang}.jsonl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Expected BiasGuard-formatted data at {file_path}."
        )

    print(f"\nLoading BiasGuard-formatted data from {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    sys_inst = get_mbbq_system_instruction(lang=lang)
    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda ex: {**ex, **format_prompt(ex, sys_inst)})
    dataset = dataset.map(lambda ex: {**ex, "lang": lang})
    print(f"Loaded {len(dataset)} total prompts.")
    return dataset

def load_biasguard_dataset(mixed: bool = False):
    """Load pre-converted BiasGuard examples and attach prompts.
    Default is English-only.
    """
    if mixed:
        en_dataset = load_biasguard_dataset_by_lang(lang="en")
        tr_dataset = load_biasguard_dataset_by_lang(lang="tr")
        combined = concatenate_datasets([en_dataset, tr_dataset])
        return combined.shuffle(seed=42)
    else:
        return load_biasguard_dataset_by_lang(lang="en")


def _is_wrong_conclusion(label, conclusion_text, lang: str):
    """Quick sanity checks to keep only label-consistent completions.

    Check verdict markers specific to the provided language.
    """
    if not conclusion_text:
        return True

    conclusion_text = conclusion_text.strip()
    biased_marker = BIASED_BY_LANGS[lang]
    unbiased_marker = UNBIASED_BY_LANGS[lang]

    contains_biased = biased_marker in conclusion_text
    contains_unbiased = unbiased_marker in conclusion_text

    if label == 1:
        # Label 1 means "biased" according to upstream dataset
        return contains_unbiased or not contains_biased
    # Else label 0 means "unbiased"
    return not contains_unbiased or contains_biased


def init_vllm_model():
    """Create a vLLM generator that mirrors the BiasGuard teacher sampling."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! vLLM generation requires a GPU.")

    print("\nInitializing teacher model with vLLM...")
    llm = LLM(
        model=TEACHER_MODEL_NAME,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=48000,
    )
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=1.0,
    )
    return llm, sampling_params


def batch_generate_responses(llm, sampling_params, batch):
    """Sample responses for a batch of prompts via vLLM."""
    prompts = batch["prompt_text"]
    lang = batch["lang"]
    categories = batch.get("category", [""] * len(prompts))
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for idx, output in enumerate(outputs):
        generations = output.outputs
        if not generations:
            print(f"Warning: No generations for prompt index {idx}, skipping.")
            continue

        text = generations[0].text.strip()
        conclusion_matches = list(re.findall(rf"{CONCLUSION_BY_LANGS[lang[idx]]}\s*(.*)", text))
        if conclusion_matches:
            conclusion = conclusion_matches[-1].strip()
            results.append(
                {
                    "prompt": batch["prompt"][idx],
                    "label": int(batch["prompt_label"][idx]),
                    "response": text,
                    "conclusion": conclusion,
                    "lang": lang[idx],
                    "category": categories[idx],
                }
            )
        else:
            print(f"Warning: No conclusion found in generation for prompt index {idx}, skipping.")
    
    print(len(results), "responses generated in batch.")
    return results


def generate_sft_data_mbbq():
    """Main entrypoint to build SFT data from vLLM generations."""
    print("=" * 60)
    print("Generating SFT Data for BiasGuard-MBBQ (vLLM)")
    print("=" * 60)

    dataset = load_biasguard_dataset(mixed=True)
    llm, sampling_params = init_vllm_model()

    # Reset output file
    if os.path.exists(SFT_DATA_PATH):
        os.remove(SFT_DATA_PATH)

    saved = 0
    label_counts = {}
    category_counts = {}

    for start in tqdm(
        range(0, len(dataset), BATCH_SIZE), desc="Generating", dynamic_ncols=True
    ):
        end = start + BATCH_SIZE
        batch = dataset[start:end]
        batch_results = batch_generate_responses(llm, sampling_params, batch)
        batch_results = [
            res
            for res in batch_results
            if not _is_wrong_conclusion(res["label"], res["conclusion"], res["lang"])
            and len(res["response"]) > 50
        ]

        if not batch_results:
            continue

        with open(SFT_DATA_PATH, "a", encoding="utf-8") as f:
            for row in batch_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved += 1
                label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
                category = row.get("category") or "unknown"
                category_counts[category] = category_counts.get(category, 0) + 1

    print(f"\nSaved {saved} examples to {SFT_DATA_PATH}")

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count}")

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    generate_sft_data_mbbq()
