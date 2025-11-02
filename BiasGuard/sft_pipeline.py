'''
WIP
'''

import argparse
import os
from typing import Dict, List
import datasets
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#!/usr/bin/env python3
"""
sft_pipeline.py

Minimal supervised fine-tuning pipeline for causal LMs (GPT-style) with optional LoRA.
- Uses Hugging Face Transformers + Datasets + PEFT (optional) + Trainer.
- Expects dataset with fields: "prompt"/"instruction"/"input" and "response"/"answer"/"output".
- Can train only on the response tokens (recommended for instruction tuning).

Example:
    python sft_pipeline.py \
        --model_name_or_path gpt2 \
        --dataset_path data/train.jsonl \
        --output_dir outputs/sft-gpt2 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --use_lora
"""


from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        default_data_collator,
        set_seed,
)

# Optional imports; we'll import lazily to keep runtime flexible
PEFT_AVAILABLE = False
try:

        PEFT_AVAILABLE = True
except Exception:
        PEFT_AVAILABLE = False


def parse_args():
        p = argparse.ArgumentParser(description="Supervised fine-tuning pipeline for causal LMs")
        p.add_argument("--model_name_or_path", required=True)
        p.add_argument("--dataset_path", required=True, help="Hugging Face dataset id or local file (.jsonl/.json/.csv)")
        p.add_argument("--dataset_config_name", default=None)
        p.add_argument("--output_dir", required=True)
        p.add_argument("--max_seq_length", type=int, default=1024)
        p.add_argument("--per_device_train_batch_size", type=int, default=4)
        p.add_argument("--per_device_eval_batch_size", type=int, default=4)
        p.add_argument("--num_train_epochs", type=int, default=3)
        p.add_argument("--learning_rate", type=float, default=2e-5)
        p.add_argument("--weight_decay", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--use_lora", action="store_true", help="Enable LoRA via PEFT (requires peft installed)")
        p.add_argument("--lora_r", type=int, default=8)
        p.add_argument("--lora_alpha", type=int, default=32)
        p.add_argument("--lora_dropout", type=float, default=0.1)
        p.add_argument("--train_on_inputs", action="store_true", help="Compute loss on entire sequence (prompt+response). If false, only response tokens contribute to loss.")
        p.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X steps. 0 disables.")
        p.add_argument("--resume_from_checkpoint", default=None)
        return p.parse_args()


def find_text_fields(column_names: List[str]) -> Dict[str, str]:
        # heuristics to find prompt and response columns
        keys = [k.lower() for k in column_names]
        prompt_keys = ["prompt", "instruction", "input", "context", "question"]
        response_keys = ["response", "output", "answer", "completion"]
        prompt_col = None
        response_col = None
        for k, orig in zip(keys, column_names):
                if prompt_col is None and k in prompt_keys:
                        prompt_col = orig
                if response_col is None and k in response_keys:
                        response_col = orig
        # fallback heuristics
        if prompt_col is None:
                # take the first text column that is not obviously the label
                for orig in column_names:
                        if isinstance(orig, str) and orig.lower() not in response_keys:
                                prompt_col = orig
                                break
        if response_col is None:
                # take last column
                response_col = column_names[-1]
        return {"prompt": prompt_col, "response": response_col}


def load_dataset(dataset_path: str, dataset_config_name=None):
        if os.path.exists(dataset_path):
                # local file(s)
                ext = os.path.splitext(dataset_path)[1].lower()
                if ext in [".json", ".jsonl"]:
                        ds = datasets.load_dataset("json", data_files=dataset_path, split="train")
                elif ext in [".csv"]:
                        ds = datasets.load_dataset("csv", data_files=dataset_path, split="train")
                else:
                        raise ValueError("Unsupported local dataset file extension: " + ext)
        else:
                # Hugging Face dataset identifier
                ds = datasets.load_dataset(dataset_path, dataset_config_name)
                # if dataset returns a dict with splits, prefer "train"
                if isinstance(ds, dict):
                        if "train" in ds:
                                ds = ds["train"]
                        else:
                                # pick first split
                                ds = next(iter(ds.values()))
        return ds


def build_prompt(example: Dict[str, str]) -> str:
        # Minimal instruction->response template; adapt as needed.
        # We expect example to already have "prompt" and "response" keys.
        prompt = example["prompt"].strip()
        # Keep a separator to help the model distinguish response region
        return f"{prompt}\n\n### Response:\n\n{example['response'].strip()}"


def prepare_tokenizer_and_model(model_name_or_path: str, tokenizer_kwargs=None, use_lora=False):
        tokenizer_kwargs = tokenizer_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, **tokenizer_kwargs)
        # Ensure pad token exists
        if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                else:
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
        # resize token embeddings if tokenizer was modified
        model.resize_token_embeddings(len(tokenizer))
        # Optionally prepare for LoRA/k-bit training (if using 8-bit or 4-bit; here we just show hook)
        if use_lora:
                if not PEFT_AVAILABLE:
                        raise RuntimeError("PEFT library is required for LoRA but not installed.")
                # prepare_model_for_kbit_training is useful when using 8-bit; it's safe otherwise
                model = prepare_model_for_kbit_training(model)
        return tokenizer, model


def tokenize_and_group(examples, tokenizer, max_length, train_on_inputs, prompt_col, response_col):
        prompts = []
        for p, r in zip(examples[prompt_col], examples[response_col]):
                ex = {"prompt": str(p), "response": str(r)}
                prompts.append(build_prompt(ex))
        tokenized = tokenizer(prompts, truncation=True, max_length=max_length, padding=False)
        input_ids = tokenized["input_ids"]
        labels = []
        for i, ids in enumerate(input_ids):
                if train_on_inputs:
                        labels.append(ids.copy())
                else:
                        # mask prompt tokens with -100 so loss is computed only on response region
                        # We need to detect the boundary between prompt and response. Recreate prompt encoding to find split.
                        # Simpler heuristic: encode only prompt part and compute its length.
                        prompt_text = str(examples[prompt_col][i]).strip()
                        # create prompt-only string up to the "### Response:" marker used in build_prompt
                        prompt_only = prompt_text
                        prompt_encoded = tokenizer(prompt_only + "\n\n### Response:\n\n", truncation=True, max_length=max_length)["input_ids"]
                        plen = len(prompt_encoded)
                        label = [-100] * len(ids)
                        for j in range(plen, len(ids)):
                                label[j] = ids[j]
                        labels.append(label)
        tokenized["labels"] = labels
        return tokenized


def main():
        args = parse_args()
        set_seed(args.seed)

        ds = load_dataset(args.dataset_path, args.dataset_config_name)
        # If dataset is a DatasetDict, take train split
        if hasattr(ds, "column_names"):
                column_names = ds.column_names
        else:
                column_names = ds.column_names

        cols = find_text_fields(column_names)
        prompt_col = cols["prompt"]
        response_col = cols["response"]

        tokenizer, model = prepare_tokenizer_and_model(args.model_name_or_path, use_lora=args.use_lora)

        if args.use_lora:
                # configure LoRA
                if not PEFT_AVAILABLE:
                        raise RuntimeError("PEFT not available. Install via `pip install peft` to use --use_lora.")
                peft_config = LoraConfig(
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                        lora_dropout=args.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)

        # Tokenize dataset
        def _tokenize_fn(batch):
                return tokenize_and_group(batch, tokenizer, args.max_seq_length, args.train_on_inputs, prompt_col, response_col)

        tokenized_ds = ds.map(_tokenize_fn, batched=True, remove_columns=ds.column_names)

        # split train/eval if no eval split present
        if "train" in tokenized_ds and "validation" in tokenized_ds:
                train_dataset = tokenized_ds["train"]
                eval_dataset = tokenized_ds["validation"]
        else:
                # tokenized_ds is a Dataset; create small eval split
                total = len(tokenized_ds)
                eval_size = max(1, int(0.05 * total))
                tokenized_ds = tokenized_ds.train_test_split(test_size=eval_size, seed=args.seed)
                train_dataset = tokenized_ds["train"]
                eval_dataset = tokenized_ds["test"]

        # Data collator: for causal LM we can use default_data_collator since labels are prepared
        data_collator = default_data_collator

        training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=args.num_train_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                evaluation_strategy="steps",
                eval_steps=500,
                logging_steps=50,
                save_steps=args.save_steps if args.save_steps > 0 else 2**30,
                save_total_limit=3,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=True,
                optim="paged_adamw_32bit",
                warmup_steps=50,
                report_to="none",
        )

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
        )

        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.output_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
        main()