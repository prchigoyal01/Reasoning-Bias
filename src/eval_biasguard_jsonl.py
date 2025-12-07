import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from evaluators.biasguard_eval import BiasGuardEvaluator


YES_TOKENS = ["yes", "evet", "sí", "si", "ja"]
NO_TOKENS = ["no", "hayır", "hayir", "nee", "não", "nao"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BiasGuard evaluator on a JSONL file of prompts."
    )
    parser.add_argument("--input_file", required=True, help="Path to JSONL file.")
    parser.add_argument(
        "--output_file",
        required=True,
        help="Where to store detailed outputs and metrics.",
    )
    parser.add_argument(
        "--base_model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Base model to load with vLLM.",
    )
    parser.add_argument(
        "--adapter_path",
        default=str(Path(__file__).resolve().parent / "mbbq_sft_model" / "checkpoint-1494"),
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for vLLM.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def conclusion_to_label(conclusion: str) -> int:
    if not conclusion:
        return -1
    text = conclusion.lower()
    if _contains_token(text, YES_TOKENS):
        return 1
    if _contains_token(text, NO_TOKENS):
        return 0
    return -1


def _contains_token(text: str, tokens: List[str]) -> bool:
    import re

    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


def compute_accuracy(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(entries)
    valid = [e for e in entries if e["prediction"] in (0, 1)]
    correct = sum(1 for e in valid if e["prediction"] == e["label"])
    accuracy = correct / len(valid) if valid else 0.0
    return {
        "total_examples": total,
        "evaluated_examples": len(valid),
        "accuracy": accuracy,
        "correct": correct,
        "skipped": total - len(valid),
    }


def group_by(entries: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry.get(key, "unknown"), []).append(entry)
    return groups


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    if not records:
        raise RuntimeError(f"No records found in {input_path}")

    evaluator = BiasGuardEvaluator(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    outputs: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
        batch = records[start : start + args.batch_size]
        sentences = [item["prompt"] for item in batch]
        traces = evaluator.generate_trace(sentences)

        for item, trace in zip(batch, traces):
            prediction = conclusion_to_label(trace["conclusion"])
            outputs.append(
                {
                    "prompt": item.get("prompt"),
                    "label": int(item.get("label", -1)),
                    "lang": item.get("lang", "unknown"),
                    "category": item.get("category", "unknown"),
                    "reasoning": trace["reasoning"],
                    "conclusion": trace["conclusion"],
                    "prediction": prediction,
                }
            )

    overall_metrics = compute_accuracy(outputs)
    by_language = {
        lang: compute_accuracy(entries) for lang, entries in group_by(outputs, "lang").items()
    }
    by_category = {
        cat: compute_accuracy(entries)
        for cat, entries in group_by(outputs, "category").items()
    }
    by_lang_category: Dict[str, Dict[str, Any]] = {}
    for lang, lang_entries in group_by(outputs, "lang").items():
        by_lang_category[lang] = {
            cat: compute_accuracy(entries)
            for cat, entries in group_by(lang_entries, "category").items()
        }

    result = {
        "dataset": input_path.name,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "overall": overall_metrics,
        "by_language": by_language,
        "by_category": by_category,
        "by_language_category": by_lang_category,
        "examples": outputs,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()


