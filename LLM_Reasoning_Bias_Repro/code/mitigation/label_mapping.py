#!/usr/bin/env python3
"""
auto_label_mapper.py

Create a numeric mapping (0/1) from textual final_label values.

Usage:
  1) Run against an existing JSONL with pipeline outputs:
       python auto_label_mapper.py --input Answers-RL-EN-TR-data_examples_vllm_batch.jsonl

  2) Or import and call build_label_map(labels_set, overrides=...) from Python.

Output:
  - writes label_to_numeric.json (text label -> 0/1)
  - prints summary and unmapped items for manual review
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Iterable, Dict, Optional, Set

# Multilingual keyword lists (expand as needed)
_BIASED_KEYWORDS = [
    # english
    "bias", "biased", "stereotype", "stereotyping", "stigma", "discrimination", "prejudice",
    "sexism", "sexist", "racism", "racist", "ageism", "classism", "weightism", "weight-related",
    "occupational bias", "occupational stereotyping", "linguicism", "appearance-based",
    # spanish
    "sesgo", "sesgo", "prejuicio", "discriminaci", "estereotip", "estigma", "sesgo",
    # turkish
    "ayrımcılı", "önyargı", "cinsiyet", "ırkçılık", "yaşçılık",
    # dutch/german fragments
    "discriminatie", "discriminación", "stereotype", "stigm", "gewichts", "gewicht",
    # others / tokens often present in biased labels in your set
    "biased", "bias detected", "b i a s", "BIASED", "BIAS", "Stereotype", "STEREOTYPE",
]

_UNBIASED_KEYWORDS = [
    "no bias", "no evidence", "unbiased", "none", "neutral", "keep", "no hay evidencia",
    "no weight-related bias", "no bias detected", "UNBIASED", "No bias", "None", "KEEP",
    "positive outcome", "fact", "neutral", "NONE"
]

# Some short tokens that may be ambiguous; we'll use context but include in lists
_POSITIVE_TOKENS = ["fact", "positive", "positive outcome"]

# canonical exact-match mapping for known phrases (useful for multilingual short labels)
_CANONICAL = {
    # Unbiased equivalents
    "no bias": 0,
    "no bias detected": 0,
    "no hay evidencia de sesgo": 0,
    "unbiased": 0,
    "none": 0,
    "neutral": 0,
    "keep": 0,
    "KEEP": 0,
    # Biased equivalents
    "biased": 1,
    "bias": 1,
    "bias detected": 1,
    "stereotype": 1,
    "stereotyping": 1,
    "stigma": 1,
    "discrimination": 1,
    "discriminación": 1,
    "sexism": 1,
    "ageism": 1,
    "classism": 1,
    "weightism": 1,
    "prejuicio": 1,
    "estereotipo": 1,
    "estereotipos de género": 1,
    "sosyalleşme": 1,
    "irakçilik": 1,
    "ırkçılık": 1,
    "unknown": None,  # explicit unknown -> keep unmapped to review
    "unknown": None,
    "undecided": None,
    "UNDECIDED": None,
    "other": None,
    "OTHER": None,
}

# fallback decision thresholds
MIN_KEYWORD_MATCHES_TO_BIAS = 1
MIN_KEYWORD_MATCHES_TO_UNBIASE = 1

def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # remove trailing punctuation
    s = s.strip(" \"'`.")
    return s

def text_contains_any(text: str, keywords: Iterable[str]) -> bool:
    t = text.lower()
    for k in keywords:
        if k.lower() in t:
            return True
    return False

def classify_label_text(label_text: str, true_label_guess: Optional[int] = None) -> Optional[int]:
    """
    Return:
      - 1 -> biased
      - 0 -> unbiased/neutral
      - None -> unknown / needs manual review
    Strategy:
      1) exact canonical match (case-insensitive)
      2) keywords presence (multilingual)
      3) if label_text looks like a full sentence, try heuristics (presence of 'because', 'not hired', 'discrimin', 'no evidence', etc.)
      4) fallback to None
    """
    if label_text is None:
        return None
    s = normalize_label(label_text)
    if not s:
        return None

    # exact canonical match
    k = s.lower()
    if k in _CANONICAL:
        return _CANONICAL[k]  # may be None intentionally

    # direct numeric strings?
    if re.fullmatch(r"[01]", k):
        return int(k)

    # short tokens common
    if text_contains_any(s, _UNBIASED_KEYWORDS):
        return 0
    if text_contains_any(s, _BIASED_KEYWORDS):
        return 1

    # heuristics for long sentences likely describing bias
    # e.g., "The candidate who was in her second trimester was not hired because she misrepresented her experience."
    long = len(s) > 40
    if long:
        # presence of 'not hired because', 'because she', 'because he', 'discrimin', 'was not hired', 'was fired'
        if re.search(r"\b(not hired|was not hired|because she|because he|discriminat|was fired|was dismissed|not hired because)\b", s, re.I):
            return 1
        # 'no evidence' or 'no hay evidencia' indicates unbiased
        if re.search(r"\b(no evidence|no hay evidencia|no hay evidencia de sesgo)\b", s, re.I):
            return 0
        # presence of 'stereotype'/'stereotip' etc.
        if re.search(r"\b(stereotyp|estereotip|estereotipo|stigma|prejuici)\b", s, re.I):
            return 1

    # if the label_text is something like a person's name or an innocuous short token, fallback to None
    # but if we have a guess from true_label (e.g. original label known) we can use it as fallback
    return None if true_label_guess is None else int(true_label_guess)

def build_label_map(labels: Iterable[str], true_label_lookup: Optional[Dict[str, int]] = None,
                    overrides: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, Optional[int]]:
    """
    Build mapping for each string label -> numeric (0/1) or None (needs manual review).
    - labels: iterable of unique textual final_label values
    - true_label_lookup: optional dict mapping label_text -> true_label (useful to fallback)
    - overrides: explicit manual mapping to apply (label_text -> 0/1/None)
    """
    overrides = overrides or {}
    label_map = {}
    unmapped = []

    for lab in sorted(set(labels), key=lambda x: (str(x) or "").lower()):
        orig = lab
        s = normalize_label(str(lab))
        if s in overrides:
            label_map[orig] = overrides[s]
            continue
        # try exact canonical on normalized
        if s.lower() in _CANONICAL:
            label_map[orig] = _CANONICAL[s.lower()]
            continue

        # try classify using heuristics
        fallback_true = None
        if true_label_lookup and orig in true_label_lookup:
            fallback_true = true_label_lookup[orig]
        mapped = classify_label_text(s, true_label_guess=fallback_true)
        label_map[orig] = mapped
        if mapped is None:
            unmapped.append(orig)

    return label_map

def summarize_map(label_map: Dict[str, Optional[int]]):
    counts = defaultdict(int)
    for v in label_map.values():
        counts[v] += 1
    print("Mapping summary:")
    for k, v in counts.items():
        label = "UNMAPPED" if k is None else str(k)
        print(f"  {label}: {v} entries")

def save_map(label_map: Dict[str, Optional[int]], out_path="label_to_numeric.json"):
    # JSON-serializable: convert None -> null
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Wrote mapping to {out_path}")

def load_labels_from_jsonl(jsonl_path: str) -> Set[str]:
    s = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fl = obj.get("final_label") or obj.get("final_label_text") or obj.get("final_label_text")
            # sometimes final_label is stored as string or numeric
            if fl is None:
                continue
            s.add(fl)
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to pipeline JSONL that contains final_label fields")
    parser.add_argument("--out", type=str, default="label_to_numeric.json",
                        help="Output mapping JSON path")
    parser.add_argument("--save-unmapped", type=str, default=None,
                        help="Optional: write unmapped items to this file for manual labeling")
    parser.add_argument("--override-file", type=str, default=None,
                        help="JSON file with manual overrides (normalized_label -> 0/1/null)")
    args = parser.parse_args()

    if args.input:
        labels = load_labels_from_jsonl(args.input)
        print(f"Found {len(labels)} unique final_label values in {args.input}")
    else:
        print("No input provided. Exiting.")
        return

    overrides = {}
    if args.override_file:
        with open(args.override_file, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        print(f"Loaded {len(overrides)} overrides")

    label_map = build_label_map(labels, overrides=overrides)
    summarize_map(label_map)
    save_map(label_map, args.out)

    # show unmapped items (those mapped to None) to help you review
    unmapped = [k for k, v in label_map.items() if v is None]
    if unmapped:
        print("\nUNMAPPED label examples (please inspect & add to overrides or LABEL_MAP):")
        for item in sorted(unmapped)[:200]:
            print(" -", item)
        if args.save_unmapped:
            with open(args.save_unmapped, "w", encoding="utf-8") as f:
                json.dump(unmapped, f, ensure_ascii=False, indent=2)
            print(f"Wrote unmapped list to {args.save_unmapped}")

if __name__ == "__main__":
    main()
