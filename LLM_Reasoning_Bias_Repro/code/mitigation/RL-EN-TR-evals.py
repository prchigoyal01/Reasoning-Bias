# # ----- EVALUATION -----
# from sklearn.metrics import accuracy_score
# import json

# OUTPUT_PATH = "Answers-RL-EN-TR-data_examples_vllm_batch.jsonl"

# # LABEL_MAP = {
# #     "The person with schizophrenia is the person who did not react violently.": 0,
# #     "The person with anxiety is the person who did not react violently.": 1,
# #     # add any other final_label_text → numeric mapping
# # }

# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f]

# def evaluate_pipeline(outputs):
#     y_true = []
#     y_pred_before = []
#     y_pred_after = []

#     corrected_count = 0
#     changed_labels = 0

#     for ex in outputs:
#         true_label = ex.get("original_label")
#         pred_before = ex.get("initial_prediction")
#         final_label = ex.get("final_label")
#         # Convert string labels if needed
#         if isinstance(final_label, str):
            
#             pred_after = final_label
#         elif final_label is None:
#             pred_after = true_label
#         else:
#             pred_after = final_label

#         # Handle missing initial predictions
#         if pred_before is None or pred_before == -1:
#             pred_before_int = -1
#         else:
#             pred_before_int = int(pred_before)

#         pred_after_int = int(pred_after)
#         true_label_int = int(true_label)

#         y_true.append(true_label_int)
#         y_pred_before.append(pred_before_int if pred_before_int != -1 else true_label_int)
#         y_pred_after.append(pred_after_int)

#         # Count dynamically if prediction was corrected
#         # if pred_before_int != -1 and pred_before_int != true_label_int and pred_after_int == true_label_int:
#         #     corrected_count += 1
#         if pred_before_int != true_label_int and pred_after_int == true_label_int:
#             corrected_count += 1

#         # Count if pipeline changed label relative to original
#         if pred_after_int != true_label_int:
#             changed_labels += 1

#         print(f"True: {true_label_int}, Before: {pred_before_int}, After: {pred_after_int}")

#     acc_before = accuracy_score(y_true, y_pred_before)
#     acc_after = accuracy_score(y_true, y_pred_after)
#     wrong_after_count = 0  # count of correct -> incorrect

#     for ex in outputs:
#         true_label = ex.get("original_label")
#         pred_before = ex.get("initial_prediction")
#         final_label = ex.get("final_label")

#         # Convert string labels if needed
#         if isinstance(final_label, str):
#             pred_after = LABEL_MAP.get(final_label, true_label)
#         elif final_label is None:
#             pred_after = true_label
#         else:
#             pred_after = final_label

#         # Handle missing initial predictions
#         if pred_before is None or pred_before == -1:
#             pred_before_int = -1
#         else:
#             pred_before_int = int(pred_before)

#         pred_after_int = int(pred_after)
#         true_label_int = int(true_label)

#         # Count corrections: wrong -> right
#         if pred_before_int != true_label_int and pred_after_int == true_label_int:
#             corrected_count += 1

#         # Count changed labels relative to original
#         if pred_after_int != true_label_int:
#             changed_labels += 1

#         # Count correct -> incorrect
#         if pred_before_int == true_label_int and pred_after_int != true_label_int:
#             wrong_after_count += 1

#     print("\n=== Evaluation Results ===")
#     print(f"Examples: {len(outputs)}")
#     print(f"Accuracy BEFORE mitigation: {acc_before*100:.2f}%")
#     print(f"Accuracy AFTER mitigation:  {acc_after*100:.2f}%")
#     print(f"Number of predictions corrected: {corrected_count}")
#     print(f"Number of labels changed: {changed_labels}")
#     print(f"Number of predictions made incorrect by pipeline: {wrong_after_count}")

# if __name__ == "__main__":
#     outputs = load_jsonl(OUTPUT_PATH)
#     evaluate_pipeline(outputs)

# ----- EVALUATION -----
# import json
# from sklearn.metrics import accuracy_score
# from pathlib import Path
# from typing import Any, Dict

# DEFAULT_FALLBACK_LABEL = -1
# OUTPUT_PATH = "Answers-RL-EN-TR-data_examples_vllm_batch.jsonl"
# MAPPING_PATH = "label_to_numeric.json"

# UNMAPPED_POLICY = "conservative"

# UNKNOWN_VALUE = -1

# def load_mapping(p: str) -> Dict[str, Any]:
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_jsonl(p: str):
#     out = []
#     with open(p, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             out.append(json.loads(line))
#     return out

# def convert_raw_label(raw, mapping: Dict[str, Any], true_label_fallback):
#     """
#     raw: could be int, numeric string, or textual final_label.
#     mapping: mapping loaded from JSON. values are 0/1 or None.
#     true_label_fallback: numeric original label from dataset (int)
#     returns: int (0/1) or UNKNOWN_VALUE if strict and unmapped
#     """
#     # already numeric
#     if raw is None:
#         return None
#     if isinstance(raw, (int, float)):
#         return int(raw)
#     # string numeric
#     s = str(raw).strip()
#     if s.isdigit():
#         return int(s)
#     # try mapping lookup exact match
#     if s in mapping:
#         v = mapping[s]
#         if v is None:
#             if UNMAPPED_POLICY == "conservative":
#                 return int(true_label_fallback)
#             else:
#                 return UNKNOWN_VALUE
#         return int(v)
#     # no mapping key
#     if UNMAPPED_POLICY == "conservative":
#         return int(true_label_fallback)
#     else:
#         return UNKNOWN_VALUE

# def evaluate(outputs, mapping):
#     y_true = []
#     y_before = []
#     y_after = []

#     corrected = 0       # wrong -> right
#     harmed = 0          # right -> wrong
#     changed = 0         # after != original
#     unmapped_seen = {}

#     # For language breakdown
#     lang_stats = {}

#     for ex in outputs:
#         orig_true = ex.get("original_label")
#         before_raw = ex.get("initial_prediction")
#         after_raw = ex.get("final_label")
#         lang = ex.get("lang", "unknown")

#         if orig_true is None:
#             continue
#         true_int = int(orig_true)

#         if before_raw is None or before_raw == -1:
#             before_int = UNKNOWN_VALUE
#         elif isinstance(before_raw, (int, float)):
#             before_int = int(before_raw)
#         else:
#             s = str(before_raw).strip()
#             before_int = int(s) if s.isdigit() else UNKNOWN_VALUE

#         after_int = convert_raw_label(after_raw, mapping, true_int)

#         y_true.append(true_int)
#         y_before.append(before_int if before_int != UNKNOWN_VALUE else ( -1 ))
#         y_after.append(after_int if after_int != UNKNOWN_VALUE else ( -1 ))

#         if after_int != true_int:
#             changed += 1
#         if before_int != UNKNOWN_VALUE and before_int != true_int and after_int == true_int:
#             corrected += 1
#         if before_int != UNKNOWN_VALUE and before_int == true_int and after_int != true_int:
#             harmed += 1
#         if isinstance(after_raw, str) and after_raw not in mapping:
#             unmapped_seen[after_raw] = unmapped_seen.get(after_raw, 0) + 1

#         # Per-language stats
#         if lang not in lang_stats:
#             lang_stats[lang] = {
#                 "y_true": [], "y_before": [], "y_after": [],
#                 "corrected": 0, "harmed": 0, "changed": 0, "count": 0
#             }
#         lang_stats[lang]["y_true"].append(true_int)
#         lang_stats[lang]["y_before"].append(before_int if before_int != UNKNOWN_VALUE else ( -1 ))
#         lang_stats[lang]["y_after"].append(after_int if after_int != UNKNOWN_VALUE else ( -1 ))
#         lang_stats[lang]["count"] += 1
#         if after_int != true_int:
#             lang_stats[lang]["changed"] += 1
#         if before_int != UNKNOWN_VALUE and before_int != true_int and after_int == true_int:
#             lang_stats[lang]["corrected"] += 1
#         if before_int != UNKNOWN_VALUE and before_int == true_int and after_int != true_int:
#             lang_stats[lang]["harmed"] += 1

#     acc_before = accuracy_score(y_true, y_before)
#     acc_after = accuracy_score(y_true, y_after)

#     # Compute per-language accuracies
#     for lang in lang_stats:
#         lang_stats[lang]["acc_before"] = accuracy_score(lang_stats[lang]["y_true"], lang_stats[lang]["y_before"])
#         lang_stats[lang]["acc_after"] = accuracy_score(lang_stats[lang]["y_true"], lang_stats[lang]["y_after"])

#     return {
#         "acc_before": acc_before,
#         "acc_after": acc_after,
#         "corrected": corrected,
#         "harmed": harmed,
#         "changed": changed,
#         "unmapped_seen": unmapped_seen,
#         "lang_stats": lang_stats,
#     }

# if __name__ == "__main__":
#     mapping = load_mapping(MAPPING_PATH)
#     outputs = load_jsonl(OUTPUT_PATH)
#     stats = evaluate(outputs, mapping)
#     print("UNMAPPED_POLICY:", UNMAPPED_POLICY)
#     print("Examples processed:", len(outputs))
#     print(f"Accuracy BEFORE: {stats['acc_before']*100:.2f}%")
#     print(f"Accuracy AFTER:  {stats['acc_after']*100:.2f}%")
#     print("Corrected (wrong->right):", stats["corrected"])
#     print("Harmed (right->wrong):", stats["harmed"])
#     print("Changed labels (after != original):", stats["changed"])

#     print("\n=== Breakdown by Language ===")
#     for lang, lstat in stats["lang_stats"].items():
#         print(f"\nLanguage: {lang}")
#         print(f"  Examples: {lstat['count']}")
#         print(f"  Accuracy BEFORE: {lstat['acc_before']*100:.2f}%")
#         print(f"  Accuracy AFTER:  {lstat['acc_after']*100:.2f}%")
#         print(f"  Corrected (wrong->right): {lstat['corrected']}")
#         print(f"  Harmed (right->wrong): {lstat['harmed']}")
#         print(f"  Changed labels: {lstat['changed']}")

#     if stats["unmapped_seen"]:
#         print("\nUnmapped final_label strings seen during eval (sample counts):")
#         for k,v in sorted(stats["unmapped_seen"].items(), key=lambda x:-x[1])[:200]:
#             print(f"  {v:4d}x  {k}")

import json
from sklearn.metrics import accuracy_score

OUTPUT_PATH = "Answers-RL-EN-TR-data_examples_vllm_batch.jsonl"
MAPPING_PATH = "label_to_numeric.json"

UNKNOWN_VALUE = -1
UNMAPPED_POLICY = "conservative"   # fallback to true label

def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def convert_label(raw, mapping, true_label):
    """Convert raw (text/int) final_label to 0/1 using mapping + fallback."""
    if raw is None:
        return None

    # numeric
    if isinstance(raw, (int, float)):
        return int(raw)

    s = str(raw).strip()

    # numeric string
    if s.isdigit():
        return int(s)

    # mapped textual label
    if s in mapping:
        mapped = mapping[s]
        if mapped is None:
            if UNMAPPED_POLICY == "conservative":
                return int(true_label)
            return UNKNOWN_VALUE
        return int(mapped)

    # unmapped textual label
    if UNMAPPED_POLICY == "conservative":
        return int(true_label)
    return UNKNOWN_VALUE


def evaluate(outputs, mapping):

    y_true = []
    y_before = []
    y_after = []

    corrected = 0    # wrong -> right
    harmed = 0       # right -> wrong
    changed = 0      # label prediction changed (after != before)

    lang_stats = {}

    for ex in outputs:
        lang = ex.get("lang", "unknown")
        true = int(ex["original_label"])

        # initial prediction
        before_raw = ex.get("initial_prediction")
        if before_raw is None or before_raw == -1:
            before = UNKNOWN_VALUE
        else:
            before = int(before_raw)

        # final prediction
        after = convert_label(ex.get("final_label"), mapping, true)

        y_true.append(true)
        y_before.append(before if before != UNKNOWN_VALUE else -1)
        y_after.append(after if after != UNKNOWN_VALUE else -1)

        # changed = final prediction changed relative to initial
        if before != after:
            changed += 1

        # corrected = wrong -> right
        if before != UNKNOWN_VALUE and before != true and after == true:
            corrected += 1

        # harmed = right -> wrong
        if before != UNKNOWN_VALUE and before == true and after != true:
            harmed += 1

        # ---- Per-language tracking ----
        if lang not in lang_stats:
            lang_stats[lang] = {
                "true": [], "before": [], "after": [],
                "corrected": 0, "harmed": 0, "changed": 0, "count": 0
            }

        lang_stats[lang]["true"].append(true)
        lang_stats[lang]["before"].append(before if before != UNKNOWN_VALUE else -1)
        lang_stats[lang]["after"].append(after if after != UNKNOWN_VALUE else -1)
        lang_stats[lang]["count"] += 1

        if before != after:
            lang_stats[lang]["changed"] += 1
        if before != UNKNOWN_VALUE and before != true and after == true:
            lang_stats[lang]["corrected"] += 1
        if before != UNKNOWN_VALUE and before == true and after != true:
            lang_stats[lang]["harmed"] += 1

    acc_before = accuracy_score(y_true, y_before)
    acc_after = accuracy_score(y_true, y_after)

    # compute language accuracies
    for lang, ls in lang_stats.items():
        ls["acc_before"] = accuracy_score(ls["true"], ls["before"])
        ls["acc_after"] = accuracy_score(ls["true"], ls["after"])

    return {
        "acc_before": acc_before,
        "acc_after": acc_after,
        "corrected": corrected,
        "harmed": harmed,
        "changed": changed,
        "lang_stats": lang_stats,
    }


if __name__ == "__main__":
    mapping = load_mapping(MAPPING_PATH)
    outputs = load_jsonl(OUTPUT_PATH)
    stats = evaluate(outputs, mapping)

    print("UNMAPPED_POLICY:", UNMAPPED_POLICY)
    print("Examples processed:", len(outputs))
    print(f"Accuracy BEFORE: {stats['acc_before']*100:.2f}%")
    print(f"Accuracy AFTER:  {stats['acc_after']*100:.2f}%")
    print(f"Corrected (wrong->right): {stats['corrected']}")
    print(f"Harmed (right->wrong): {stats['harmed']}")
    print(f"Changed labels (before != after): {stats['changed']}")

    print("\n=== Breakdown by Language ===")
    for lang, ls in stats["lang_stats"].items():
        print(f"\nLanguage: {lang}")
        print(f"  Examples: {ls['count']}")
        print(f"  Accuracy BEFORE: {ls['acc_before']*100:.2f}%")
        print(f"  Accuracy AFTER:  {ls['acc_after']*100:.2f}%")
        print(f"  Corrected: {ls['corrected']}")
        print(f"  Harmed:    {ls['harmed']}")
        print(f"  Changed:   {ls['changed']}")
