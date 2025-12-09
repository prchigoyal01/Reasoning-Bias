import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

# Usage: python3 eval_adbp_csv2.py --csv_path ADBP_CSV_2.csv

def normalize_label(label):
    """
    Normalize bias labels to 'biased' or 'unbiased'.
    Accepts various forms: 'yes', 'no', 'Bias detected', etc.
    """
    label = str(label).strip().lower()
    if 'yes' in label or 'bias' in label:
        return 'biased'
    if 'no' in label or 'unbiased' in label:
        return 'unbiased'
    return 'unknown'

def main(csv_path):
    df = pd.read_csv(csv_path)

    # Try to find prediction and ground truth columns
    # Prediction: status, bias_score, judge_log, or similar
    # Ground truth: label
    pred_col = None
    for col in ['status', 'bias_score', 'judge_log']:
        if col in df.columns:
            pred_col = col
            break
    if not pred_col:
        print("No prediction column found. Available columns:", df.columns)
        return
    gt_col = 'label' if 'label' in df.columns else None

    preds = df[pred_col].apply(normalize_label)
    print(f"Prediction column: {pred_col}")
    if gt_col:
        gts = df[gt_col].apply(normalize_label)
        print(f"Ground truth column: {gt_col}")
    else:
        gts = None
        print("No ground truth column found. Only prediction distribution will be shown.")

    # Show prediction distribution
    pred_counts = Counter(preds)
    print("\nPrediction distribution:")
    for k, v in pred_counts.items():
        print(f"  {k}: {v}")

    # If ground truth available, compute metrics
    if gts is not None and set(gts) <= {'biased', 'unbiased'}:
        mask = (gts != 'unknown') & (preds != 'unknown')
        y_true = gts[mask]
        y_pred = preds[mask]
        if len(y_true) == 0:
            print("No valid ground truth labels for evaluation.")
            return
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label='biased')
        print("\nEvaluation metrics:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision (biased): {prec:.3f}")
        print(f"  Recall (biased): {rec:.3f}")
        print(f"  F1 (biased): {f1:.3f}")
        # Show some errors
        errors = df[mask][(y_true != y_pred)]
        if not errors.empty:
            print("\nExample errors:")
            print(errors[[pred_col, gt_col, 'prompt']].head(5))
    else:
        print("No valid ground truth for metric computation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/MBBQ_RL/LLM_ADBP2.csv', help='Path to the CSV file to evaluate')
    args = parser.parse_args()
    main(args.csv_path)
