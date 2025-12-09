import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

# Usage: python3 analyze_new_log_effect.py --csv_path LLM_ADBP2.csv

def normalize_label(label):
    label = str(label).strip().lower()
    if 'yes' in label or 'bias' in label:
        return 'biased'
    if 'no' in label or 'unbiased' in label:
        return 'unbiased'
    return 'unknown'

def has_new_log(log):
    return isinstance(log, str) and len(log.strip()) > 0


def main(csv_path):
    df = pd.read_csv(csv_path)
    pred_col = None
    for col in ['status', 'bias_score', 'judge_log']:
        if col in df.columns:
            pred_col = col
            break
    gt_col = 'label' if 'label' in df.columns else None
    if not pred_col or not gt_col:
        print("Required columns not found.")
        return
    # Print sample values for debugging
    print("Sample values from CSV:")
    print(df[[pred_col, gt_col, 'new_log']].head(5))
    # Only require new_log to be non-empty for analysis
    has_log = df['new_log'].apply(has_new_log)
    print(f"\nTotal rows: {len(df)}")
    print(f"Rows with non-empty new_log: {has_log.sum()}")
    print(f"Rows with empty new_log: {(~has_log).sum()}")
    # Show some examples with and without new_log
    print("\nExamples with non-empty new_log:")
    print(df[has_log][['prompt', 'new_log', pred_col, gt_col]].head(3))
    print("\nExamples with empty new_log:")
    print(df[~has_log][['prompt', 'new_log', pred_col, gt_col]].head(3))
    # If you want to compute metrics, relax normalization
    preds = df[pred_col]
    gts = df[gt_col]
    # Optionally, show value counts
    print(f"\nValue counts for {pred_col}:")
    print(preds.value_counts())
    print(f"\nValue counts for {gt_col}:")
    print(gts.value_counts())

    # --- Compute metrics from new_log ---
    def extract_pred(log):
        if pd.isnull(log):
            return None
        log = str(log).lower()
        if "unbiased" in log:
            return 0
        if "biased" in log:
            return 1
        return None

    df['pred'] = df['new_log'].apply(extract_pred)
    mask = df['pred'].notnull()
    y_true = df.loc[mask, gt_col]
    y_pred = df.loc[mask, 'pred']
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
        print(f"\nMetrics for bias mitigation (from new_log):")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision (biased): {prec:.3f}")
        print(f"  Recall (biased): {rec:.3f}")
        print(f"  F1 (biased): {f1:.3f}")
    else:
        print("\nNo valid predictions found in new_log for metric computation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='LLM_ADBP2.csv', help='Path to the CSV file to analyze')
    args = parser.parse_args()
    main(args.csv_path)
