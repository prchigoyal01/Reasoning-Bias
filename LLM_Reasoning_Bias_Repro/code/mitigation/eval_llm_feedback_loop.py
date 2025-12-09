import pandas as pd
import argparse
from collections import Counter

def extract_label(row):
    # Use the bias_score column for label
    if row['bias_score'] == 1:
        return 'biased'
    elif row['bias_score'] == 0:
        return 'unbiased'
    else:
        return 'unknown'

def main(csv_path):
    df = pd.read_csv(csv_path)
    # True labels (if available)
    if 'label' in df.columns:
        true_labels = df['label'].apply(lambda x: 'biased' if x == 1 else ('unbiased' if x == 0 else 'unknown'))
    else:
        true_labels = None
    pred_labels = df.apply(extract_label, axis=1)
    # Basic counts
    counts = Counter(pred_labels)
    print('Prediction counts:', dict(counts))
    # If true labels available, compute metrics
    if true_labels is not None:
        correct = (true_labels == pred_labels)
        accuracy = correct.mean()
        print(f'Accuracy: {accuracy:.3f}')
        # Precision, recall, F1 for "biased" class
        tp = ((true_labels == 'biased') & (pred_labels == 'biased')).sum()
        fp = ((true_labels != 'biased') & (pred_labels == 'biased')).sum()
        fn = ((true_labels == 'biased') & (pred_labels != 'biased')).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f'Precision (biased): {precision:.3f}')
        print(f'Recall (biased): {recall:.3f}')
        print(f'F1 (biased): {f1:.3f}')
    # Show a few errors
    if true_labels is not None:
        errors = df[true_labels != pred_labels]
        print(f'Num errors: {len(errors)}')
        if len(errors) > 0:
            print(errors[['example_id','prompt','label','bias_score','judge_log']].head())

        # Per-language breakdown
        if 'lang' in df.columns:
            print('\nPer-language breakdown:')
            langs = df['lang'].unique()
            for lang in langs:
                mask = df['lang'] == lang
                lang_preds = pred_labels[mask]
                print(f'  {lang}: {Counter(lang_preds)}')
                if true_labels is not None:
                    lang_true = true_labels[mask]
                    lang_acc = (lang_true == lang_preds).mean()
                    print(f'    Accuracy: {lang_acc:.3f}')

        # Accuracy over only known predictions
        known_mask = pred_labels.isin(['biased','unbiased'])
        if true_labels is not None:
            known_acc = (true_labels[known_mask] == pred_labels[known_mask]).mean() if known_mask.any() else 0.0
            print(f'\nAccuracy over only known predictions: {known_acc:.3f} (n={known_mask.sum()})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to LLM_feedback_loop.csv')
    args = parser.parse_args()
    main(args.csv_path)
