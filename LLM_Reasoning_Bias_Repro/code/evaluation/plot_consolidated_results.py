#!/usr/bin/env python3
"""
Generate consolidated visualizations comparing bias across all categories.
"""

import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
JUDGE_RESULTS_DIR = "llm_as_judge_results"
OUTPUT_DIR = "heatmap_visualizations/consolidated"
JUDGE_KEY = "judge_responses"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600


def parse_filename(filename):
    """Extract metadata from filename."""
    base = filename.replace('.jsonl', '')
    match = re.match(r'^([A-Za-z]+)_', base)
    category = match.group(1) if match else "Unknown"
    
    if 'equal_equal' in base:
        condition = 'correct'
    elif 'equal_not_equal' in base:
        condition = 'incorrect'
    else:
        condition = 'unknown'
    
    return category, condition


def compute_bias_statistics(file_path):
    """Compute mean bias score for all examples in a file."""
    bias_averages = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps = entry.get(JUDGE_KEY, [])
            bias_scores = [s["bias_score"] for s in steps if "bias_score" in s]
            if len(bias_scores) >= 2:
                bias_averages.append(np.mean(bias_scores))
    
    if len(bias_averages) == 0:
        return None
    
    return {
        'mean': np.mean(bias_averages),
        'std': np.std(bias_averages),
        'median': np.median(bias_averages),
        'count': len(bias_averages),
        'data': bias_averages
    }


def compute_context_statistics(file_path):
    """Compute bias statistics separated by context condition."""
    ambig_scores = []
    disambig_scores = []
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps = entry.get(JUDGE_KEY, [])
            bias_scores = [s["bias_score"] for s in steps if "bias_score" in s]
            condition = entry.get("context_condition", None)
            
            if len(bias_scores) >= 2:
                mean_bias = np.mean(bias_scores)
                if condition == "ambig":
                    ambig_scores.append(mean_bias)
                elif condition == "disambig":
                    disambig_scores.append(mean_bias)
    
    result = {}
    if len(ambig_scores) > 0:
        result['ambig'] = {
            'mean': np.mean(ambig_scores),
            'std': np.std(ambig_scores),
            'count': len(ambig_scores),
            'data': ambig_scores
        }
    if len(disambig_scores) > 0:
        result['disambig'] = {
            'mean': np.mean(disambig_scores),
            'std': np.std(disambig_scores),
            'count': len(disambig_scores),
            'data': disambig_scores
        }
    
    return result


def collect_all_statistics():
    """Collect statistics from all files."""
    pattern = os.path.join(JUDGE_RESULTS_DIR, "*.jsonl")
    files = glob.glob(pattern)
    
    data = defaultdict(dict)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        category, condition = parse_filename(filename)
        
        stats = compute_bias_statistics(file_path)
        if stats:
            data[category][condition] = stats
        
        # Also get context-separated stats for incorrect answers
        if condition == 'incorrect':
            context_stats = compute_context_statistics(file_path)
            if context_stats:
                data[category]['context'] = context_stats
    
    return data


def plot_comparison_bars(data, save_path):
    """Bar plot comparing correct vs incorrect bias across categories."""
    categories = sorted([k for k in data.keys() if 'correct' in data[k] or 'incorrect' in data[k]])
    
    correct_means = []
    correct_stds = []
    incorrect_means = []
    incorrect_stds = []
    
    for cat in categories:
        if 'correct' in data[cat]:
            correct_means.append(data[cat]['correct']['mean'])
            correct_stds.append(data[cat]['correct']['std'])
        else:
            correct_means.append(0)
            correct_stds.append(0)
        
        if 'incorrect' in data[cat]:
            incorrect_means.append(data[cat]['incorrect']['mean'])
            incorrect_stds.append(data[cat]['incorrect']['std'])
        else:
            incorrect_means.append(0)
            incorrect_stds.append(0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, correct_means, width, yerr=correct_stds, 
                   label='Correct Answers', color='#2ecc71', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, incorrect_means, width, yerr=incorrect_stds,
                   label='Incorrect Answers', color='#e74c3c', alpha=0.8, capsize=3)
    
    ax.set_xlabel('Bias Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Bias Score', fontsize=14, fontweight='bold')
    ax.set_title('Bias in Reasoning: Correct vs Incorrect Answers', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(correct_means), max(incorrect_means)) * 1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(save_path)}")


def plot_context_comparison(data, save_path):
    """Compare bias in ambiguous vs disambiguated contexts."""
    categories = sorted([k for k in data.keys() if 'context' in data[k]])
    
    ambig_means = []
    disambig_means = []
    ambig_stds = []
    disambig_stds = []
    
    for cat in categories:
        context_data = data[cat]['context']
        if 'ambig' in context_data:
            ambig_means.append(context_data['ambig']['mean'])
            ambig_stds.append(context_data['ambig']['std'])
        else:
            ambig_means.append(0)
            ambig_stds.append(0)
        
        if 'disambig' in context_data:
            disambig_means.append(context_data['disambig']['mean'])
            disambig_stds.append(context_data['disambig']['std'])
        else:
            disambig_means.append(0)
            disambig_stds.append(0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, ambig_means, width, yerr=ambig_stds,
                   label='Ambiguous Context', color='#3498db', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, disambig_means, width, yerr=disambig_stds,
                   label='Disambiguated Context', color='#9b59b6', alpha=0.8, capsize=3)
    
    ax.set_xlabel('Bias Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Bias Score', fontsize=14, fontweight='bold')
    ax.set_title('Bias in Reasoning: Ambiguous vs Disambiguated Context (Incorrect Answers)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(disambig_means), max(ambig_means)) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(save_path)}")


def plot_bias_distributions(data, save_path):
    """Violin plot showing distribution of bias scores across categories."""
    categories = sorted([k for k in data.keys() if 'correct' in data[k] or 'incorrect' in data[k]])
    
    plot_data = []
    labels = []
    colors = []
    
    for cat in categories:
        if 'correct' in data[cat]:
            plot_data.append(data[cat]['correct']['data'])
            labels.append(f"{cat}\n(Correct)")
            colors.append('#2ecc71')
        
        if 'incorrect' in data[cat]:
            plot_data.append(data[cat]['incorrect']['data'])
            labels.append(f"{cat}\n(Incorrect)")
            colors.append('#e74c3c')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bias Score Distribution', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Bias Scores Across Categories', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.5, 4.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(save_path)}")


def plot_bias_delta(data, save_path):
    """Plot the difference in bias between correct and incorrect answers."""
    categories = sorted([k for k in data.keys() if 'correct' in data[k] and 'incorrect' in data[k]])
    
    deltas = []
    for cat in categories:
        delta = data[cat]['correct']['mean'] - data[cat]['incorrect']['mean']
        deltas.append(delta)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#27ae60' if d > 0 else '#c0392b' for d in deltas]
    bars = ax.barh(categories, deltas, color=colors, alpha=0.8)
    
    ax.set_xlabel('Bias Difference (Correct - Incorrect)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Category', fontsize=14, fontweight='bold')
    ax.set_title('Bias Difference: Correct vs Incorrect Answers\n(Positive = More bias in correct answers)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        ax.text(val + (0.05 if val > 0 else -0.05), bar.get_y() + bar.get_height()/2,
               f'{val:.2f}',
               ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(save_path)}")


def plot_summary_statistics(data, save_path):
    """Create a comprehensive summary table visualization."""
    categories = sorted([k for k in data.keys() if 'correct' in data[k] or 'incorrect' in data[k]])
    
    fig, ax = plt.subplots(figsize=(14, len(categories) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Category', 'Correct\n(Mean±SD)', 'Incorrect\n(Mean±SD)', 
                      'Ambig\n(Mean±SD)', 'Disambig\n(Mean±SD)', 'Δ Bias'])
    
    for cat in categories:
        row = [cat]
        
        # Correct
        if 'correct' in data[cat]:
            row.append(f"{data[cat]['correct']['mean']:.2f}±{data[cat]['correct']['std']:.2f}")
        else:
            row.append('N/A')
        
        # Incorrect
        if 'incorrect' in data[cat]:
            row.append(f"{data[cat]['incorrect']['mean']:.2f}±{data[cat]['incorrect']['std']:.2f}")
        else:
            row.append('N/A')
        
        # Context stats
        if 'context' in data[cat]:
            ctx = data[cat]['context']
            if 'ambig' in ctx:
                row.append(f"{ctx['ambig']['mean']:.2f}±{ctx['ambig']['std']:.2f}")
            else:
                row.append('N/A')
            if 'disambig' in ctx:
                row.append(f"{ctx['disambig']['mean']:.2f}±{ctx['disambig']['std']:.2f}")
            else:
                row.append('N/A')
        else:
            row.extend(['N/A', 'N/A'])
        
        # Delta
        if 'correct' in data[cat] and 'incorrect' in data[cat]:
            delta = data[cat]['correct']['mean'] - data[cat]['incorrect']['mean']
            row.append(f"{delta:+.2f}")
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.18, 0.15, 0.15, 0.15, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
    
    plt.title('Bias Statistics Summary Across All Categories', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(save_path)}")


def main():
    """Generate all consolidated visualizations."""
    print("Collecting statistics from all categories...")
    data = collect_all_statistics()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nGenerating consolidated visualizations...")
    print("="*60)
    
    # 1. Comparison bar chart
    plot_comparison_bars(data, os.path.join(OUTPUT_DIR, "comparison_correct_vs_incorrect.pdf"))
    
    # 2. Context comparison
    plot_context_comparison(data, os.path.join(OUTPUT_DIR, "comparison_ambig_vs_disambig.pdf"))
    
    # 3. Distribution violin plots
    plot_bias_distributions(data, os.path.join(OUTPUT_DIR, "bias_distributions.pdf"))
    
    # 4. Bias delta (difference)
    plot_bias_delta(data, os.path.join(OUTPUT_DIR, "bias_delta_correct_minus_incorrect.pdf"))
    
    # 5. Summary table
    plot_summary_statistics(data, os.path.join(OUTPUT_DIR, "summary_statistics_table.pdf"))
    
    print("="*60)
    print(f"✅ All consolidated plots saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
