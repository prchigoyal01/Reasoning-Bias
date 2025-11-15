#!/usr/bin/env python3
"""
Generate heatmaps for all LLM-as-judge result files.
Systematically processes all files and saves visualizations with descriptive names.
"""

import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from pathlib import Path

# Configuration
JUDGE_RESULTS_DIR = "llm_as_judge_results"
OUTPUT_DIR = "heatmap_visualizations"
JUDGE_KEY = "judge_responses"  # Key for judge responses in JSONL files

# Colormap configuration
COLORS = [
    "#ffffff",  # Bias 0
    "#ffb3b3",  # Bias 1
    "#ff6666",  # Bias 2
    "#cc0000",  # Bias 3
    "#660000"   # Bias 4
]
CMAP = mcolors.ListedColormap(COLORS)
BOUNDS = [0, 1, 2, 3, 4, 5]
NORM = mcolors.BoundaryNorm(BOUNDS, CMAP.N)
CBAR_TICKS = [0.5, 1.5, 2.5, 3.5, 4.5]
CBAR_LABELS = ["Bias 0", "Bias 1", "Bias 2", "Bias 3", "Bias 4"]


def load_bias_heatmap_data(file_path, num_points=100, judge_key=JUDGE_KEY):
    """Load and normalize bias scores for standard heatmap."""
    heatmap_rows = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps = entry.get(judge_key, [])
            bias_scores = [s["bias_score"] for s in steps if "bias_score" in s]
            if len(bias_scores) < 2:
                continue
            original_x = np.linspace(0, 1, len(bias_scores))
            target_x = np.linspace(0, 1, num_points)
            interp_fn = interp1d(original_x, bias_scores, kind='nearest', fill_value='extrapolate')
            normalized_bias = interp_fn(target_x)
            heatmap_rows.append(normalized_bias)
    return np.array(heatmap_rows)


def load_bias_heatmap_by_condition(file_path, num_points=100, judge_key=JUDGE_KEY):
    """Load bias scores separated by context condition (ambig/disambig)."""
    ambig_rows = []
    disambig_rows = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps = entry.get(judge_key, [])
            bias_scores = [s["bias_score"] for s in steps if "bias_score" in s]
            condition = entry.get("context_condition", None)
            if len(bias_scores) < 2 or condition not in ("ambig", "disambig"):
                continue
            original_x = np.linspace(0, 1, len(bias_scores))
            target_x = np.linspace(0, 1, num_points)
            interp_fn = interp1d(original_x, bias_scores, kind='nearest', fill_value='extrapolate')
            normalized_bias = interp_fn(target_x)
            if condition == "ambig":
                ambig_rows.append(normalized_bias)
            elif condition == "disambig":
                disambig_rows.append(normalized_bias)
    return np.array(ambig_rows), np.array(disambig_rows)


def plot_heatmap(data, title, save_path, figsize=(8, 4), title_fontsize=24, 
                 title_fontweight='bold', axis_labelsize=20, cbar_ticksize=12):
    """Generate and save a heatmap visualization."""
    if len(data) == 0:
        print(f"  ⚠️  No data to plot for {save_path}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect='auto', cmap=CMAP, norm=NORM)
    ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=20)
    ax.set_xlabel("Normalized Reasoning Step", fontsize=axis_labelsize)
    ax.set_ylabel("Question Index", fontsize=axis_labelsize)
    cbar = fig.colorbar(im, ax=ax, ticks=CBAR_TICKS, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(CBAR_LABELS, fontsize=cbar_ticksize)
    cbar.ax.tick_params(labelsize=cbar_ticksize)
    plt.tight_layout()
    
    # Save as both PNG and PDF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def parse_filename(filename):
    """
    Extract metadata from filename.
    Format: Category_model1_model2_condition_judge_eval_type.jsonl
    Example: Sexualorientation_llama3_8b_deepseek_8b_equal_equal_llama_2_7b_chat_hf_single_eval_original.jsonl
    """
    # Remove .jsonl extension
    base = filename.replace('.jsonl', '')
    
    # Extract category (everything before first model name pattern)
    # Common patterns: llama, deepseek, gpt, qwen, etc.
    match = re.match(r'^([A-Za-z]+)_', base)
    category = match.group(1) if match else "Unknown"
    
    # Extract condition (equal_equal or equal_not_equal)
    if 'equal_equal' in base:
        condition = 'equal_equal'
        condition_name = 'correct'
    elif 'equal_not_equal' in base:
        condition = 'equal_not_equal'
        condition_name = 'incorrect'
    else:
        condition = 'unknown'
        condition_name = 'unknown'
    
    return {
        'category': category,
        'condition': condition,
        'condition_name': condition_name,
        'filename': filename
    }


def compute_statistics(data, name):
    """Compute and print bias statistics."""
    if len(data) == 0:
        print(f"{name}: No data available")
        return None
    
    per_row_avg = data.mean(axis=1)
    stats = {
        'count': len(per_row_avg),
        'mean': per_row_avg.mean(),
        'std': per_row_avg.std(),
        'min': per_row_avg.min(),
        'max': per_row_avg.max()
    }
    
    print(f"{name}:")
    print(f"  Examples: {stats['count']}")
    print(f"  Mean bias: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return stats


def process_all_files():
    """Process all JSONL files in the judge results directory."""
    # Find all JSONL files
    pattern = os.path.join(JUDGE_RESULTS_DIR, "*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No JSONL files found in {JUDGE_RESULTS_DIR}")
        return
    
    print(f"Found {len(files)} files to process\n")
    
    # Group files by category
    files_by_category = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        metadata = parse_filename(filename)
        category = metadata['category']
        
        if category not in files_by_category:
            files_by_category[category] = {'correct': None, 'incorrect': None}
        
        if metadata['condition_name'] == 'correct':
            files_by_category[category]['correct'] = file_path
        elif metadata['condition_name'] == 'incorrect':
            files_by_category[category]['incorrect'] = file_path
    
    # Process each category
    all_stats = {}
    
    for category in sorted(files_by_category.keys()):
        print(f"\n{'='*60}")
        print(f"Processing: {category}")
        print(f"{'='*60}")
        
        category_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        files = files_by_category[category]
        category_stats = {}
        
        # Process correct answers
        if files['correct']:
            print(f"\n📊 Correct Answers (equal_equal)")
            try:
                data = load_bias_heatmap_data(files['correct'])
                plot_heatmap(
                    data,
                    f"{category} - Correct Answers",
                    os.path.join(category_dir, f"{category}_correct.pdf")
                )
                stats = compute_statistics(data, "Statistics")
                category_stats['correct'] = stats
            except Exception as e:
                print(f"  ❌ Error processing correct answers: {e}")
        
        # Process incorrect answers
        if files['incorrect']:
            print(f"\n📊 Incorrect Answers (equal_not_equal)")
            try:
                data = load_bias_heatmap_data(files['incorrect'])
                plot_heatmap(
                    data,
                    f"{category} - Incorrect Answers",
                    os.path.join(category_dir, f"{category}_incorrect.pdf")
                )
                stats = compute_statistics(data, "Statistics")
                category_stats['incorrect'] = stats
                
                # Also generate context-separated heatmaps for incorrect answers
                print(f"\n📊 By Context Condition")
                ambig_data, disambig_data = load_bias_heatmap_by_condition(files['incorrect'])
                
                if len(ambig_data) > 0:
                    plot_heatmap(
                        ambig_data,
                        f"{category} - Incorrect (Ambiguous)",
                        os.path.join(category_dir, f"{category}_incorrect_ambig.pdf")
                    )
                    ambig_stats = compute_statistics(ambig_data, "Ambiguous Context")
                    category_stats['incorrect_ambig'] = ambig_stats
                
                if len(disambig_data) > 0:
                    plot_heatmap(
                        disambig_data,
                        f"{category} - Incorrect (Disambiguated)",
                        os.path.join(category_dir, f"{category}_incorrect_disambig.pdf")
                    )
                    disambig_stats = compute_statistics(disambig_data, "Disambiguated Context")
                    category_stats['incorrect_disambig'] = disambig_stats
                    
            except Exception as e:
                print(f"  ❌ Error processing incorrect answers: {e}")
        
        all_stats[category] = category_stats
    
    # Save summary statistics
    print(f"\n{'='*60}")
    print("Saving summary statistics...")
    print(f"{'='*60}")
    
    summary_file = os.path.join(OUTPUT_DIR, "statistics_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Bias Heatmap Statistics Summary\n")
        f.write("="*60 + "\n\n")
        
        for category in sorted(all_stats.keys()):
            f.write(f"\n{category}\n")
            f.write("-"*60 + "\n")
            
            for condition, stats in all_stats[category].items():
                if stats:
                    f.write(f"\n{condition.replace('_', ' ').title()}:\n")
                    f.write(f"  Examples: {stats['count']}\n")
                    f.write(f"  Mean bias: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    f.write(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]\n")
    
    print(f"✓ Summary saved to {summary_file}")
    print(f"\n{'='*60}")
    print(f"✅ All heatmaps generated in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_files()
