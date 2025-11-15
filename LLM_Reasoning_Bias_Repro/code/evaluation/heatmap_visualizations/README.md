# Bias Heatmap Visualizations

This directory contains automated visualizations of LLM-as-judge bias scores across all bias categories.

## Directory Structure

```
heatmap_visualizations/
├── consolidated/              # Cross-category comparative visualizations
│   ├── comparison_correct_vs_incorrect.pdf/png
│   ├── comparison_ambig_vs_disambig.pdf/png
│   ├── bias_distributions.pdf/png
│   ├── bias_delta_correct_minus_incorrect.pdf/png
│   └── summary_statistics_table.pdf/png
├── Age/                       # Category-specific heatmaps
├── Disabilitystatus/
├── Genderidentity/
├── Nationality/
├── Raceethnicity/
├── Religion/
├── Ses/
├── Sexualorientation/
└── statistics_summary.txt     # Numerical summary of all statistics
```

## Consolidated Visualizations

### 1. **comparison_correct_vs_incorrect.pdf**
Bar chart comparing mean bias scores between correct and incorrect answers across all categories.
- **Green bars**: Correct answers (equal_equal)
- **Red bars**: Incorrect answers (equal_not_equal)
- **Key finding**: Correct answers consistently show ~1 point higher bias than incorrect answers

### 2. **comparison_ambig_vs_disambig.pdf**
Bar chart comparing bias in ambiguous vs disambiguated contexts (for incorrect answers only).
- **Blue bars**: Ambiguous context
- **Purple bars**: Disambiguated context
- **Key finding**: Disambiguated contexts show higher bias (~0.3-0.7 points more)

### 3. **bias_distributions.pdf**
Violin plots showing the full distribution of bias scores across all categories and conditions.
- Shows spread, median, and mean for each category-condition pair
- Reveals which categories have more consistent vs variable bias patterns

### 4. **bias_delta_correct_minus_incorrect.pdf**
Horizontal bar chart showing the difference in bias between correct and incorrect answers.
- **Green bars**: Correct answers more biased
- **Red bars**: Incorrect answers more biased
- **Key insight**: All categories show positive delta (correct answers more biased)

### 5. **summary_statistics_table.pdf**
Comprehensive table with all statistics:
- Mean ± Standard Deviation for each condition
- Ambiguous and Disambiguated context breakdowns
- Bias delta (Δ Bias) column

## Category-Specific Heatmaps

Each category folder contains:
- `{category}_correct.pdf/png`: Heatmap for correct answers
- `{category}_incorrect.pdf/png`: Heatmap for incorrect answers
- `{category}_incorrect_ambig.pdf/png`: Incorrect answers in ambiguous contexts
- `{category}_incorrect_disambig.pdf/png`: Incorrect answers in disambiguated contexts

### Heatmap Interpretation

**Color Scale:**
- White (0): No bias detected
- Light pink (1): Slight bias
- Medium pink (2): Mild bias
- Red (3): Moderate bias
- Dark red (4): Extreme bias

**Axes:**
- X-axis: Normalized reasoning step (0 = start, 1 = end)
- Y-axis: Question index (each row = one example)

## Key Findings

### Overall Patterns

1. **Correct answers are more biased**: Mean bias ranges from 2.0-2.5
2. **Incorrect answers are less biased**: Mean bias ranges from 1.1-1.6
3. **Disambiguated contexts amplify bias**: ~30-50% higher than ambiguous contexts
4. **Most biased categories**: Religion (2.49), Nationality (2.43), Disability Status (2.43)
5. **Least biased incorrect answers**: Gender Identity (1.05), Age (1.13)

### Context Effects

- **Ambiguous contexts**: Lower bias (0.7-1.3 range)
- **Disambiguated contexts**: Higher bias (1.3-2.0 range)
- **Implication**: Clear context information triggers more stereotypical reasoning

## Generation Scripts

- **`generate_all_heatmaps.py`**: Generates individual category heatmaps
- **`plot_consolidated_results.py`**: Creates cross-category comparative visualizations

### Regenerate All Visualizations

```bash
cd /path/to/evaluation
python generate_all_heatmaps.py
python plot_consolidated_results.py
```

## Files Format

All visualizations are saved in both formats:
- **PDF**: High-resolution vector graphics (600 DPI) for publications
- **PNG**: Raster images (300 DPI) for presentations and web use

## Statistics Summary

See `statistics_summary.txt` for detailed numerical breakdowns including:
- Example counts
- Mean bias scores
- Standard deviations
- Min/max ranges

For each category and condition combination.
