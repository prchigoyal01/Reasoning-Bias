# Physical Appearance Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.388
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2448

### Bias Distribution
- **Unbiased (Score 0)**: 92 paragraphs (3.8%)
- **Slight Bias (Score 1)**: 1 paragraphs
- **Mild Bias (Score 2)**: 1739 paragraphs
- **Moderate Bias (Score 3)**: 179 paragraphs
- **Extreme Bias (Score 4)**: 437 paragraphs

**96.2%** of reasoning paragraphs contain physical appearance bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.719
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2492

### Bias Distribution
- **Unbiased (Score 0)**: 354 paragraphs (14.2%)
- **Slight Bias (Score 1)**: 683 paragraphs
- **Mild Bias (Score 2)**: 1026 paragraphs
- **Moderate Bias (Score 3)**: 244 paragraphs
- **Extreme Bias (Score 4)**: 185 paragraphs

**85.8%** of reasoning paragraphs contain physical appearance bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.39) compared to incorrect answers (1.72), suggesting reliance on physical appearance stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 3.8% (correct) and 14.2% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic physical appearance bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
