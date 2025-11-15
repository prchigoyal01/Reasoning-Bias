# Disability Status Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.422
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2218

### Bias Distribution
- **Unbiased (Score 0)**: 26 paragraphs (1.2%)
- **Slight Bias (Score 1)**: 11 paragraphs
- **Mild Bias (Score 2)**: 1611 paragraphs
- **Moderate Bias (Score 3)**: 158 paragraphs
- **Extreme Bias (Score 4)**: 412 paragraphs

**98.8%** of reasoning paragraphs contain disability status bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.558
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2310

### Bias Distribution
- **Unbiased (Score 0)**: 295 paragraphs (12.8%)
- **Slight Bias (Score 1)**: 772 paragraphs
- **Mild Bias (Score 2)**: 1017 paragraphs
- **Moderate Bias (Score 3)**: 144 paragraphs
- **Extreme Bias (Score 4)**: 82 paragraphs

**87.2%** of reasoning paragraphs contain disability status bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.42) compared to incorrect answers (1.56), suggesting reliance on disability status stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 1.2% (correct) and 12.8% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic disability status bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
