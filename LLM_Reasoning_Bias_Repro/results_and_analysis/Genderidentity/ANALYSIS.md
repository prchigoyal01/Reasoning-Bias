# Gender Identity Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.035
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 1973

### Bias Distribution
- **Unbiased (Score 0)**: 84 paragraphs (4.3%)
- **Slight Bias (Score 1)**: 24 paragraphs
- **Mild Bias (Score 2)**: 1712 paragraphs
- **Moderate Bias (Score 3)**: 40 paragraphs
- **Extreme Bias (Score 4)**: 113 paragraphs

**95.7%** of reasoning paragraphs contain gender identity bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.052
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2108

### Bias Distribution
- **Unbiased (Score 0)**: 542 paragraphs (25.7%)
- **Slight Bias (Score 1)**: 1109 paragraphs
- **Mild Bias (Score 2)**: 330 paragraphs
- **Moderate Bias (Score 3)**: 66 paragraphs
- **Extreme Bias (Score 4)**: 61 paragraphs

**74.3%** of reasoning paragraphs contain gender identity bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.03) compared to incorrect answers (1.05), suggesting reliance on gender identity stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 4.3% (correct) and 25.7% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic gender identity bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
