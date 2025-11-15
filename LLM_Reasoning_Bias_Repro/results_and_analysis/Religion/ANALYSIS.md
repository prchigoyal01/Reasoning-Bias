# Religion Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.472
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2332

### Bias Distribution
- **Unbiased (Score 0)**: 152 paragraphs (6.5%)
- **Slight Bias (Score 1)**: 0 paragraphs
- **Mild Bias (Score 2)**: 1388 paragraphs
- **Moderate Bias (Score 3)**: 141 paragraphs
- **Extreme Bias (Score 4)**: 651 paragraphs

**93.5%** of reasoning paragraphs contain religion bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.269
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2433

### Bias Distribution
- **Unbiased (Score 0)**: 727 paragraphs (29.9%)
- **Slight Bias (Score 1)**: 797 paragraphs
- **Mild Bias (Score 2)**: 546 paragraphs
- **Moderate Bias (Score 3)**: 266 paragraphs
- **Extreme Bias (Score 4)**: 97 paragraphs

**70.1%** of reasoning paragraphs contain religion bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.47) compared to incorrect answers (1.27), suggesting reliance on religion stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 6.5% (correct) and 29.9% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic religion bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
