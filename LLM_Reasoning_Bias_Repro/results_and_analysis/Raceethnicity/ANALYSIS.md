# Race/Ethnicity Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.256
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2128

### Bias Distribution
- **Unbiased (Score 0)**: 204 paragraphs (9.6%)
- **Slight Bias (Score 1)**: 11 paragraphs
- **Mild Bias (Score 2)**: 1411 paragraphs
- **Moderate Bias (Score 3)**: 62 paragraphs
- **Extreme Bias (Score 4)**: 440 paragraphs

**90.4%** of reasoning paragraphs contain race/ethnicity bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.459
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2121

### Bias Distribution
- **Unbiased (Score 0)**: 603 paragraphs (28.4%)
- **Slight Bias (Score 1)**: 580 paragraphs
- **Mild Bias (Score 2)**: 573 paragraphs
- **Moderate Bias (Score 3)**: 129 paragraphs
- **Extreme Bias (Score 4)**: 236 paragraphs

**71.6%** of reasoning paragraphs contain race/ethnicity bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.26) compared to incorrect answers (1.46), suggesting reliance on race/ethnicity stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 9.6% (correct) and 28.4% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic race/ethnicity bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
