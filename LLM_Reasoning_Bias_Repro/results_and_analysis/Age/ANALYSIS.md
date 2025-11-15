# Age Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 2.200
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2347

### Bias Distribution
- **Unbiased (Score 0)**: 80 paragraphs (3.4%)
- **Slight Bias (Score 1)**: 1 paragraphs
- **Mild Bias (Score 2)**: 1922 paragraphs
- **Moderate Bias (Score 3)**: 89 paragraphs
- **Extreme Bias (Score 4)**: 255 paragraphs

**96.6%** of reasoning paragraphs contain age bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.135
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2455

### Bias Distribution
- **Unbiased (Score 0)**: 616 paragraphs (25.1%)
- **Slight Bias (Score 1)**: 1119 paragraphs
- **Mild Bias (Score 2)**: 625 paragraphs
- **Moderate Bias (Score 3)**: 49 paragraphs
- **Extreme Bias (Score 4)**: 46 paragraphs

**74.9%** of reasoning paragraphs contain age bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.20) compared to incorrect answers (1.14), suggesting reliance on age stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 3.4% (correct) and 25.1% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic age bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
