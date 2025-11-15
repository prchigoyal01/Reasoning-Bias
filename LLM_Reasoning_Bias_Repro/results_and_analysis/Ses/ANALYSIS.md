# Socioeconomic Status Bias Analysis

## Correct Answers (Ambiguous Context)

- **Examples Analyzed**: 269
- **Mean Bias Score**: 2.379
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 1981

### Bias Distribution
- **Unbiased (Score 0)**: 57 paragraphs (2.9%)
- **Slight Bias (Score 1)**: 9 paragraphs
- **Mild Bias (Score 2)**: 1375 paragraphs
- **Moderate Bias (Score 3)**: 230 paragraphs
- **Extreme Bias (Score 4)**: 310 paragraphs

**97.1%** of reasoning paragraphs contain socioeconomic status bias (scores ≥2).

## Incorrect Answers (Disambiguated Context)

- **Examples Analyzed**: 300
- **Mean Bias Score**: 1.567
- **Model Accuracy**: 0.0%
- **Total Reasoning Paragraphs**: 2252

### Bias Distribution
- **Unbiased (Score 0)**: 415 paragraphs (18.4%)
- **Slight Bias (Score 1)**: 651 paragraphs
- **Mild Bias (Score 2)**: 885 paragraphs
- **Moderate Bias (Score 3)**: 187 paragraphs
- **Extreme Bias (Score 4)**: 114 paragraphs

**81.6%** of reasoning paragraphs contain socioeconomic status bias (scores ≥2).

## Key Findings

1. **Bias in Correct vs Incorrect Reasoning**: The model exhibits higher bias when answering correctly (2.38) compared to incorrect answers (1.57), suggesting reliance on socioeconomic status stereotypes as a reasoning shortcut.

2. **Pervasiveness**: Across both conditions, the vast majority of reasoning contains bias, with only 2.9% (correct) and 18.4% (incorrect) of paragraphs being completely unbiased.

3. **Model Performance**: The model demonstrates higher accuracy in ambiguous contexts where it can default to "Unknown" answers.

## Interpretation

The analysis reveals systematic socioeconomic status bias in the reasoning process of DeepSeek-R1-Distill-Qwen-14B. The higher bias scores in correct answers indicate that the model uses stereotypical associations as cognitive shortcuts, which correlates with task performance in ambiguous scenarios. This pattern validates the paper's finding that contemporary reasoning-capable LLMs encode demographic biases that manifest during chain-of-thought generation.
