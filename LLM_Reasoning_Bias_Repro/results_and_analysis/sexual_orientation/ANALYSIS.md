# Sexual Orientation Bias Analysis

## Overview
Analysis of DeepSeek-R1-Distill-Llama-8B chain-of-thought reasoning on the BBQ Sexual Orientation dataset, evaluated using LLaMA-2-7B-chat as judge.

**Date:** November 15, 2025  
**Dataset:** BBQ Sexual Orientation (100 sampled examples)  
**Model:** DeepSeek-R1-Distill-Llama-8B (8-bit quantization)  
**Judge:** LLaMA-2-7B-chat-hf (single voting)

---

## Key Findings

### 1. Extremely High Bias Detection (82-85% Severe Bias)

Both **Equal-Equal** (ambiguous/correct) and **Equal-Not-Equal** (disambiguated/incorrect) show **82-85% of reasoning steps scored as Bias 4** (severe bias).

**Interpretation:** The DeepSeek-8B model is making strong stereotype-based assumptions about sexual orientation throughout its reasoning process.

---

### 2. Minimal Difference Between Correct and Incorrect Answers

| Metric | Equal-Equal (Correct) | Equal-Not-Equal (Incorrect) | Difference |
|--------|----------------------|----------------------------|------------|
| **Mean Bias Score** | 3.61 | 3.68 | 0.07 |
| **Std Bias Score** | 0.91 | 0.84 | -0.07 |
| **Severe Bias (Score 4)** | 82.5% | 85.0% | +2.5% |

**Interpretation:** Only a **0.07 difference** in mean bias - the model shows high bias regardless of whether it gets the answer right or wrong. This suggests the bias is **systematic in the reasoning process**, not just in wrong answers.

---

### 3. Bias Score Distribution

#### Equal-Equal (Ambiguous Context - Correct "Cannot be determined")
- **Examples:** 50 (19 ambiguous, 31 disambiguated)
- **Total reasoning steps evaluated:** 418
- **Distribution:**
  - Score 0 (No bias): 11 (2.6%)
  - Score 1 (Minimal bias): 0 (0.0%)
  - Score 2 (Moderate bias): 59 (14.1%)
  - Score 3 (Significant bias): 3 (0.7%)
  - Score 4 (Severe bias): **345 (82.5%)**

#### Equal-Not-Equal (Disambiguated Context - Incorrect specific answer)
- **Examples:** 50 (21 ambiguous, 29 disambiguated)
- **Total reasoning steps evaluated:** 394
- **Distribution:**
  - Score 0 (No bias): 9 (2.3%)
  - Score 1 (Minimal bias): 1 (0.3%)
  - Score 2 (Moderate bias): 38 (9.6%)
  - Score 3 (Significant bias): 11 (2.8%)
  - Score 4 (Severe bias): **335 (85.0%)**

---

### 4. Very Low "No Bias" Scores

- Only **2-3% of reasoning steps** received Bias Score 0 (no bias)
- Scores 1 and 3 are nearly absent (0-3%)
- The distribution is **heavily polarized** toward severe bias (4) with some moderate bias (2)

**Interpretation:** The model rarely produces unbiased reasoning about sexual orientation topics.

---

## Detailed Analysis

### Critical Issue: Pervasive Stereotyping

The DeepSeek-8B model demonstrates **pervasive stereotyping** about sexual orientation in its chain-of-thought reasoning. The LLaMA judge identified bias in:

1. **Relationship Stability Assumptions**
   - Making assumptions about commitment levels based on sexual orientation
   - Attributing different relationship timelines to LGBTQ+ vs. straight individuals

2. **Behavioral Pattern Attribution**
   - Assuming different priorities or values based on sexual orientation
   - Using "societal norms" as justification for differential treatment

3. **Systematic Bias Across Answer Types**
   - High bias appears even when the model correctly answers "Cannot be determined"
   - Suggests bias is embedded in the reasoning process itself, not just wrong conclusions

### Paper Replication Context

This aligns with the **"Does Reasoning Introduce Bias?"** hypothesis from the paper (arXiv:2502.15361):
- The verbose reasoning process **amplifies stereotypes** that might not appear in direct answer generation
- Chain-of-thought reasoning creates more opportunities for biased assumptions to manifest
- The model "thinks through" stereotypes explicitly rather than bypassing them

---

## Heatmap Visualizations

The following heatmaps show bias scores (0-4) across normalized reasoning steps for each question:

1. **`8b_equal_equal.pdf`** - Correct answers (ambiguous context)
2. **`8b_equal_not_equal.pdf`** - Incorrect answers (disambiguated context)
3. **`8b_ene_ambig.pdf`** - Incorrect answers, ambiguous context only
4. **`8b_ene_disambig.pdf`** - Incorrect answers, disambiguated context only

**Color Scheme:**
- White: Bias 0 (No bias)
- Light red: Bias 1 (Minimal bias)
- Medium red: Bias 2 (Moderate bias)
- Dark red: Bias 3 (Significant bias)
- Darkest red: Bias 4 (Severe bias)

**Observation:** The heatmaps show predominantly dark red (Bias 4) across most reasoning steps and questions, confirming the systematic nature of the bias.

---

## Methodology

### Data Generation
- **Model:** DeepSeek-R1-Distill-Llama-8B
- **Settings:** 8-bit quantization, batch_size=8, max_new_tokens=512, greedy decoding
- **Dataset:** BBQ Sexual Orientation CSV (864 total examples)
- **Splits:** 
  - Equal-equal: label=2 (ambiguous, "Cannot be determined" is correct)
  - Equal-not-equal: label≠2 (disambiguated, specific answer is correct)
- **Sampling:** 50 examples per split (random_state=42)

### Bias Evaluation
- **Judge Model:** LLaMA-2-7B-chat-hf
- **Voting:** Single evaluation per reasoning step (majority_voting=1)
- **Prompt:** Each reasoning paragraph evaluated for sexual orientation bias on 0-4 scale
- **Extraction:** Multi-method JSON parsing with fallbacks

### Known Limitations
- ~7% of CoT responses truncated at 512 tokens before completing answer tags
- Single voting mode (faster but less robust than majority voting)
- Judge model may have its own biases affecting evaluation

---

## Files in this Directory

- `8b_equal_equal.pdf` - Heatmap: Correct answers (equal-equal split)
- `8b_equal_not_equal.pdf` - Heatmap: Incorrect answers (equal-not-equal split)
- `8b_ene_ambig.pdf` - Heatmap: Incorrect answers, ambiguous context
- `8b_ene_disambig.pdf` - Heatmap: Incorrect answers, disambiguated context
- `ANALYSIS.md` - This analysis document

## Source Data

- CoT Responses: `code/evaluation/eval_results/Sexualorientation_llama3_8b_deepseek_8b_equal_equal.csv`
- CoT Responses: `code/evaluation/eval_results/Sexualorientation_llama3_8b_deepseek_8b_equal_not_equal.csv`
- Judge Results: `code/evaluation/llm_as_judge_results/Sexualorientation_llama3_8b_deepseek_8b_equal_equal_llama_single_eval_original.jsonl`
- Judge Results: `code/evaluation/llm_as_judge_results/Sexualorientation_llama3_8b_deepseek_8b_equal_not_equal_llama_single_eval_original.jsonl`
