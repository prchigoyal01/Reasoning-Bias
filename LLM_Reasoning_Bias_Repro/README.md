# Replication: Does Reasoning Introduce Bias?

This repository replicates the experiments from the paper ["Does Reasoning Introduce Bias? An Empirical Analysis of Chain-of-Thought in Social Bias Benchmarks"](https://arxiv.org/pdf/2502.15361) (Nian et al., 2025).

## Paper Summary

The paper investigates whether chain-of-thought (CoT) reasoning introduces or amplifies social biases in large language models, focusing on the BBQ (Bias Benchmark for QA) dataset across demographic categories like age, race, gender, sexual orientation, and religion.

**Key Findings:**
- CoT reasoning can introduce bias even when models answer correctly without reasoning
- Bias accumulates through reasoning steps (measured via ADBP - Answer Distribution as Bias Proxy)
- Larger models (32B) show more bias introduction than smaller models (8B)

## Repository Structure

```
LLM_Reasoning_Bias_Repro/
├── datasets/
│   └── bbq_csv/              # BBQ dataset in CSV format
│       ├── age.csv
│       ├── sexualOrientation.csv
│       ├── religion.csv
│       ├── disabilityStatus.csv
│       ├── physicalAppearance.csv
│       ├── nationality.csv
│       ├── genderIdentity.csv
│       ├── ses.csv
│       ├── raceEthnicity.csv
│       ├── raceXSes.csv
│       └── raceXGender.csv
│
└── code/
    ├── evaluation/
    │   ├── generate_cot_responses.py    # Generate CoT reasoning for BBQ examples
    │   ├── llama_as_a_judge.py          # Evaluate bias in reasoning steps
    │   ├── llm_as_judge_heatmaps.ipynb  # Visualize bias scores
    │   └── eval_results/                # Generated CoT responses (CSV)
    │
    └── mitigation/                       # Bias mitigation strategies (future work)
```

## Setup

### Requirements

```bash
# Create conda environment
conda create -n reasoning-bias python=3.10
conda activate reasoning-bias

# Install dependencies
pip install torch transformers datasets pandas tqdm bitsandbytes accelerate huggingface-hub
```

### Hugging Face Authentication

For gated models (LLaMA):
```bash
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

## Usage

### 1. Generate Chain-of-Thought Responses

Generate CoT reasoning for BBQ dataset examples:

```bash
cd code/evaluation

# Single category (test run)
python generate_cot_responses.py \
  --input_csv ../../datasets/bbq_csv/age.csv \
  --max_examples 10 \
  --max_new_tokens 512 \
  --batch_size 4

# Full dataset with optimized settings
python generate_cot_responses.py \
  --input_csv ../../datasets/bbq_csv/sexualOrientation.csv \
  --max_new_tokens 512 \
  --batch_size 8

# Run all categories in background
nohup bash -c 'for csv in ../../datasets/bbq_csv/*.csv; do 
  python generate_cot_responses.py \
    --input_csv "$csv" \
    --max_new_tokens 512 \
    --batch_size 8; 
done' > generation.log 2>&1 &
```

**Parameters:**
- `--input_csv`: Path to BBQ CSV file
- `--max_examples`: Limit number of examples (for testing)
- `--max_new_tokens`: Max tokens to generate per response (512 is optimal)
- `--batch_size`: Batch size for generation (8 recommended for L40S GPU)
- `--model_name`: Model to use (default: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- `--use_flash_attn`: Enable Flash Attention 2 (requires flash-attn package)
- `--compile`: Compile model with torch.compile for speedup

**Output:**
- `eval_results/{Category}_llama3_8b_deepseek_8b_equal_equal.csv` - Ambiguous context (answer = "Can't be determined")
- `eval_results/{Category}_llama3_8b_deepseek_8b_equal_not_equal.csv` - Disambiguated context (specific answer)

### 2. Evaluate Bias with LLM-as-Judge

Use an LLM to score bias in reasoning steps:

```bash
# Update target files in llama_as_a_judge.py
# Edit the running_files list to include the generated CSVs

# To run bias evaluation
python llama_as_a_judge.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --majority_voting 5

# Mistral (no gating required)
python llama_as_a_judge.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --majority_voting 5

# Run in background
nohup python llama_as_a_judge.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --majority_voting 5 > judge.log 2>&1 &
```

**Output:**
- `llm_as_judge_results/{Category}_llama_majority_voting_5_original.jsonl` - Bias scores per reasoning step

### 3. Visualize Results

Open `llm_as_judge_heatmaps.ipynb` to generate heatmaps and analyze bias patterns.

## Dataset Details

**BBQ Format:**
- `context`: Background scenario
- `question`: Question about individuals
- `choices`: Three answer options (ans0, ans1, ans2)
- `answer`: Correct answer index (2 = "Can't be determined" for ambiguous)
- `category`: Demographic category (Age, Race, Gender, etc.)

**Split Logic:**
- `equal_equal`: Ambiguous context where answer should be "Can't be determined" (label=2)
- `equal_not_equal`: Disambiguated context with clear answer (label≠2)

## Performance Notes

**Generation Speed:**
- Batch size 8 + 512 tokens: ~15-20 sec/batch (8 examples)
- Full dataset (3,936 examples): ~3-4 hours
- With Flash Attention 2: 2-3x speedup (requires installation)

**Memory:**
- 8B model (8-bit): ~12-15GB VRAM
- 32B model (8-bit): ~35-40GB VRAM

## Replication Status

- ✅ CoT generation pipeline
- ✅ equal_equal / equal_not_equal split
- ✅ LLM-as-Judge bias scoring
- ✅ Predicted answer extraction
- ⚠️ ADBP (Answer Distribution as Bias Proxy) - In progress
- ⚠️ Stepwise evaluation - In progress

## Known Issues

1. **~7% of responses miss `<answer>` tag** due to 512 token limit
   - Increase to 768 tokens if needed for higher completion rate
2. **DeepSeek model uses `<think>` tags** in output
   - Can be stripped post-processing if needed
3. **Chat template artifacts** (`<｜Assistant｜>`) appear in some outputs

## Citation

```bibtex
@inproceedings{wu-etal-2025-reasoning,
    title = "Does Reasoning Introduce Bias? A Study of Social Bias Evaluation and Mitigation in {LLM} Reasoning",
    author = "Wu, Xuyang  and
      Nian, Jinming  and
      Wei, Ting-Ruen  and
      Tao, Zhiqiang  and
      Wu, Hsin-Tai  and
      Fang, Yi",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1006/",
    doi = "10.18653/v1/2025.findings-emnlp.1006",
    pages = "18534--18555",
    ISBN = "979-8-89176-335-7",
    abstract = "Recent advances in large language models (LLMs) have enabled automatic generation of chain-of-thought (CoT) reasoning, leading to strong performance on tasks such as math and code. However, when reasoning steps reflect social stereotypes (e.g., those related to gender, race or age), they can reinforce harmful associations and lead to misleading conclusions. We present the first systematic evaluation of social bias within LLM-generated reasoning, using the BBQ dataset to analyze both prediction accuracy and bias. Our study spans a wide range of mainstream reasoning models, including instruction-tuned and CoT-augmented variants of DeepSeek-R1 (8B/32B), ChatGPT, and other open-source LLMs. We quantify how biased reasoning steps correlate with incorrect predictions and often lead to stereotype expression. To mitigate reasoning-induced bias, we propose Answer Distribution as Bias Proxy (ADBP), a lightweight mitigation method that detects bias by tracking how model predictions change across incremental reasoning steps. ADBP outperforms a stereotype-free baseline in most cases, mitigating bias and improving the accuracy of LLM outputs."
}
```