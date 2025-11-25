# Reasoning-Bias

A comprehensive framework for detecting, analyzing, and mitigating social biases in Large Language Models (LLMs), with a focus on reasoning tasks and question answering.

## 📋 Overview

This repository combines multiple components to evaluate and address biases in LLM reasoning:

- **BBQ (Bias Benchmark for QA)**: A dataset highlighting social biases across multiple dimensions
- **BiasGuard**: A training pipeline for fine-tuning models to reduce bias through supervised fine-tuning (SFT) and direct preference optimization (DPO)
- **MBBQ Data**: Multilingual extensions and variations of the BBQ benchmark
- **Evaluation Framework**: Tools to assess model reasoning capabilities and bias in responses

## 🚀 Quick Start

### Environment Setup

Create a conda environment:
```bash
conda env create -f env.yml -n reasoningBias
conda activate reasoningBias
```

Or install dependencies manually:
```bash
bash setup.sh
```

### Dependencies

Key packages required:
- Python 3.12
- PyTorch with CUDA support
- Transformers (Hugging Face)
- DeepSeek models
- TRL (Transformers Reinforcement Learning)
- Datasets
- vLLM (for optimized inference)

See `env.yml` for the complete dependency list.

## 📁 Project Structure

### `/BBQ` - Bias Benchmark for QA
The official BBQ dataset and generation scripts.

**Key files:**
- `data/`: Generated BBQ datasets across 9 social bias dimensions
- `templates/`: Templates for generating bias benchmark questions
- `generate_from_template_*.py`: Scripts to generate benchmark questions
- `analysis_scripts/`: R scripts for bias analysis and evaluation
- `utils.py`: Utility functions for data processing

**Bias categories covered:**
- Age, Disability Status, Gender Identity
- Physical Appearance, Sexual Orientation, SES
- And more with control conditions

**Usage:**
```bash
cd BBQ
python generate_from_template_all_categories.py
```

See [BBQ README](BBQ/README.md) for detailed information.

### `/BiasGuard` - Bias Mitigation Training Pipeline

Debiasing pipeline using Supervised Fine-Tuning and Direct Preference Optimization.

**Key files:**
- `config.py`: Configuration for models and training parameters
- `generate_sft_data.py`: Generate SFT training data from raw datasets
- `sft_pipeline.py`: Supervised Fine-Tuning implementation
- `rl_training.py`: Direct Preference Optimization (DPO) for preference learning
- `clean_sft_data.py`: Data cleaning and validation
- `prompt_templates/`: JSON templates for bias detection and evaluation

**Configuration:**
Edit `config.py` to modify:
- Teacher model: `TEACHER_MODEL_NAME`
- SFT model: `SFT_MODEL_NAME`
- Batch size: `BATCH_SIZE`
- Data paths: `FINETUNE_DATA_PATH`

**Pipeline workflow:**
```
Raw Data → generate_sft_data.py → clean_sft_data.py → sft_pipeline.py → rl_training.py
```

**Usage:**
```bash
cd BiasGuard

# Step 1: Generate SFT training data
python generate_sft_data.py

# Step 2: Clean the generated data
python clean_sft_data.py

# Step 3: Run supervised fine-tuning
python sft_pipeline.py

# Step 4: Apply DPO for preference optimization
python rl_training.py
```

### `/src` - Evaluation Framework

Tools for evaluating model reasoning and bias in responses.

**Key components:**
- `main.py`: Main evaluation script with argument parsing
- `evaluators/`: Evaluation modules
  - `reasoning_eval.py`: Evaluates reasoning task performance
  - `short_answer_eval.py`: Evaluates short-form answers
- `utils/`: Utility functions for evaluation
- `results/`: Output directory for evaluation results

**Usage:**
```bash
cd src
python main.py --category Gender_identity --output_dir results
```

### `/DataAnalysis` - Data Analysis Tools

Analysis and visualization of evaluation results.

### `/MBBQ_data` - Multilingual BBQ

Extended BBQ datasets in multiple languages and formats.

## 🔬 Key Features

### 1. Bias Detection
- Identifies stereotyping in model outputs
- Tests both ambiguous and disambiguated contexts
- Evaluates if models override correct answers for biased responses

### 2. Multi-dimensional Bias Analysis
Covers 9+ social bias dimensions relevant to U.S. English contexts

### 3. Model Training Pipeline
- **SFT Stage**: Supervised fine-tuning on bias-annotated data
- **DPO Stage**: Direct preference optimization for preferred debiased responses
- Supports quantization (4-bit and 8-bit) for memory efficiency

### 4. Flexible Evaluation
- Supports multiple model types (DeepSeek, Llama, etc.)
- vLLM optimization for faster inference
- Batch processing support

## 🛠️ Advanced Usage

### Running Model Sweeps

Execute evaluation across multiple models:
```bash
bash sweep_mbbq.sh
```

This script tests models against the MBBQ dataset and generates comprehensive results.

### Tensorboard Logging

Training logs are saved to the `logs/` directory. View with:
```bash
tensorboard --logdir=BiasGuard/results
```

## 📊 Output Files

After running evaluation:
- `results/`: Contains model predictions and metrics
- `logs/`: Training/inference logs
- `*.jsonl`: Data in JSONL format for easy processing

## 📖 Related Work

This project builds on:
- **BBQ Paper**: "BBQ: A Hand-Built Bias Benchmark for Question Answering" (ACL 2022 Findings)
- **Reasoning Evaluation**: Custom reasoning task evaluators
- **Debiasing Methods**: SFT + DPO approach inspired by recent alignment literature

## 🔗 Key References

- BBQ Dataset: [GitHub Repository](https://github.com/nyu-mll/BBQ)
- DeepSeek Models: [Hugging Face Hub](https://huggingface.co/deepseek-ai)
- TRL Documentation: [Transformers Reinforcement Learning](https://github.com/huggingface/trl)

## 📝 Development Notes

- Python 3.12+ required for conda environment
- GPU/CUDA support recommended for training
- Models require HuggingFace authentication for access
- Batch processing with vLLM for optimized inference speed
---

**Last Updated:** November 2025
