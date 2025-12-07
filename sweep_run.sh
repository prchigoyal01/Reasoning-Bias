#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/logs/biasguard_eval_%A_%a.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/logs/biasguard_eval_%A_%a.err

# ============================================================================
# Environment Setup
# ============================================================================
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/logs
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/src/results/biasguard_eval

source /ocean/projects/cis250182p/pgoyal2/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250182p/pgoyal2/conda_envs/MBBQEnv

export HF_HOME=/ocean/projects/cis250182p/shared/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/cis250182p/shared/huggingface/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250182p/shared/huggingface/datasets
export VLLM_CACHE_DIR=/ocean/projects/cis250182p/shared/vllm_cache

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR="/jet/home/pgoyal2/Reasoning-Bias/src/results/biasguard_eval"
EVAL_SCRIPT="/jet/home/pgoyal2/Reasoning-Bias/src/eval_biasguard_jsonl.py"
BATCH_SIZE=4
MAX_TOKENS=1024
TEMPERATURE=0.7
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
ADAPTER_PATH="/jet/home/pgoyal2/Reasoning-Bias/src/mbbq_rl_model_mix/checkpoint-1936"

DATASETS=(
    "/jet/home/pgoyal2/Reasoning-Bias/src/matched_test_examples.jsonl"
)

# ============================================================================
# Run Experiment
# ============================================================================
echo "========================================================================"
echo "SLURM Job: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo "Datasets: ${DATASETS[*]}"
echo "Eval Mode: BiasGuard JSONL"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

for DATASET in "${DATASETS[@]}"; do
    if [ ! -f "$DATASET" ]; then
        echo "Dataset not found: $DATASET"
        exit 1
    fi

    NAME=$(basename "$DATASET" .jsonl)
    OUTPUT_PATH="$OUTPUT_DIR/${NAME}_biasguard_eval.json"

    echo "Evaluating $DATASET ..."
    python "$EVAL_SCRIPT" \
        --input_file "$DATASET" \
        --output_file "$OUTPUT_PATH" \
        --base_model "$BASE_MODEL" \
        --adapter_path "$ADAPTER_PATH" \
        --batch_size $BATCH_SIZE \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE

    if [ $? -ne 0 ]; then
        echo "Evaluation failed for $DATASET"
        exit 1
    fi
done

EXIT_CODE=$?

# Cleanup
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
