#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=24:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/logs/english_sweep_%A_%a.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/logs/english_sweep_%A_%a.err

# ============================================================================
# Environment Setup
# ============================================================================
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/logs
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/src/results

source /ocean/projects/cis250182p/pgoyal2/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250182p/pgoyal2/conda_envs/MBBQEnv

export HF_HOME=/ocean/projects/cis250182p/shared/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/cis250182p/shared/huggingface/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250182p/shared/huggingface/datasets
export VLLM_CACHE_DIR=/ocean/projects/cis250182p/shared/vllm_cache

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR="/jet/home/pgoyal2/Reasoning-Bias/src/results"
DATA_DIR="/jet/home/pgoyal2/Reasoning-Bias/MBBQ_data"
MAIN_SCRIPT="/jet/home/pgoyal2/Reasoning-Bias/src/main.py"
BATCH_SIZE=

LANGUAGE="en"

CATEGORIES=(
    "Age"
    "Age_control"
    "Disability_status"
    "Disability_status_control"
    "Gender_identity"
    "Gender_identity_control"
    "Physical_appearance"
    "Physical_appearance_control"
    "SES"
    "SES_control"
    "Sexual_orientation"
    "Sexual_orientation_control"
)

TOTAL_CATEGORIES=${#CATEGORIES[@]}

CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}

# ============================================================================
# Run Experiment
# ============================================================================
echo "========================================================================"
echo "SLURM Job: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo "Language:  $LANGUAGE"
echo "Category:  $CATEGORY"
echo "Eval Mode: short_answer"
echo "========================================================================"

python $MAIN_SCRIPT \
    --eval_mode "short_answer" \
    --language "$LANGUAGE" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR"

EXIT_CODE=$?

# Cleanup
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
