#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=48:00:00
#SBATCH --array=0-179%8        # 180 total jobs (192 - 12 excluded), max 8 running simultaneously
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/logs/mbbq_sweep_%A_%a.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/logs/mbbq_sweep_%A_%a.err

# ============================================================================
# Environment Setup
# ============================================================================
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/logs
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/src/results

source /ocean/projects/cis250182p/pgoyal2/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250182p/pgoyal2/conda_envs/MBBQEnv


# Set cache directories
export HF_HOME=/ocean/projects/cis250182p/shared/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/cis250182p/shared/huggingface/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250182p/shared/huggingface/datasets
export VLLM_CACHE_DIR=/ocean/projects/cis250182p/shared/vllm_cache
export CONDA_PKGS_DIRS=/ocean/projects/cis250182p/pgoyal2/conda_pkgs

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR="/jet/home/pgoyal2/Reasoning-Bias/src/results"
DATA_DIR="/jet/home/pgoyal2/Reasoning-Bias/MBBQ_data"
MAIN_SCRIPT="/jet/home/pgoyal2/Reasoning-Bias/src/main.py"
BATCH_SIZE=2

# Models
# LLAMA_3_1="meta-llama/Llama-3.1-8B-Instruct"
# LLAMA_2="meta-llama/Llama-2-7b-chat-hf"
# DEEPSEEK_R1="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Languages
LANGUAGES=("en" "es" "tr" "nl")

# Categories
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

# ============================================================================
# Job Array Mapping (with exclusion)
# ============================================================================

# Original: 4 eval_modes × 4 languages × 12 categories = 192 jobs
# Excluding: short_answer, Llama 3.1, en (12 categories) = -12 jobs
# Total: 180 jobs

TOTAL_CATEGORIES=${#CATEGORIES[@]}
TOTAL_LANGUAGES=${#LANGUAGES[@]}

# Map compressed array index to actual configuration
# Jobs 0-11: short_answer with Llama 3.1, en - EXCLUDED (these would be 0-11)
# Jobs 0-35: short_answer with Llama 3.1, es/tr/nl (36 jobs, previously 12-47)
# Jobs 36-83: short_answer with Llama 2, all languages (48 jobs, previously 48-95)
# Jobs 84-131: cot with Llama 3.1, all languages (48 jobs, previously 96-143)
# Jobs 132-179: reasoning with DeepSeek-R1, all languages (48 jobs, previously 144-191)

if [ $SLURM_ARRAY_TASK_ID -lt 36 ]; then
    # Jobs 0-35: short_answer with Llama 3.1, es/tr/nl only
    EVAL_MODE="short_answer"
    MODEL=$LLAMA_3_1
    # Add 12 to skip the 'en' language block (0-11)
    LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 12))
    LANGUAGE_IDX=$((LOCAL_TASK_ID / TOTAL_CATEGORIES))
    CATEGORY_IDX=$((LOCAL_TASK_ID % TOTAL_CATEGORIES))
elif [ $SLURM_ARRAY_TASK_ID -lt 84 ]; then
    # Jobs 36-83: short_answer with Llama 2, all languages
    EVAL_MODE="short_answer"
    MODEL=$LLAMA_2
    LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID - 36))
    LANGUAGE_IDX=$((LOCAL_TASK_ID / TOTAL_CATEGORIES))
    CATEGORY_IDX=$((LOCAL_TASK_ID % TOTAL_CATEGORIES))
elif [ $SLURM_ARRAY_TASK_ID -lt 132 ]; then
    # Jobs 84-131: cot with Llama 3.1, all languages
    EVAL_MODE="cot"
    MODEL=$LLAMA_3_1
    LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID - 84))
    LANGUAGE_IDX=$((LOCAL_TASK_ID / TOTAL_CATEGORIES))
    CATEGORY_IDX=$((LOCAL_TASK_ID % TOTAL_CATEGORIES))
elif [ $SLURM_ARRAY_TASK_ID -lt 180 ]; then
    # Jobs 132-179: reasoning with DeepSeek-R1, all languages
    EVAL_MODE="reasoning"
    MODEL=$DEEPSEEK_R1
    LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID - 132))
    LANGUAGE_IDX=$((LOCAL_TASK_ID / TOTAL_CATEGORIES))
    CATEGORY_IDX=$((LOCAL_TASK_ID % TOTAL_CATEGORIES))
else
    echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

LANGUAGE=${LANGUAGES[$LANGUAGE_IDX]}
CATEGORY=${CATEGORIES[$CATEGORY_IDX]}

# ============================================================================
# Create output directory
# ============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "/jet/home/pgoyal2/Reasoning-Bias/logs"

# ============================================================================
# Run Experiment
# ============================================================================

echo "========================================================================"
echo "SLURM Job: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo "Eval Mode:  $EVAL_MODE"
echo "Model:      $MODEL"
echo "Language:   $LANGUAGE"
echo "Category:   $CATEGORY"
echo "========================================================================"

python $MAIN_SCRIPT \
    --model_name "$MODEL" \
    --eval_mode "$EVAL_MODE" \
    --language "$LANGUAGE" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR"

EXIT_CODE=$?

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi

echo "========================================================================"

exit $EXIT_CODE