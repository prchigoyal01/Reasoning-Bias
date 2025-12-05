#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=8:00:00
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ/logs/sft_training_%j.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ/logs/sft_training_%j.err

# ============================================================================
# Environment Setup
# ============================================================================
SCRIPT_DIR="/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ"
# Ensure logs directory exists (SLURM needs it when processing SBATCH directives)
# This is a safety check - directory should be created before job submission
mkdir -p "${SCRIPT_DIR}/logs" || true

source /ocean/projects/cis250182p/pgoyal2/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250182p/pgoyal2/conda_envs/MBBQEnv

export HF_HOME=/ocean/projects/cis250182p/shared/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/cis250182p/shared/huggingface/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250182p/shared/huggingface/datasets
export VLLM_CACHE_DIR=/ocean/projects/cis250182p/shared/vllm_cache

# Set CUDA visible devices (use all available GPUs)
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# Configuration
# ============================================================================
TRAINING_SCRIPT="${SCRIPT_DIR}/sft_training_mbbq.py"

# Change to script directory
cd "$SCRIPT_DIR"

# ============================================================================
# Run Training
# ============================================================================
echo "========================================================================"
echo "SLURM Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo "Starting SFT Training for BiasGuard-MBBQ"
echo "========================================================================"

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" || true

# Run training
python "$TRAINING_SCRIPT"

EXIT_CODE=$?

# Cleanup
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Training completed"
    echo "Model saved to: ${SCRIPT_DIR}/mbbq_sft_model"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

exit $EXIT_CODE

