#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=12:00:00
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ/logs/rl_training_%j.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ/logs/rl_training_%j.err

# ============================================================================
# Environment Setup
# ============================================================================
SCRIPT_DIR="/jet/home/pgoyal2/Reasoning-Bias/BiasGuard-MBBQ"
# Ensure logs directory exists (SLURM needs it when processing SBATCH directives)
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
RL_DATA_SCRIPT="${SCRIPT_DIR}/generate_mbbq_rl_data.py"
DPO_TRAINING_SCRIPT="${SCRIPT_DIR}/dpo_training_mbbq.py"

# Change to script directory
cd "$SCRIPT_DIR"

# ============================================================================
# Run RL Data Generation and DPO Training
# ============================================================================
echo "========================================================================"
echo "SLURM Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" || true

# ============================================================================
# Step 1: Generate RL Data
# ============================================================================
# echo "========================================================================"
# echo "Step 1: Generating RL Training Data"
# echo "========================================================================"
# python "$RL_DATA_SCRIPT"

# EXIT_CODE_RL=$?

# if [ $EXIT_CODE_RL -ne 0 ]; then
#     echo "========================================================================"
#     echo "✗ FAILED - RL data generation failed (exit code: $EXIT_CODE_RL)"
#     echo "========================================================================"
#     exit $EXIT_CODE_RL
# fi

# echo "========================================================================"
# echo "✓ SUCCESS - RL data generation completed"
# echo "RL data saved to: ${SCRIPT_DIR}/mbbq_rl_data.jsonl"
# echo "========================================================================"

# Cleanup GPU memory between steps
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ============================================================================
# Step 2: DPO Training
# ============================================================================
echo "========================================================================"
echo "Step 2: Starting DPO Training"
echo "========================================================================"
python "$DPO_TRAINING_SCRIPT"

EXIT_CODE_DPO=$?

# Cleanup
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "========================================================================"
if [ $EXIT_CODE_DPO -eq 0 ]; then
    echo "SUCCESS - DPO training completed"
    echo "Model saved to: ${SCRIPT_DIR}/mbbq_rl_model"
else
    echo "FAILED - DPO training failed (exit code: $EXIT_CODE_DPO)"
fi
echo "========================================================================"

exit $EXIT_CODE_DPO

