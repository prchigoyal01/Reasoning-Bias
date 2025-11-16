#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250182p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=48:00:00
#SBATCH --output=/jet/home/pgoyal2/Reasoning-Bias/logs/rl_run_%j.out
#SBATCH --error=/jet/home/pgoyal2/Reasoning-Bias/logs/rl_run_%j.err

# ============================================================================
# Environment Setup
# ============================================================================
mkdir -p /jet/home/pgoyal2/Reasoning-Bias/logs

source /ocean/projects/cis250182p/pgoyal2/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250182p/pgoyal2/conda_envs/MBBQEnv

# Optional: Speed up HF loading
export HF_HOME=/ocean/projects/cis250182p/shared/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/cis250182p/shared/huggingface/transformers
export HF_DATASETS_CACHE=/ocean/projects/cis250182p/shared/huggingface/datasets
export VLLM_CACHE_DIR=/ocean/projects/cis250182p/shared/vllm_cache
export CONDA_PKGS_DIRS=/ocean/projects/cis250182p/pgoyal2/conda_pkgs

# ============================================================================
# Run Script
# ============================================================================

SCRIPT="/jet/home/pgoyal2/Reasoning-Bias/BiasGuard/generate_rl_data.py"

echo "Running generate_rl_data.py on H100..."
python $SCRIPT

EXIT_CODE=$?

# Clear GPU memory (optional)
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
