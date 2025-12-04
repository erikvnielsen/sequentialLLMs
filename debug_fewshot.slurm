#!/bin/bash
#SBATCH --job-name=fewshot_debug
#SBATCH --partition=spgpu        # Use A40 GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/fewshot_debug_%j.out

####################################
# MODULES & ENVIRONMENT
####################################
module purge
module load python/3.10.4
module load cuda/12.1.1

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

####################################
# INSTALL DEPENDENCIES
####################################
pip install -q --upgrade pip
pip install -q \
    transformers \
    datasets \
    accelerate \
    evaluate \
    sentence-transformers \
    sympy \
    --no-build-isolation

####################################
# RUN DEBUG SCRIPT
####################################
echo "=== Running Few-Shot Reasoning Debug ==="
srun python debug_distill_fewshot.py
