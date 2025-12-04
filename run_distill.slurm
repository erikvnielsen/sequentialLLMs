#!/bin/bash
#SBATCH --job-name=distill_reasoning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=distill_%j.out

############################################
# ENVIRONMENT SETUP
############################################
module purge
module load python/3.10.4
module load cuda/12.2

# HuggingFace cache directories (prevents permission issues)
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub

############################################
# PERFORMANCE AND DEBUG SETTINGS
############################################
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# Prevent NCCL failures on UMich clusters
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

############################################
# PYTHON DEPENDENCIES
############################################
echo "Installing Python dependencies..."

pip install -q --upgrade pip

pip install -q \
    transformers \
    datasets \
    accelerate \
    evaluate \
    trl \
    peft \
    hf_transfer \
    rouge-score \
    sacrebleu \
    sentence-transformers \
    sympy \
    --no-build-isolation

############################################
# RUN TRAINING
############################################
echo "Starting multi-GPU distillation job..."

# Use accelerate for multi-GPU training
srun accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    multi_distill.py
