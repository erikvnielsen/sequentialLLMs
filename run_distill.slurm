#!/bin/bash
#SBATCH --job-name=distill_reasoning
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill_%j.out

module purge
module load python/3.10.4
module load cuda/12.1.1

mkdir -p logs

########################################
# HuggingFace / caching / transfer
########################################
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub

# Prevent hf_transfer crashing by disabling acceleration on large shards
export HF_HUB_ENABLE_HF_TRANSFER=0      # ← TURNED OFF (fixes your previous issue)

export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

########################################
# NCCL / multiprocessing
########################################
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

########################################
# Install Python deps into a per-job venv
# (fixes: srun tasks not finding pip-installed packages)
########################################
echo "Setting up Python venv..."
python3 -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install -q --upgrade pip
pip install -q \
    transformers datasets accelerate evaluate trl peft hf_transfer \
    rouge-score sacrebleu sentence-transformers sympy --no-build-isolation

########################################
# Launch distributed training
########################################
echo "Starting multi-GPU distillation job..."

srun --ntasks=$SLURM_NTASKS --nodes=1 \
    bash -c "source $SLURM_TMPDIR/venv/bin/activate && \
             accelerate launch --num_processes $SLURM_NTASKS --mixed_precision bf16 multi_distill.py"
