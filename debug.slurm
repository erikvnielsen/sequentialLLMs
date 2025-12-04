#!/bin/bash
#SBATCH --job-name=distill_debug
#SBATCH --partition=debug             # <-- shorter queue
#SBATCH --gres=gpu:1                  # only 1 GPU needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=debug_%j.out

############################################
# ENVIRONMENT
############################################
module purge
module load python/3.10.4
module load cuda/12.2

export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

############################################
# INSTALL MINIMAL DEPENDENCIES
############################################
pip install -q --upgrade pip
pip install -q transformers datasets accelerate evaluate trl sentence-transformers sympy --no-build-isolation

############################################
# DEBUG SCRIPT
############################################
echo "=== Debug Distillation Test ==="

cat << 'EOF' > debug_distill.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ----------------
# Load teacher + student
# ----------------
teacher_name = "arcee-ai/Arcee-Spark"
student_name = "Qwen/Qwen2-0.5B"

print("Loading teacher...")
tok_t = AutoTokenizer.from_pretrained(teacher_name)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Loading student...")
tok_s = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)

# ----------------
# Tiny 1-batch distillation test
# ----------------
print("Running 1-batch distillation test...")

sample = "The capital of France is"
batch = tok_s(sample, return_tensors="pt").to(student.device)

with torch.no_grad():
    t_out = teacher(**batch).logits

s_out = student(**batch).logits

loss = torch.nn.functional.kl_div(
    torch.nn.functional.log_softmax(s_out, dim=-1),
    torch.nn.functional.softmax(t_out, dim=-1),
    reduction="batchmean"
)

print("Distillation KL loss:", float(loss))

# ----------------
# Tiny reasoning eval (1 GSM8K + 1 MATH)
# ----------------
print("\nRunning tiny reasoning eval...")

gsm = load_dataset("openai/gsm8k", "main", split="test[:1]")
math = load_dataset("hendrycks/competition_math", split="test[:1]")

qs = [
    gsm[0]["question"],
    math[0]["problem"]
]

for q in qs:
    enc = tok_s(q, return_tensors="pt").to(student.device)
    out = student.generate(**enc, max_new_tokens=64)
    text = tok_s.decode(out[0], skip_special_tokens=True)
    print("\nQUESTION:", q)
    print("MODEL OUTPUT:", text)

print("\n=== Debug Complete ===")
EOF

############################################
# RUN
############################################
srun python debug_distill.py
