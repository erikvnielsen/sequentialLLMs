#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

####################################
# CONFIG
####################################
teacher_name = "arcee-ai/Arcee-Spark"
student_name = "Qwen/Qwen2-0.5B"

####################################
# LOAD MODELS
####################################
print("Loading teacher model...")
tok_t = AutoTokenizer.from_pretrained(teacher_name)
tok_t.pad_token = tok_t.eos_token
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher.eval()

print("Loading student model...")
tok_s = AutoTokenizer.from_pretrained(student_name)
tok_s.pad_token = tok_s.eos_token
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)
student.eval()

####################################
# HELPER: Only decode the completion
####################################
def generate_completion(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Slice off the prompt tokens → return ONLY generated continuation
    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

####################################
# 1-BATCH DISTILLATION TEST
####################################
print("\nRunning 1-batch distillation test...")

sample_text = "The capital of France is"
batch = tok_s(sample_text, return_tensors="pt").to(student.device)

with torch.no_grad():
    t_out = teacher(**batch).logits

s_out = student(**batch).logits

# Align vocab size
min_vocab = min(s_out.size(-1), t_out.size(-1))
s_logits = s_out[..., :min_vocab]
t_logits = t_out[..., :min_vocab]

kl_loss = torch.nn.functional.kl_div(
    torch.nn.functional.log_softmax(s_logits, dim=-1),
    torch.nn.functional.softmax(t_logits, dim=-1),
    reduction="batchmean"
)

print("KL loss (debug-safe):", float(kl_loss.detach()))

####################################
# FEW-SHOT REASONING TEST
####################################
print("\nRunning few-shot reasoning test...")

# GSM8K
gsm = load_dataset("openai/gsm8k", "main", split="test[:1]")

# ARC (correct field: "question")
arc = load_dataset("ai2_arc", "ARC-Challenge", split="test[:1]")

few_shot_examples = [
    "Q: What is 2 + 2?\nA: 4",
    "Q: What is 3 * 3?\nA: 9",
]

questions = [
    gsm[0]["question"],      # GSM field: "question"
    arc[0]["question"],      # ARC CHALLENGE field is "question"
]

for q in questions:
    prompt = "\n".join(few_shot_examples) + f"\nQ: {q}\nA:"
    ans = generate_completion(student, tok_s, prompt)

    print("\nQUESTION:", q)
    print("MODEL OUTPUT:", ans)

print("\n=== Debug Complete ===")
