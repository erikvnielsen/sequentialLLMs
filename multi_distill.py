#!/usr/bin/env python3
"""
multi_distill.py

Multi-stage distillation + reasoning evaluation pipeline.

Features:
- CONTROL: original teacher -> each student
- CASCADE: teacher -> s1 -> s2 -> s3 -> s4
- Evaluation on GSM8K and ARC-Challenge in 0-shot, 1-shot, 3-shot modes
- Auto-grading and CSV + plot outputs
- Cluster-safe (matplotlib Agg), uses device_map="auto", bf16 option
"""

import os
import json
import math
import re
import gc
import time
from typing import List, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
import evaluate
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
set_seed(42)

CONFIG = {
    "project_name": "distil_reasoning_eval",
    "teacher": "arcee-ai/Arcee-Spark",
    "students": [
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2-0.5B",
        "TinyLlama/TinyLlama-1.1B-Chat",
        "EleutherAI/pythia-160m",
    ],
    "base_output_dir": "./distill_results",
    "tokenizer": {"max_length": 1024},
    "training": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 50,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,  # set True on A100 with correct drivers
        "dataloader_num_workers": 2,
        "seed": 42,
    },
    "distillation": {"temperature": 2.0, "alpha": 0.5},
    "evaluation": {
        "gsm8k_samples": 50,       # reduce for speed; change to larger numbers for final runs
        "arc_samples": 50,
        "shots": {"0-shot": 0, "1-shot": 1, "3-shot": 3},
        "max_new_tokens": 64,
        "do_sample": False,
    },
    "perf": {"allow_tf32": True},
}

os.makedirs(CONFIG["base_output_dir"], exist_ok=True)

if CONFIG["perf"].get("allow_tf32", False):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------------------------
# Utilities
# -------------------------
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model(model_id, bf16=True):
    kwargs = {"device_map": "auto", "trust_remote_code": True, "low_cpu_mem_usage": True}
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model

# -------------------------
# KD Trainer (SFTTrainer subclass)
# -------------------------
class KDTrainer(SFTTrainer):
    """
    SFTTrainer subclass that mixes KL distillation with original CE loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # ensure teacher is on same device
        self.teacher_model = self.teacher_model.to(device)

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, "module") else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, student_outputs.loss)
        return (loss, student_outputs) if return_outputs else loss

    def distillation_loss(self, student_logits, teacher_logits, original_loss):
        T = CONFIG["distillation"]["temperature"]
        alpha = CONFIG["distillation"]["alpha"]

        # pad vocab sizes if needed
        s_size, t_size = student_logits.size(-1), teacher_logits.size(-1)
        if s_size != t_size:
            pad_size = abs(s_size - t_size)
            pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
            if s_size < t_size:
                student_logits = torch.cat([student_logits, pad_tensor], dim=-1)
            else:
                teacher_logits = torch.cat([teacher_logits, pad_tensor], dim=-1)

        s_scaled = student_logits / T
        t_scaled = teacher_logits / T

        loss_kd = F.kl_div(
            F.log_softmax(s_scaled, dim=-1),
            F.softmax(t_scaled, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)

        if original_loss is None:
            return alpha * loss_kd
        return alpha * loss_kd + (1.0 - alpha) * original_loss

# -------------------------
# Prompts and extractors for reasoning datasets
# -------------------------
def zero_shot_prompt(question: str) -> str:
    return f"Answer the question below.\n\nQuestion: {question}\n\nAnswer:"

def one_shot_prompt(question: str) -> str:
    # simple 1-shot demo (not CoT)
    demo_q = "If Sarah has 5 apples and buys 3 more, how many apples does she have?"
    demo_a = "8"
    return f"Example:\nQ: {demo_q}\nA: {demo_a}\n\nQuestion: {question}\n\nAnswer:"

def few_shot_prompt(question: str) -> str:
    exs = [
        ("Tom has 7 marbles and gives away 2. How many remain?", "5"),
        ("A toy costs 6 dollars and you buy 3. Total cost?", "18"),
        ("Jenny has 12 candies, eats 4, then buys 2. How many?", "10"),
    ]
    demos = "\n".join([f"Q: {q}\nA: {a}\n" for q, a in exs])
    return f"{demos}\nQuestion: {question}\n\nAnswer:"

# extract last integer for GSM8K (robust)
def extract_last_integer(text: str):
    if text is None:
        return None
    # remove commas in numbers, find integers
    tokens = re.findall(r"-?\d+", text.replace(",", ""))
    if not tokens:
        return None
    return int(tokens[-1])

# extract choice letter A/B/C/D
def extract_choice_letter(text: str):
    if text is None:
        return None
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    # try to match "choice: A" style
    m2 = re.search(r"(choice|answer)[:\s]*([A-Da-d])", text)
    if m2:
        return m2.group(2).upper()
    # otherwise try to map textual yes/no to a choice? (not applicable)
    return None

# -------------------------
# Distillation (train single student)
# -------------------------
def distill_one(teacher_id: str, student_id: str, out_dir: str, tokenized_dataset):
    """
    teacher_id: model id or path (string)
    student_id: student model id (string)
    out_dir: where to save student
    tokenized_dataset: dict with 'train' and 'test' datasets tokenized for student tokenizer
    """
    print(f"\n-- Distilling: teacher={teacher_id} -> student={student_id} -> out={out_dir}")
    safe_mkdir(out_dir)

    # model kwargs
    bf16 = CONFIG["training"]["bf16"]
    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # load teacher (could be a local path)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, device_map="auto", trust_remote_code=True, **({"torch_dtype": torch.bfloat16} if bf16 else {}))
    teacher.eval()

    # load student model
    student = AutoModelForCausalLM.from_pretrained(student_id, device_map="auto", trust_remote_code=True, **({"torch_dtype": torch.bfloat16} if bf16 else {}))
    try:
        student.gradient_checkpointing_enable()
    except Exception:
        pass

    # tokenizer for student
    student_tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # TrainingArguments
    targs = dict(CONFIG["training"])
    targs["output_dir"] = out_dir
    targs.pop("base_output_dir", None)
    training_args = TrainingArguments(**targs)

    trainer = KDTrainer(
        teacher_model=teacher,
        temperature=CONFIG["distillation"]["temperature"],
        alpha=CONFIG["distillation"]["alpha"],
        model=student,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=student_tokenizer,
    )

    trainer.teacher_model = teacher

    try:
        trainer.train(resume_from_checkpoint=CONFIG["training"]["resume_from_checkpoint"])
    except Exception as e:
        print("Training error (stage may be partially saved):", e)
    finally:
        try:
            trainer.save_model(out_dir)
        except Exception as e:
            print("Warning: save_model failed:", e)
        try:
            student_tokenizer.save_pretrained(out_dir)
        except Exception as e:
            print("Warning: saving tokenizer failed:", e)
        try:
            logs = trainer.state.log_history
        except Exception:
            logs = []
        with open(os.path.join(out_dir, "train_logs.json"), "w") as f:
            json.dump(logs, f, indent=2)
        # cleanup
        try:
            del trainer, student, teacher
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    return out_dir

# -------------------------
# Prepare dataset per student tokenizer (tokenize the message/chat prompt format)
# -------------------------
def prepare_tokenized_dataset_for_student_raw_dataset(raw_ds, tokenizer, max_length):
    """
    raw_ds: HF dataset with field 'text' (we use mlabonne/FineTome-100k originally) but for distillation training
    We will render the raw text into simple 'text' field tokenization for LM training.
    """
    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = raw_ds.map(tok_fn, batched=True, remove_columns=raw_ds.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=CONFIG["training"]["seed"])
    return split

# -------------------------
# Reasoning evaluation routine
# -------------------------
def evaluate_reasoning_models(models_to_eval: List[Dict], out_dir="reasoning_eval"):
    """
    models_to_eval: list of dicts: [{"name":"model_path_or_id","label":"S1_control"}, ...]
    Saves per-model CSVs and a reasoning_summary.csv, returns the DataFrame.
    """
    safe_mkdir(out_dir)
    rows = []

    # Preload dataset splits
    gsm8k_full = load_dataset("gsm8k", "main", split="test")
    arc_full = load_dataset("ai2_arc", "ARC-Challenge", split="validation")

    # sample sizes
    gN = min(CONFIG["evaluation"]["gsm8k_samples"], len(gsm8k_full))
    aN = min(CONFIG["evaluation"]["arc_samples"], len(arc_full))

    gsm = gsm8k_full.select(range(gN))
    arc = arc_full.select(range(aN))

    shots = CONFIG["evaluation"]["shots"]
    max_new_tokens = CONFIG["evaluation"]["max_new_tokens"]
    do_sample = CONFIG["evaluation"]["do_sample"]

    for model_spec in models_to_eval:
        model_id = model_spec["name"]
        label = model_spec.get("label", model_id.replace("/", "_"))
        print(f"\n=== Evaluating reasoning for model {label} ({model_id}) ===")

        # Load tokenizer and model (device_map auto)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, **({"torch_dtype": torch.bfloat16} if CONFIG["training"]["bf16"] else {}))
        model.eval()

        for shot_label, n_shot in shots.items():
            # build shot examples pools
            # For GSM8K use examples from train split
            gsm_shot_pool = load_dataset("gsm8k", "main", split="train").shuffle(seed=42).select(range(50))
            arc_shot_pool = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed=42).select(range(50))

            # pick first n_shot examples
            gsm_examples = [{"q": ex["question"], "a": ex["answer"].strip()} for ex in gsm_shot_pool[:n_shot]]
            arc_examples = []
            for ex in arc_shot_pool[:n_shot]:
                # arc train has 'question', 'choices' possibly
                choices = ex.get("choices", {}).get("text") or ex.get("choices", [])
                # Pick answer text (train may not have 'answerKey' for train, but we only need example formatting)
                answer_key = ex.get("answerKey", "") or ""
                # Build simplistic answer text: "A" style if available else blank
                arc_examples.append({"q": ex["question"], "a": answer_key})

            # For each mode (0/1/3-shot) we will evaluate
            # GSM8K eval
            correct = 0
            total = 0
            per_example = []
            for item in gsm:
                q = item["question"]
                # gold answer extraction: dataset 'answer' often includes final number; try extract last int
                gold_raw = item.get("answer", "")
                # extract integer from gold
                gold_num = extract_last_integer(gold_raw) if isinstance(gold_raw, str) else None

                # build prompt
                if n_shot == 0:
                    prompt = zero_shot_prompt(q)
                elif n_shot == 1:
                    # use first gsm_examples
                    ex = gsm_examples[0]
                    prompt = f"Q: {ex['q']}\nA: {ex['a']}\n\nQuestion: {q}\n\nAnswer:"
                else:
                    # few-shot
                    demos = "\n".join([f"Q: {e['q']}\nA: {e['a']}\n" for e in gsm_examples])
                    prompt = f"{demos}\nQuestion: {q}\n\nAnswer:"

                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=CONFIG["tokenizer"]["max_length"]).to(next(model.parameters()).device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred_num = extract_last_integer(gen)
                is_correct = (pred_num is not None and gold_num is not None and pred_num == gold_num)
                per_example.append({"question": q, "gold": gold_num, "pred": pred_num, "correct": is_correct, "generation": gen})
                correct += int(is_correct)
                total += 1

            acc = correct/total if total>0 else 0.0
            rows.append({"model": label, "dataset": "GSM8K", "shot": shot_label, "accuracy": acc, "num": total})
            # save CSV of per-example
            pd.DataFrame(per_example).to_csv(os.path.join(out_dir, f"{label}_GSM8K_{shot_label}.csv"))

            # ARC evaluation (multiple choice)
            correct = 0
            total = 0
            per_example = []
            for item in arc:
                q = item["question"]
                choices = item["choices"]["text"] if isinstance(item["choices"], dict) else (item["choices"] if isinstance(item["choices"], list) else [])
                gold_key = item.get("answerKey", None)

                if n_shot == 0:
                    base = "Choose the correct answer (A/B/C/D)."
                    prompt = f"{base}\n\nQuestion: {q}\n\nChoices:\n" + "\n".join([f"{c}" for c in choices]) + "\n\nAnswer:"
                elif n_shot == 1:
                    ex = arc_examples[0] if arc_examples else {"q":"", "a": ""}
                    demo = f"Q: {ex['q']}\nA: {ex['a']}\n\n"
                    prompt = f"{demo}Question: {q}\n\nChoices:\n" + "\n".join([f"{c}" for c in choices]) + "\n\nAnswer:"
                else:
                    demos = "\n".join([f"Q: {e['q']}\nA: {e['a']}\n" for e in arc_examples])
                    prompt = f"{demos}\nQuestion: {q}\n\nChoices:\n" + "\n".join([f"{c}" for c in choices]) + "\n\nAnswer:"

                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=CONFIG["tokenizer"]["max_length"]).to(next(model.parameters()).device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred_choice = extract_choice_letter(gen)
                # if dataset gold is letter, compare directly; else skip
                is_correct = (pred_choice is not None and gold_key is not None and pred_choice == gold_key)
                per_example.append({"question": q, "choices": choices, "gold": gold_key, "pred": pred_choice, "correct": is_correct, "generation": gen})
                correct += int(is_correct)
                total += 1

            acc = correct/total if total>0 else 0.0
            rows.append({"model": label, "dataset": "ARC-Challenge", "shot": shot_label, "accuracy": acc, "num": total})
            pd.DataFrame(per_example).to_csv(os.path.join(out_dir, f"{label}_ARC_{shot_label}.csv"))

        # cleanup model
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    # summary CSV + plots
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "reasoning_summary.csv"), index=False)

    # plotting
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        plt.figure(figsize=(10, 6))
        # x axis: unique model labels
        models = sub["model"].unique()
        x = range(len(models))
        # one line per shot
        for shot_label in sub["shot"].unique():
            y = []
            for m in models:
                r = sub[(sub["model"] == m) & (sub["shot"] == shot_label)]
                y.append(r["accuracy"].values[0] if len(r) > 0 else float("nan"))
            plt.plot(models, y, marker="o", label=shot_label)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Reasoning accuracy on {ds}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"accuracy_{ds.replace(' ', '_')}.png"))
        plt.close()

    return df

# -------------------------
# Top-level experiments: CONTROL + CASCADE
# -------------------------
def run_all():
    base_out = CONFIG["base_output_dir"]
    safe_mkdir(base_out)

    teacher_orig = CONFIG["teacher"]
    students = CONFIG["students"]

    results_control = {}
    results_cascade = {}

    # ---- Load raw distillation dataset (we'll use wikitext as LM training proxy) ----
    # NOTE: you can change this to your original FineTome dataset if desired
    print("Loading distillation raw dataset (wikitext-2 small subset for speed)...")
    raw_text_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # small for speed by default

    # CONTROL
    control_dir = os.path.join(base_out, "control")
    safe_mkdir(control_dir)
    print("\n=== RUNNING CONTROL (original teacher -> each student) ===")
    for i, student in enumerate(students, start=1):
        stage_name = f"control_stage_{i}_{student.replace('/', '_')}"
        out_dir = os.path.join(control_dir, stage_name)
        safe_mkdir(out_dir)

        # tokenize for student
        student_tok = load_tokenizer(student)
        tokenized = prepare_tokenized_dataset_for_student_raw_dataset(raw_text_ds, student_tok, CONFIG["tokenizer"]["max_length"])

        saved_path = distill_one(teacher_orig, student, out_dir, tokenized)

        # Evaluate reasoning for this student (control)
        df = evaluate_reasoning_models([{"name": saved_path, "label": f"{student}_control"}], out_dir=os.path.join(out_dir, "reasoning_eval"))
        results_control[student] = {"eval_summary": df.to_dict(orient="records")}

    # CASCADE
    cascade_dir = os.path.join(base_out, "cascade")
    safe_mkdir(cascade_dir)
    print("\n=== RUNNING CASCADE (teacher -> s1 -> s2 -> s3 -> s4) ===")
    current_teacher = teacher_orig
    for i, student in enumerate(students, start=1):
        stage_name = f"cascade_stage_{i}_{student.replace('/', '_')}"
        out_dir = os.path.join(cascade_dir, stage_name)
        safe_mkdir(out_dir)

        student_tok = load_tokenizer(student)
        tokenized = prepare_tokenized_dataset_for_student_raw_dataset(raw_text_ds, student_tok, CONFIG["tokenizer"]["max_length"])

        saved_path = distill_one(current_teacher, student, out_dir, tokenized)

        df = evaluate_reasoning_models([{"name": saved_path, "label": f"{student}_cascade"}], out_dir=os.path.join(out_dir, "reasoning_eval"))
        results_cascade[student] = {"eval_summary": df.to_dict(orient="records")}

        # next teacher becomes this saved_path
        current_teacher = saved_path

    # Save aggregated results
    with open(os.path.join(base_out, "results_control.json"), "w") as f:
        json.dump(results_control, f, indent=2)
    with open(os.path.join(base_out, "results_cascade.json"), "w") as f:
        json.dump(results_cascade, f, indent=2)

    print("\nALL DONE. Results saved to:", base_out)

if __name__ == "__main__":
    run_all()