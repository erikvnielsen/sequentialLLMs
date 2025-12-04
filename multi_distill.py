#!/usr/bin/env python3
"""
multi_distill.py

Multi-stage distillation + reasoning evaluation pipeline.

- CONTROL: original teacher -> each student
- CASCADE: teacher -> s1 -> s2 -> s3 -> s4
- Evaluation on GSM8K and ARC-Challenge in 0-shot, 1-shot, 3-shot modes
- Saves per-model CSVs and summary + plots

This version:
- Builds few-shot examples FROM the datasets (train split)
- Uses a generation helper that returns ONLY the continuation and truncates at stop sequences
- Attempts to map ARC answers robustly (letter or matched choice text)
- Pads/truncates logits for KL distillation safely
- Contains additional try/except guards so debug runs are less likely to crash
"""

import os
import json
import math
import re
import gc
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainingArguments
from trl import SFTTrainer
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
        "bf16": True,
        "dataloader_num_workers": 2,
        "seed": 42,
    },
    "distillation": {"temperature": 2.0, "alpha": 0.5},
    "evaluation": {
        "gsm8k_samples": 50,
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

        loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, getattr(student_outputs, "loss", None))
        return (loss, student_outputs) if return_outputs else loss

    def distillation_loss(self, student_logits, teacher_logits, original_loss):
        T = CONFIG["distillation"]["temperature"]
        alpha = CONFIG["distillation"]["alpha"]

        # pad vocab sizes if needed (safe)
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
# Prompt builders & extractors
# -------------------------
def zero_shot_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def build_gsm8k_examples(n: int) -> str:
    if n <= 0:
        return ""
    try:
        train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    except Exception:
        # fallback small subset if dataset fails
        train = load_dataset("gsm8k", "main", split="train[:100]").shuffle(seed=42)
    demos = []
    for idx, ex in enumerate(train.select(range(n))):
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        demos.append(f"Question: {q}\nAnswer: {a}\n")
    return "\n".join(demos)

def build_arc_examples(n: int) -> str:
    if n <= 0:
        return ""
    try:
        train = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed=42)
    except Exception:
        train = load_dataset("ai2_arc", "ARC-Challenge", split="train[:100]").shuffle(seed=42)
    demos = []
    for ex in train.select(range(n)):
        q = ex.get("question", "").strip()
        choices = ex.get("choices", {}).get("text") if isinstance(ex.get("choices"), dict) else ex.get("choices", [])
        key = ex.get("answerKey", "")
        opts = [f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]
        opts_joined = "\n".join(opts)
        demos.append(f"Question: {q}\n{opts_joined}\nAnswer: {key}\n")
    return "\n".join(demos)

def build_gsm_prompt(q: str, n_shot: int) -> str:
    if n_shot == 0:
        return zero_shot_prompt(q)
    return build_gsm8k_examples(n_shot) + f"\nQuestion: {q}\nAnswer:"

def build_arc_prompt(q: str, choices: List[str], n_shot: int) -> str:
    ch_fmt = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])
    if n_shot == 0:
        return f"Question: {q}\n{ch_fmt}\nAnswer:"
    return build_arc_examples(n_shot) + f"\nQuestion: {q}\n{ch_fmt}\nAnswer:"

def extract_last_integer(text: Optional[str]):
    if text is None:
        return None
    tokens = re.findall(r"-?\d+", (text or "").replace(",", ""))
    if not tokens:
        return None
    try:
        return int(tokens[-1])
    except:
        return None

def extract_choice_letter(text: Optional[str]):
    if text is None:
        return None
    t = (text or "").upper()
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"(choice|answer)[:\s]*([A-Da-d])", text or "", re.IGNORECASE)
    if m2:
        return m2.group(2).upper()
    return None

def match_choice_by_text(gen: str, choices: List[str]) -> Optional[str]:
    if not gen or not choices:
        return None
    g = gen.lower()
    # exact substring match first
    for i, c in enumerate(choices):
        if c and c.lower() in g:
            return chr(65 + i)
    # fallback: token overlap score
    def score(a, b):
        aset = set(re.findall(r"\w+", a.lower()))
        bset = set(re.findall(r"\w+", b.lower()))
        if not aset or not bset:
            return 0
        return len(aset & bset) / max(1, len(aset | bset))
    best, best_i = 0.0, None
    for i, c in enumerate(choices):
        s = score(g, c)
        if s > best:
            best, best_i = s, i
    if best_i is not None and best > 0.1:
        return chr(65 + best_i)
    return None

# -------------------------
# Stop-sequence cleaning + generation helper
# -------------------------
STOP_SEQS = ["\nQ:", "\nQuestion:", "\nA:"]

def truncate_at_stop(text: str) -> str:
    if not text:
        return ""
    idxs = [text.find(s) for s in STOP_SEQS if text.find(s) != -1]
    if not idxs:
        return text.strip()
    cut = min(idxs)
    return text[:cut].strip()

def generate_completion(model, tokenizer, prompt, max_new_tokens=64, do_sample=False, stop_cleanup:bool=True):
    # ensure pad/eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONFIG["tokenizer"]["max_length"]).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # early_stopping True helps but not enough; we clean post-hoc
            early_stopping=True,
        )
    # only the continuation tokens
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_cleanup:
        gen_text = truncate_at_stop(gen_text)
    return gen_text

# -------------------------
# Distill single student
# -------------------------
def distill_one(teacher_id: str, student_id: str, out_dir: str, tokenized_dataset):
    safe_mkdir(out_dir)
    bf16 = CONFIG["training"]["bf16"]
    kwargs = {"device_map": "auto", "trust_remote_code": True, "low_cpu_mem_usage": True}
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    print(f"Loading teacher for distillation: {teacher_id}")
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, **kwargs)
    teacher.eval()
    print(f"Loading student for distillation: {student_id}")
    student = AutoModelForCausalLM.from_pretrained(student_id, **kwargs)
    try:
        student.gradient_checkpointing_enable()
    except Exception:
        pass

    student_tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    targs = dict(CONFIG["training"])
    targs["output_dir"] = out_dir
    targs.pop("base_output_dir", None)
    training_args = TrainingArguments(**targs)

    trainer = KDTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=student_tokenizer
    )
    trainer.teacher_model = teacher

    try:
        trainer.train(resume_from_checkpoint=CONFIG["training"]["resume_from_checkpoint"])
    except Exception as e:
        print("Training error:", e)
    finally:
        try:
            trainer.save_model(out_dir)
        except Exception:
            pass
        try:
            student_tokenizer.save_pretrained(out_dir)
        except Exception:
            pass
        try:
            logs = trainer.state.log_history
        except Exception:
            logs = []
        with open(os.path.join(out_dir, "train_logs.json"), "w") as f:
            json.dump(logs, f, indent=2)
        try:
            del trainer, student, teacher
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    return out_dir

# -------------------------
# Prepare tokenized dataset (simple LM proxy)
# -------------------------
def prepare_tokenized_dataset_for_student_raw_dataset(raw_ds, tokenizer, max_length):
    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = raw_ds.map(tok_fn, batched=True, remove_columns=raw_ds.column_names, num_proc=1)
    split = tokenized.train_test_split(test_size=0.1, seed=CONFIG["training"]["seed"])
    return split

# -------------------------
# Evaluation routine
# -------------------------
def evaluate_reasoning_models(models_to_eval: List[Dict], out_dir="reasoning_eval"):
    safe_mkdir(out_dir)
    rows = []

    # load small samples (test/validation)
    try:
        gsm8k_full = load_dataset("gsm8k", "main", split="test")
    except Exception:
        gsm8k_full = load_dataset("gsm8k", "main", split="test[:200]")

    try:
        arc_full = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    except Exception:
        arc_full = load_dataset("ai2_arc", "ARC-Challenge", split="validation[:200]")

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

        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            tok = load_tokenizer(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, **({"torch_dtype": torch.bfloat16} if CONFIG["training"]["bf16"] else {}))
        model.eval()

        for shot_label, n_shot in shots.items():
            print(f"Evaluating {label} - {shot_label} (n_shot={n_shot})")

            # Build few-shot pools once per model/shot
            try:
                gsm_shot_pool = load_dataset("gsm8k", "main", split="train").shuffle(seed=42).select(range(max(50, n_shot)))
            except Exception:
                gsm_shot_pool = load_dataset("gsm8k", "main", split="train[:200]").shuffle(seed=42).select(range(max(50, n_shot)))
            try:
                arc_shot_pool = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed=42).select(range(max(50, n_shot)))
            except Exception:
                arc_shot_pool = load_dataset("ai2_arc", "ARC-Challenge", split="train[:200]").shuffle(seed=42).select(range(max(50, n_shot)))

            # Convert shot pools to examples
            gsm_examples = [{"q": ex["question"], "a": ex["answer"].strip()} for ex in gsm_shot_pool.select(range(n_shot))] if n_shot > 0 else []
            arc_examples = []
            for ex in arc_shot_pool.select(range(n_shot)):
                choices = ex.get("choices", {}).get("text") if isinstance(ex.get("choices"), dict) else ex.get("choices", [])
                key = ex.get("answerKey", "")
                idx = None
                if isinstance(choices, list) and key:
                    labels = ex.get("choices", {}).get("label") if isinstance(ex.get("choices"), dict) else None
                    if labels and key in labels:
                        idx = labels.index(key)
                    else:
                        idx = 0
                choice_text = choices[idx] if (isinstance(choices, list) and idx is not None and idx < len(choices)) else (choices[0] if isinstance(choices, list) and len(choices) > 0 else "")
                # store 'a' as choice text if available else letter
                arc_examples.append({"q": ex.get("question", ""), "a": (choice_text if choice_text else (ex.get("answerKey", "") or "").strip())})

            # GSM8K evaluation
            gsm_results = []
            correct = 0
            total = 0
            for item in gsm:
                q = item["question"]
                gold_raw = item.get("answer", "")
                gold_num = extract_last_integer(gold_raw) if isinstance(gold_raw, str) else None

                if n_shot == 0:
                    prompt = build_gsm_prompt(q, 0)
                else:
                    if len(gsm_examples) < n_shot:
                        prompt = build_gsm_prompt(q, len(gsm_examples))
                    else:
                        demos = "\n".join([f"Question: {e['q']}\nAnswer: {e['a']}\n" for e in gsm_examples])
                        prompt = demos + f"\nQuestion: {q}\nAnswer:"

                gen = generate_completion(model, tok, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
                pred_num = extract_last_integer(gen)
                is_correct = (pred_num is not None and gold_num is not None and pred_num == gold_num)
                gsm_results.append({"question": q, "gold": gold_num, "pred": pred_num, "correct": is_correct, "generation": gen})
                correct += int(is_correct)
                total += 1

            acc = correct / total if total > 0 else 0.0
            rows.append({"model": label, "dataset": "GSM8K", "shot": shot_label, "accuracy": acc, "num": total})
            pd.DataFrame(gsm_results).to_csv(os.path.join(out_dir, f"{label}_GSM8K_{shot_label}.csv"), index=False)

            # ARC evaluation
            arc_results = []
            correct = 0
            total = 0
            for item in arc:
                q = item["question"]
                choices = item.get("choices", {}).get("text") if isinstance(item.get("choices"), dict) else item.get("choices", [])
                gold_key = item.get("answerKey", None)

                if n_shot == 0:
                    prompt = build_arc_prompt(q, choices, 0)
                else:
                    if len(arc_examples) < n_shot:
                        prompt = build_arc_prompt(q, choices, len(arc_examples))
                    else:
                        # arc_examples 'a' is stored as choice text or letter earlier
                        demos = "\n".join([f"Question: {e['q']}\n{e['a']}\n" for e in arc_examples])
                        prompt = demos + f"\nQuestion: {q}\n" + "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]) + "\nAnswer:"

                gen = generate_completion(model, tok, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
                # try letter first
                pred_choice = extract_choice_letter(gen)
                # if no letter, try to match by text
                if pred_choice is None and isinstance(choices, list) and choices:
                    pred_choice = match_choice_by_text(gen, choices)
                is_correct = (pred_choice is not None and gold_key is not None and pred_choice == gold_key)
                arc_results.append({"question": q, "choices": choices, "gold": gold_key, "pred": pred_choice, "correct": is_correct, "generation": gen})
                correct += int(is_correct)
                total += 1

            acc = correct / total if total > 0 else 0.0
            rows.append({"model": label, "dataset": "ARC-Challenge", "shot": shot_label, "accuracy": acc, "num": total})
            pd.DataFrame(arc_results).to_csv(os.path.join(out_dir, f"{label}_ARC_{shot_label}.csv"), index=False)

        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "reasoning_summary.csv"), index=False)

    # plotting
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        plt.figure(figsize=(10, 6))
        models = sub["model"].unique()
        for shot_label in sorted(sub["shot"].unique()):
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
# Top-level experiments
# -------------------------
def run_all():
    base_out = CONFIG["base_output_dir"]
    safe_mkdir(base_out)

    teacher_orig = CONFIG["teacher"]
    students = CONFIG["students"]

    results_control = {}
    results_cascade = {}

    print("Loading small LM training proxy dataset (wikitext subset)...")
    raw_text_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    # CONTROL
    control_dir = os.path.join(base_out, "control")
    safe_mkdir(control_dir)
    print("\n=== RUNNING CONTROL (teacher -> each student) ===")
    for i, student in enumerate(students, start=1):
        label = student.replace("/", "_")
        out_dir = os.path.join(control_dir, f"stage_{i}_{label}")
        safe_mkdir(out_dir)

        student_tok = load_tokenizer(student)
        tokenized = prepare_tokenized_dataset_for_student_raw_dataset(raw_text_ds, student_tok, CONFIG["tokenizer"]["max_length"])

        saved_path = distill_one(teacher_orig, student, out_dir, tokenized)

        df = evaluate_reasoning_models([{"name": saved_path, "label": f"{label}_control"}], out_dir=os.path.join(out_dir, "reasoning_eval"))
        results_control[student] = {"eval_summary": df.to_dict(orient="records")}

    # CASCADE
    cascade_dir = os.path.join(base_out, "cascade")
    safe_mkdir(cascade_dir)
    print("\n=== RUNNING CASCADE (teacher -> chain students) ===")
    current_teacher = teacher_orig
    for i, student in enumerate(students, start=1):
        label = student.replace("/", "_")
        out_dir = os.path.join(cascade_dir, f"stage_{i}_{label}")
        safe_mkdir(out_dir)

        student_tok = load_tokenizer(student)
        tokenized = prepare_tokenized_dataset_for_student_raw_dataset(raw_text_ds, student_tok, CONFIG["tokenizer"]["max_length"])

        saved_path = distill_one(current_teacher, student, out_dir, tokenized)

        df = evaluate_reasoning_models([{"name": saved_path, "label": f"{label}_cascade"}], out_dir=os.path.join(out_dir, "reasoning_eval"))
        results_cascade[student] = {"eval_summary": df.to_dict(orient="records")}

        current_teacher = saved_path

    with open(os.path.join(base_out, "results_control.json"), "w") as f:
        json.dump(results_control, f, indent=2)
    with open(os.path.join(base_out, "results_cascade.json"), "w") as f:
        json.dump(results_cascade, f, indent=2)

    print("\nALL DONE. Results saved to:", base_out)

if __name__ == "__main__":
    run_all()
