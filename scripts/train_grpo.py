"""GRPO on the merged SFT model with a verifiable reward.

Unlike DPO (which learns from a *preference* between two given answers), GRPO lets
the model generate its own answers and scores each one against a programmable reward.
PubMedQA gives us a clean verifiable signal: the reference answer starts with a
yes/no/maybe verdict, so we can reward the model's own verdict for matching it.

    reward = 1.0  if generated verdict == reference verdict     (the real signal)
           + 0.2  if the answer even starts with a verdict word (light shaping)

Policy starts as sft-merged (loaded 4-bit, QLoRA); a fresh LoRA adapter is trained.
The KL reference is that same SFT model with the adapter disabled.

Run:  python scripts/train_grpo.py         (full run)
      SMOKE=1 python scripts/train_grpo.py (tiny sanity run)
"""
import os
import re
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

RL = Path(__file__).resolve().parent.parent
SFT_MERGED = RL / "models" / "sft-merged"
TRAIN = RL.parent / "medical-llm" / "data" / "processed" / "medical_train.json"
OUT = RL / "outputs" / "grpo"
SMOKE = os.environ.get("SMOKE") == "1"

MAX_INPUT_CHARS = 2400
PER_CLASS = 6 if SMOKE else 80  # balanced subset so the model can't just spam "yes"
VERDICTS = ("yes", "no", "maybe")


def parse_verdict(text: str) -> str:
    m = re.match(r"\s*(yes|no|maybe)\b", text.strip().lower())
    return m.group(1) if m else "none"


def user_content(ex: dict) -> str:
    inp = (ex.get("input") or "").strip()[:MAX_INPUT_CHARS]
    instr = ex["instruction"].strip()
    return f"{instr}\n\n{inp}" if inp else instr


def build_dataset() -> Dataset:
    data = json.load(open(TRAIN))
    buckets = {v: [] for v in VERDICTS}
    for ex in data:
        v = parse_verdict(ex["output"])
        if v in buckets:
            buckets[v].append(ex)
    rng = random.Random(0)
    rows = []
    for v in VERDICTS:
        rng.shuffle(buckets[v])
        for ex in buckets[v][:PER_CLASS]:
            rows.append({
                "prompt": [{"role": "user", "content": user_content(ex)}],
                "reference_verdict": v,
            })
    rng.shuffle(rows)
    print(f"GRPO train prompts: {len(rows)} "
          f"({', '.join(f'{v}={min(len(buckets[v]), PER_CLASS)}' for v in VERDICTS)})")
    return Dataset.from_list(rows)


def reward_correct(completions, reference_verdict, **kwargs):
    """+1 when the model's own verdict matches the reference verdict."""
    out = []
    for comp, ref in zip(completions, reference_verdict):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(1.0 if parse_verdict(text) == ref else 0.0)
    return out


def reward_format(completions, **kwargs):
    """+0.2 for leading with a verdict word at all (keeps answers verdict-first)."""
    out = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.2 if parse_verdict(text) != "none" else 0.0)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(SFT_MERGED)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # GRPO generates in-loop; TRL casts the model to bf16 right before generate(),
    # which corrupts 4-bit weights. So the policy runs as bf16 + LoRA (not QLoRA) —
    # still a frozen base with a small trainable adapter, still fits one 24GB GPU.
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # let grads reach the LoRA params through the frozen base

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    args = GRPOConfig(
        output_dir=str(OUT),
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1 if not SMOKE else 1,
        max_steps=3 if SMOKE else -1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_prompt_length=768,
        max_completion_length=200,
        temperature=0.9,
        top_p=1.0,
        beta=0.04,
        loss_type="bnpo",
        scale_rewards=True,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_correct, reward_format],
        args=args,
        train_dataset=build_dataset(),
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"saved GRPO adapter -> {OUT}")


if __name__ == "__main__":
    main()
