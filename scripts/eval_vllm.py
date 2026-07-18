"""Evaluate one model variant on the held-out PubMedQA test set with vLLM.

Scores three things on identical prompts (same builder as gen_base_vllm.py):
  - verdict accuracy : does the generated yes/no/maybe match the reference?
  - disclaimer rate  : fraction of answers carrying a medical-disclaimer phrase
  - mean length      : answer length in characters

Run once per variant, e.g.:
    ~/.venv/bin/python scripts/eval_vllm.py --model models/biomistral-7b     --name base
    ~/.venv/bin/python scripts/eval_vllm.py --model models/sft-merged        --name sft
    ~/.venv/bin/python scripts/eval_vllm.py --model models/dpo-merged        --name dpo
    ~/.venv/bin/python scripts/eval_vllm.py --model models/grpo-merged       --name grpo
"""
import argparse
import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams

RL = Path(__file__).resolve().parent.parent
TEST = RL.parent / "medical-llm" / "data" / "processed" / "medical_test.json"
OUTDIR = RL / "reports" / "eval"
MAX_INPUT_CHARS = 2400

DISCLAIMER_RE = re.compile(
    r"(consult|healthcare provider|medical professional|physician|not.*substitute|"
    r"educational purpose|seek.*advice|qualified)", re.I)


def parse_verdict(text: str) -> str:
    m = re.match(r"\s*(yes|no|maybe)\b", text.strip().lower())
    return m.group(1) if m else "none"


def user_content(ex: dict) -> str:
    inp = (ex.get("input") or "").strip()[:MAX_INPUT_CHARS]
    instr = ex["instruction"].strip()
    return f"{instr}\n\n{inp}" if inp else instr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    data = json.load(open(TEST))
    llm = LLM(model=str(RL / args.model) if not args.model.startswith("/") else args.model,
              dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": user_content(ex)}],
                                tokenize=False, add_generation_prompt=True)
        for ex in data
    ]
    # greedy decode for a deterministic, reproducible verdict
    params = SamplingParams(temperature=0.0, max_tokens=220, truncate_prompt_tokens=3600)
    outs = llm.generate(prompts, params)

    rows, correct, disc, lengths = [], 0, 0, []
    for ex, o in zip(data, outs):
        gen = o.outputs[0].text.strip()
        ref_v = parse_verdict(ex["output"])
        gen_v = parse_verdict(gen)
        ok = gen_v == ref_v
        correct += ok
        has_disc = bool(DISCLAIMER_RE.search(gen))
        disc += has_disc
        lengths.append(len(gen))
        rows.append({"instruction": ex["instruction"], "ref_verdict": ref_v,
                     "gen_verdict": gen_v, "correct": ok, "disclaimer": has_disc,
                     "generation": gen})

    n = len(data)
    summary = {
        "name": args.name,
        "n": n,
        "verdict_accuracy": round(correct / n, 4),
        "disclaimer_rate": round(disc / n, 4),
        "mean_length_chars": round(sum(lengths) / n, 1),
        "verdict_none_rate": round(sum(1 for r in rows if r["gen_verdict"] == "none") / n, 4),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{args.name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
