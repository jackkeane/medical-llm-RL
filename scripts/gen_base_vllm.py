"""Generate base-model (pre-SFT) answers for the training prompts with vLLM.

These become the *rejected* side of the DPO preference pairs: raw BioMistral-7B
output, before any of our finetuning taught it the curated PubMedQA answer style.
Run with the vLLM venv:  ~/.venv/bin/python scripts/gen_base_vllm.py

Prompt construction here is the single source of truth reused by eval_vllm.py so
every model is scored on identical inputs.
"""
import json
from pathlib import Path
from vllm import LLM, SamplingParams

RL = Path(__file__).resolve().parent.parent
BASE_MODEL = RL.parent / "medical-llm" / "models" / "biomistral-7b"
TRAIN = RL.parent / "medical-llm" / "data" / "processed" / "medical_train.json"
OUT = RL / "data" / "base_generations.json"

MAX_INPUT_CHARS = 2400  # bound the abstract so prompts stay well under the context window


def user_content(ex: dict) -> str:
    """Combine instruction + context the way the Mistral chat turn expects."""
    inp = (ex.get("input") or "").strip()
    if len(inp) > MAX_INPUT_CHARS:
        inp = inp[:MAX_INPUT_CHARS]
    instr = ex["instruction"].strip()
    return f"{instr}\n\n{inp}" if inp else instr


def build_prompt(tokenizer, ex: dict) -> str:
    msgs = [{"role": "user", "content": user_content(ex)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    data = json.load(open(TRAIN))
    llm = LLM(
        model=str(BASE_MODEL),
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=False,
    )
    tok = llm.get_tokenizer()
    prompts = [build_prompt(tok, ex) for ex in data]
    params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=220, seed=0,
        truncate_prompt_tokens=3600,
    )
    outs = llm.generate(prompts, params)
    records = []
    for ex, o in zip(data, outs):
        records.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "reference": ex["output"],
            "base_output": o.outputs[0].text.strip(),
        })
    OUT.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} base generations -> {OUT}")
    # quick peek
    for r in records[:2]:
        print("\nQ:", r["instruction"][:90])
        print("REF   :", r["reference"][:120])
        print("BASE  :", r["base_output"][:160])


if __name__ == "__main__":
    main()
