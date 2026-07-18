"""Assemble the DPO preference file from reference + base generations.

chosen   = the curated PubMedQA reference answer (verdict-first, with disclaimer)
rejected = raw BioMistral-7B output for the same prompt (abstract-style, no verdict)

Output is LLaMA-Factory alpaca-ranking format.
"""
import json
from pathlib import Path

RL = Path(__file__).resolve().parent.parent
GEN = RL / "data" / "base_generations.json"
OUT = RL / "data" / "medical_dpo.json"

gens = json.load(open(GEN))
pairs = []
for r in gens:
    chosen = r["reference"].strip()
    rejected = r["base_output"].strip()
    # drop degenerate pairs: empty rejected, or the two collapsed to the same text
    if len(rejected) < 5 or rejected == chosen:
        continue
    pairs.append({
        "instruction": r["instruction"],
        "input": r["input"],
        "chosen": chosen,
        "rejected": rejected,
    })

OUT.write_text(json.dumps(pairs, indent=2))
print(f"wrote {len(pairs)} preference pairs -> {OUT}")
