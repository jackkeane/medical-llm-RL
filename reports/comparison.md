# DPO vs GRPO — comparison on the medical SFT model

Both methods start from the **same merged SFT checkpoint** (`models/sft-merged`) and train a fresh rank-16 LoRA adapter on one RTX 4090. They differ in *what signal they optimize*.

## Headline metrics (100-sample held-out test set)

*Greedy decoding (temperature 0), verdict parsed as the leading yes/no/maybe token. `Disclaimer rate` is a substring check for a memorized footer — it saturates at 100% the moment SFT is applied, so it separates base from finetuned but says little between the finetuned variants.*

| Model | Verdict accuracy | Disclaimer rate | No-verdict rate | Mean length (chars) |
|---|---|---|---|---|
| Base (BioMistral-7B) | 0% | 0% | 100% | 462 |
| SFT | 75% | 100% | 0% | 245 |
| DPO | 75% | 100% | 0% | 229 |
| GRPO | 75% | 100% | 0% | 244 |

## What moved

- **DPO** shifted verdict accuracy by **+0 pts** vs SFT (75% → 75%). DPO optimizes a *preference* (curated answer ≻ raw base answer), so its main effect is on answer **style/consistency**, not necessarily verdict correctness.
- **GRPO** shifted verdict accuracy by **+0 pts** vs SFT (75% → 75%). GRPO optimizes a *verifiable reward* (does the verdict match?), so it targets **correctness** directly.

## Same score, different models

All three post-SFT models score the same overall, so the averages hide what each method actually did. Measured against the SFT model, question by question:

| vs SFT | Verdicts changed /100 | Answers reworded /100 |
|---|---|---|
| DPO | 11 | 78 |
| GRPO | 2 | 40 |

- **GRPO stayed close to SFT.** Its KL penalty anchors the policy to the starting model, so it reworded answers but rarely overturned a verdict.
- **DPO drifted further.** With no answer it was told to stay near, the preference signal rewrote most answers and flipped more verdicts — gaining on some classes, losing on others, for no net accuracy change.

### Predicted-verdict distribution (reference: 60 yes / 31 no / 9 maybe)

| Model | yes | no | maybe |
|---|---|---|---|
| SFT | 59 | 38 | 3 |
| DPO | 70 | 30 | 0 |
| GRPO | 59 | 37 | 4 |

DPO's preference optimization pushed the model toward answering **yes** more often (a common DPO distribution-shift effect); GRPO kept SFT's balance.

### Per-class recall (correct / total for each reference class)

| Model | yes | no | maybe |
|---|---|---|---|
| SFT | 49/60 | 26/31 | 0/9 |
| DPO | 54/60 | 21/31 | 0/9 |
| GRPO | 49/60 | 26/31 | 0/9 |

> **The shared blind spot:** every model scores **0/9 on `maybe`**. It is the rarest class — 90 of 800 training examples (11%) and just 9 of the 100 test items — and none of SFT, DPO, or GRPO learned to produce it correctly. With only 9 test cases, treat 0/9 as a qualitative flag rather than a precise rate; still, it — not the yes/no split — is the clearest place a targeted reward or rebalanced data would help.

## DPO training signal

Preference margin (reward for *chosen* − reward for *rejected*) and implicit accuracy over training:

| Step | Loss | Reward margin | Pref. accuracy |
|---|---|---|---|
| 5 | 0.695 | -0.00 | 23% |
| 15 | 0.599 | 0.20 | 100% |
| 25 | 0.309 | 1.05 | 100% |
| 35 | 0.118 | 2.32 | 100% |
| 45 | 0.073 | 3.39 | 100% |
| 55 | 0.029 | 4.30 | 100% |
| 65 | 0.021 | 4.97 | 100% |
| 75 | 0.010 | 5.22 | 100% |
| 85 | 0.009 | 5.36 | 100% |
| 95 | 0.010 | 5.83 | 100% |

## GRPO training signal

Per-step reward is noisy (only 4 prompts per optimizer step over 60 steps), so this shows the verdict-match reward averaged over consecutive windows. It hovers near its starting level — consistent with GRPO staying anchored to SFT rather than climbing:

| Steps | Avg. verdict-correct reward | Avg. total reward | Avg. KL |
|---|---|---|---|
| 1–10 | 0.60 | 0.80 | 0.0003 |
| 11–20 | 0.70 | 0.90 | 0.0005 |
| 21–30 | 0.67 | 0.87 | 0.0007 |
| 31–40 | 0.60 | 0.80 | 0.0008 |
| 41–50 | 0.64 | 0.84 | 0.0008 |
| 51–60 | 0.60 | 0.80 | 0.0009 |

## Example outputs (where the models disagree)

**Q:** Does increasing blood pH stimulate protein synthesis in dialysis patients?  
**Reference verdict:** `no`

- **Base (BioMistral-7B)** `none` ✗: The study by Krediet et al. aimed to investigate whether alkalizing patients on hemodialysis with oral sodium bicarbonate supplementation would have a positive effect on protein sy
- **SFT** `no` ✓: no In patients on hemodialysis, oral NaHCO(3) supplementation induced a significant increase in serum bicarbonate and blood pH, but did not change protein synthesis rates or nutrit
- **DPO** `no` ✓: no Increasing blood pH by oral NaHCO(3) supplementation does not stimulate protein synthesis in patients on hemodialysis. [Based on published research. Consult healthcare provider.
- **GRPO** `no` ✓: no In patients on hemodialysis, oral NaHCO(3) supplementation induced a significant increase in serum bicarbonate and blood pH, but did not change protein synthesis rates or nutrit

*Note how GRPO's answer here is byte-identical to SFT (its KL leash kept it in place) while DPO rewrote the same answer — the anchoring vs. drift contrast, in one question.*

**Q:** Sternal fracture in growing children : A rare and often overlooked fracture?  
**Reference verdict:** `maybe`

- **Base (BioMistral-7B)** `none` ✗: [CONCLUSION] Sternal fractures in children are rare but should be considered in the differential diagnosis of chest trauma. The diagnosis can be made by clinical examination and co
- **SFT** `no` ✗: no Sternal fractures in children are rare and often overlooked. The accident mechanism is important for the detection of the fracture. Sonography is a useful tool for the detection
- **DPO** `yes` ✗: yes Sternal fractures in children are rare and often overlooked. The accident mechanism is important for the detection. Sonography is a useful tool for the detection of sternal fra
- **GRPO** `yes` ✗: yes Sternal fractures in children are rare and often overlooked. The accident mechanism is important for the detection of the fracture. Sonography is a useful tool for the detectio

## How to read this (caveats)

This is a compact educational demonstration, not a benchmark. Before generalizing any number above:

- **Single run, no seeds or confidence intervals.** Each method was trained once. At n=100, a 75% rate has a 95% interval of roughly 66–84%, so this experiment cannot detect a modest true gain (±5–8 pts) — "no accuracy change" means *not detectable here*, not *proven equal*. The `maybe` class is n=9; its 0/9 is a flag, not a measurement.
- **Verdict-only metric.** Accuracy scores just the leading yes/no/maybe token. It ignores rationale quality, faithfulness, and safety — exactly where DPO's rewriting might help or hurt. A rubric or LLM-judge on the explanation would tell a fuller story.
- **DPO and GRPO optimized different things.** GRPO's reward *is* verdict correctness; DPO's preference was only "look like the curated answer, not the raw base output" — a style/format contrast, not a correctness one. So comparing them on verdict accuracy is partly apples-to-oranges: DPO was never really asked to fix verdicts. Rejected answers built from *wrong-verdict* negatives would make it a fair correctness comparison.

## Takeaways

1. **SFT did the heavy lifting.** Base → SFT is 0% → 75% verdict accuracy; the base model never even emits a verdict. Alignment starts from a capable model.
2. **In this setup, RL didn't move headline accuracy.** Neither DPO nor GRPO changed the 75% — on a 7B model with a small preference/reward budget and a verdict-only metric, SFT had already extracted what a single short run recovers. Read this as an outcome of *this* experiment (see caveats), not a law about RLHF: RL here redistributed and stabilized existing behavior rather than adding knowledge.
3. **The signal shapes the change.** DPO (learn a *preference*) rewrote answer style aggressively and shifted the output distribution; GRPO (optimize a *verifiable reward* under a KL leash) stayed anchored to SFT. Same score, different failure modes — the clearest lesson here.
4. **Where to push next:** the `maybe` class (0/9) is the clearest target — more `maybe` data, a class-balanced reward, more GRPO steps at lower KL, or DPO pairs built from *wrong-verdict* rejections rather than raw-base rejections. Adding seeds and paired tests (McNemar) would let the behavioral differences be stated with confidence.

## Reproduce

```bash
# 1. merge SFT adapter -> shared start policy (py312 / LLaMA-Factory)
llamafactory-cli export configs/export_sft.yaml
# 2. DPO preference data (rejected = base generations, vLLM) + train
~/.venv/bin/python scripts/gen_base_vllm.py && python scripts/build_dpo.py
llamafactory-cli train configs/dpo.yaml
# 3. GRPO with verifiable verdict reward (TRL)
python scripts/train_grpo.py
# 4. merge both, evaluate 4 variants, build this report
llamafactory-cli export configs/export_dpo.yaml
llamafactory-cli export configs/export_grpo.yaml
for n in base sft dpo grpo; do ~/.venv/bin/python scripts/eval_vllm.py --model models/$n-merged --name $n; done
python scripts/make_report.py
```
