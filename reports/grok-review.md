# Independent Technical Review: DPO vs GRPO Medical LLM Comparison

**Source:** `comparison.md` (claims) vs `data/{base,sft,dpo,grpo}.json` (raw 100-row held-out eval) and `metrics.json`.  
**Method:** Recomputed all headline and behavioral metrics from the four JSON files; treated `correct` as the ground truth for accuracy (`mean(correct)`), and cross-checked against `gen_verdict == ref_verdict` (100% consistent for all four models).  
**Verdict on numerics:** All report tables that depend on the eval dumps **PASS**. Interpretive claims are mostly directionally sound but several are overstated relative to sample size and metric design.

---

## (1) Numerical Audit

### 1.1 Headline table

Report values (rounded as shown):

| Model | Verdict accuracy | Disclaimer rate | No-verdict rate | Mean length (chars) |
|-------|------------------|-----------------|-----------------|---------------------|
| Base  | 0%               | 0%              | 100%            | 462                 |
| SFT   | 75%              | 100%            | 0%              | 245                 |
| DPO   | 75%              | 100%            | 0%              | 229                 |
| GRPO  | 75%              | 100%            | 0%              | 244                 |

Recomputed from `data/*.json` rows (`n=100` each):

| Model | mean(correct) | mean(disclaimer) | mean(gen_verdict==none) | mean(len(generation)) | report int length |
|-------|---------------|------------------|-------------------------|-----------------------|-------------------|
| base  | 0.00          | 0.00             | 1.00                    | 461.84                | 462               |
| sft   | 0.75          | 1.00             | 0.00                    | 245.15                | 245               |
| dpo   | 0.75          | 1.00             | 0.00                    | 229.15                | 229               |
| grpo  | 0.75          | 1.00             | 0.00                    | 244.32                | 244               |

| Check | Result |
|-------|--------|
| Base accuracy 0% | **PASS** (0/100 correct) |
| SFT / DPO / GRPO accuracy 75% | **PASS** (75/100 each) |
| Base disclaimer 0%, others 100% | **PASS** |
| Base no-verdict 100%, others 0% | **PASS** (base: 100×`none`) |
| Mean lengths 462 / 245 / 229 / 244 | **PASS** (integer round of recomputed means) |

**Note (not a report mismatch):** `metrics.json` stores lengths to one decimal (461.8, 245.2, 229.2, 244.3), which are 1-d.p. rounds of the exact means above. The report’s integer table is consistent with both.

`correct` flag is exactly `gen_verdict == ref_verdict` for every row of every model (0 inconsistencies).

---

### 1.2 “vs SFT” table (verdicts changed / answers reworded)

Report claims: **DPO 11 / 78**, **GRPO 2 / 40**.

Recomputed by pairing each model to SFT on the same instruction (rows share identical order across all four files):

| Model | Verdicts changed (`gen_verdict` ≠ SFT) | Generations textually different | Generations identical |
|-------|----------------------------------------|---------------------------------|------------------------|
| DPO   | **11**                                 | **78**                          | 22                     |
| GRPO  | **2**                                  | **40**                          | 60                     |

| Check | Result |
|-------|--------|
| DPO 11 changed / 78 reworded | **PASS** |
| GRPO 2 changed / 40 reworded | **PASS** |

Whitespace-insensitive comparison (`strip`) yields the same rewrite counts (78 / 40).

**Flip detail (supports later critique):**
- DPO’s 11 flips are **all toward `yes`**: 8× `no→yes`, 3× `maybe→yes`. Paired accuracy: 5 wins, 5 losses, 1 wrong→wrong → net 0.
- GRPO’s 2 flips: `no→yes` (ref=`maybe`, both wrong) and `yes→maybe` (ref=`no`, both wrong) → 0 accuracy change.

---

### 1.3 Predicted-verdict distribution

Report (ref: 60 yes / 31 no / 9 maybe):

| Model | yes | no | maybe |
|-------|-----|----|-------|
| SFT   | 59  | 38 | 3     |
| DPO   | 70  | 30 | 0     |
| GRPO  | 59  | 37 | 4     |

Recomputed `Counter(gen_verdict)`:

| Model | yes | no | maybe | none |
|-------|-----|----|-------|------|
| base  | 0   | 0  | 0     | 100  |
| sft   | 59  | 38 | 3     | 0    |
| dpo   | 70  | 30 | 0     | 0    |
| grpo  | 59  | 37 | 4     | 0    |

Reference labels (all four files identical): **60 yes / 31 no / 9 maybe**.

| Check | Result |
|-------|--------|
| Ref 60/31/9 | **PASS** |
| SFT 59/38/3 | **PASS** |
| DPO 70/30/0 | **PASS** |
| GRPO 59/37/4 | **PASS** |

---

### 1.4 Per-class recall

Report:

| Model | yes   | no    | maybe |
|-------|-------|-------|-------|
| SFT   | 49/60 | 26/31 | 0/9   |
| DPO   | 54/60 | 21/31 | 0/9   |
| GRPO  | 49/60 | 26/31 | 0/9   |

Recomputed (`correct` among rows with each `ref_verdict`):

| Model | yes   | no    | maybe | sum correct |
|-------|-------|-------|-------|-------------|
| base  | 0/60  | 0/31  | 0/9   | 0           |
| sft   | 49/60 | 26/31 | 0/9   | 75          |
| dpo   | 54/60 | 21/31 | 0/9   | 75          |
| grpo  | 49/60 | 26/31 | 0/9   | 75          |

| Check | Result |
|-------|--------|
| SFT 49/60, 26/31, 0/9 | **PASS** |
| DPO 54/60, 21/31, 0/9 | **PASS** |
| GRPO 49/60, 26/31, 0/9 | **PASS** |
| Every model 0/9 on `maybe` | **PASS** (base, SFT, DPO, GRPO all 0/9) |

**Extra (not in report table):** SFT’s 3 predicted `maybe`s all land on `ref=yes` (never on true maybe). GRPO’s 4 predicted `maybe`s land on refs `[yes, yes, no, yes]`. So models sometimes *emit* `maybe`, but never correctly on the held-out maybe class.

---

### 1.5 Training-curve tables (secondary, from `metrics.json` only)

Not recomputable from `data/*.json`, but checked against `metrics.json` curves:

| Check | Result |
|-------|--------|
| DPO loss/margin/acc table (steps 5…95) | **PASS** vs `metrics.json` (e.g. step 5 acc 0.225 → “23%”, margin ≈ −0.00) |
| GRPO 10-step window averages (correct / total / KL) | **PASS** (all six windows match the report to displayed precision) |

---

### 1.6 Numerical audit summary

**All primary numbers in the report that derive from the evaluation dumps are correct.** No MISMATCH on headline metrics, vs-SFT diffs, predicted distributions, per-class recall, or the 0/9 maybe claim. The only nuance is cosmetic length rounding (`metrics.json` 1 d.p. vs exact means), which does not change the report table.

---

## (2) Critique — Soundness as an Educational ML Writeup

### 2.1 What the report gets right

1. **SFT did the heavy lifting** is well supported. Base is 0/100 on verdict accuracy with 100% `none`; SFT jumps to 75/100 and 100% structured verdicts + the fixed disclaimer footer. That is a qualitative capability gap, not a small effect size.

2. **Same headline score, different behavior** is the best pedagogical point. DPO and GRPO both sit at 75%, but DPO rewrites 78/100 answers and flips 11 verdicts; GRPO rewrites 40 and flips 2. Presenting the vs-SFT and class-recall tables prevents the false conclusion “alignment methods did nothing.”

3. **DPO yes-bias** is real in this dump, not just narrative. Predicted yes rises 59 → 70; all 11 SFT→DPO flips are toward `yes`. Class recall: yes +5 (49→54), no −5 (26→21). The report’s “gaining on some classes, losing on others, for no net accuracy change” is literally exact here.

4. **GRPO stayed anchored to SFT** matches the data: 98/100 identical verdicts, identical per-class recall to SFT, KL in training curves remaining ~1e−3, reward not climbing. “KL leash” is a reasonable mechanism story given those numbers.

5. **Maybe blind spot** is correctly highlighted. 0/9 across post-SFT models is the only class with total failure; calling it the next lever (data rebalance / class-aware reward / better DPO rejections) is appropriate.

6. **DPO rejected = raw base** is acknowledged in takeaways and the repro script — important honesty for readers who might assume rejected = wrong-verdict hard negatives.

---

### 2.2 Overclaims, soft language, and missing caveats

#### A. “RL is not magic” / “75% is near what SFT already extracted”

Directionally fair for *this* setup, but over-generalized as a takeaway:
- Only **one** DPO run and **one** GRPO run, small LoRA budget, short training (DPO ~100 steps with preference accuracy saturating at 100% by step 15; GRPO 60 steps, 4 prompts/step).
- Metric is **verdict-token accuracy only** — rationale quality, faithfulness, and medical safety are unmeasured. DPO clearly changes style (shorter answers: 245 → 229 mean chars; more paraphrasing), which could be good or bad under a better metric.
- DPO’s preference task (SFT-style chosen ≻ base-style rejected) is mostly a **format/style contrast**, not a correctness contrast. Preferring “has a verdict + disclaimer” over free-form base text can hit 100% train pref-acc without teaching harder yes/no/maybe decisions. That makes “RL didn’t raise accuracy” partly a **signal design** result, not a general fact about RLHF/DPO/GRPO.

#### B. Statistical thin ice at n=100 (and n=9)

| Claim type | Issue |
|------------|--------|
| Overall 75% vs 75% | Exact tie, but Wilson/approx 95% CI for a single 75/100 rate is roughly **[0.66, 0.84]**. The experiment is underpowered to detect modest true gains (e.g. +5–8 pts). |
| DPO class shifts (+5 yes, −5 no) | Only **11** verdict disagreements; McNemar counts for accuracy are **5 vs 5**. No evidence of net accuracy change — also no power to claim DPO “doesn’t help.” |
| “Every model 0/9 on maybe” | True on this set, but **n=9**. Rule-of-3-style upper bound on a 0-event rate is ~30%+. Framing as *the* shared blind spot is reasonable as a qualitative finding; treating 0/9 as a precise capability estimate is not. |
| GRPO “2 changed” | Could be sampling noise under greedy or near-greedy decoding; two flips both wrong→wrong. |

The report never reports CIs, paired tests, or multiple seeds. For an educational writeup that is forgivable, but phrases like “RL redistributes and stabilizes existing behavior; it does not inject knowledge” should be scoped to **this experiment**, not stated as a law.

#### C. Disclaimer rate

All three post-SFT models score 100% disclaimer, and every generation inspected ends with the fixed string  
`[Based on published research. Consult healthcare provider.]`.  
That makes `disclaimer_rate` a **nearly trivial format metric** (likely regex/substring match on a memorized footer from SFT), not independent evidence of safety behavior. Base at 0% mostly shows it never learned the template. The report lists the metric in the headline table without warning that it is saturated and low-signal once SFT has run.

#### D. Decoding / evaluation protocol

The report does not state temperature, top-p, seed, or greedy vs sampling. If eval is greedy (common for vLLM eval scripts), then:
- Rewrite rates (78 / 40) reflect **deterministic** policy differences — good.
- But there is still **no multi-seed training variance**, so method differences are single-run anecdotes.

#### E. Verdict-only accuracy ignores rationale

Examples in the report show SFT/GRPO often copy paper-ish abstracts while DPO shortens and rephrases. Under the official metric they can be equally “correct” while differing in:
- factual compression errors,
- hedging language,
- whether the explanation supports the verdict.

Especially for medical use, this is a material gap. The writeup mentions style change but still centers the story on verdict accuracy.

#### F. DPO rejected = base output (design caveat)

Takeaway #4 mentions better rejections, but the main narrative could stress earlier that **DPO was never optimized for verdict correctness**. Comparing DPO and GRPO on verdict accuracy is slightly apples-to-oranges: GRPO’s reward is verifiable verdict match; DPO’s BT objective is “look more like chosen than base.” A reader might conclude “DPO fails at accuracy” when the setup barely asked it to improve accuracy.

#### G. “Maybe is only 9% of the training data”

Stated as fact; **not verifiable from the files in this folder** (only the test split is present). If training is similarly imbalanced, the claim is plausible; if not, the blind spot needs another explanation (label ambiguity, format priors, reward sparsity). Should be marked as a hypothesis or backed by a train-label histogram.

#### H. GRPO reward “hovers near starting level”

True for the windowed means (~0.60–0.70 correct reward). That is consistent with “no learning,” but also with **reward noise + tiny batches (4 prompts)** + strong KL, or with the policy already near the reward ceiling on easy yes/no items. The curve does not distinguish “GRPO can’t improve” from “GRPO wasn’t given enough signal/budget.”

#### I. Example section

The dialysis example shows all post-SFT models correct with style differences — fine for style. The sternal-fracture `maybe` example shows SFT=`no`, DPO=`yes`, GRPO=`yes`, all wrong — good illustration of the maybe failure and of DPO/GRPO not fixing it. It does **not** illustrate GRPO “staying close to SFT” (here GRPO flipped away from SFT). One carefully chosen GRPO-identical example would balance the section.

---

### 2.3 Are the interpretive claims justified?

| Claim | Data support | Verdict |
|-------|--------------|---------|
| DPO drifted to a yes-bias | 59→70 yes preds; 11/11 flips toward yes; +5 yes / −5 no recall | **Justified** (descriptive); causal “common DPO distribution-shift effect” is plausible but not proven by one run |
| GRPO stayed anchored to SFT | 2 flips, 60 identical texts, same recall, tiny KL | **Justified** for this run |
| SFT did the heavy lifting | 0% → 75%, format emergence | **Justified** |
| RL is not magic / neither raised accuracy | Both +0 pts; DPO McNemar 5–5 | **Justified as outcome of this setup**; overclaimed as general lesson without seeds/CIs/stronger signals |
| Signal shapes the change | DPO rewrites more + shifts class balance; GRPO local edits | **Justified** as the cleanest comparative lesson |
| Shared maybe blind spot is where to push | 0/9 all models; class rare | **Directionally justified**; n=9 and train-% claim weak |

---

### 2.4 Prioritized suggestions (most valuable first)

1. **Report uncertainty and paired stats.** For overall accuracy: bootstrap or Wilson CIs. For SFT vs DPO/GRPO: McNemar (or exact binomial on discordant pairs). State explicitly that 5–5 discordant pairs for DPO mean “no detectable difference,” not “proven equal.”

2. **Multi-seed or multi-checkpoint eval.** Even 3 seeds for DPO and GRPO would separate method effect from run noise — essential before teaching “DPO biases yes / GRPO anchors.”

3. **Fix the DPO data construction for a fair correctness comparison.** Build rejected answers that are **wrong-verdict** (or high-reward-margin hard negatives), not raw base free-text. Optionally run a second DPO condition with the current base-rejected setup to show the pedagogical contrast.

4. **Enrich the metric beyond verdict token.** At minimum: (a) exact verdict, (b) disclaimer present, (c) rubric or LLM-judge on rationale faithfulness, (d) length/verbosity. Medical alignment claims need more than yes/no/maybe matching.

5. **Address class imbalance explicitly in training and metrics.** Oversample `maybe`, report macro-F1 / per-class recall as primary, or use class-weighted GRPO reward. Do not let 60% yes dominate the headline accuracy.

6. **Document eval decoding** (greedy? temp? max tokens?) and the **disclaimer detector** (regex). Note that disclaimer_rate is saturated post-SFT and is a format check.

7. **Show train label distribution** to ground the “maybe is 9% of training data” sentence; add a confusion matrix (not only recall) so readers see false-positive yes inflation under DPO.

8. **Scale the GRPO experiment slightly** before concluding it “stays flat”: more prompts per step, more steps, or a looser KL — or report a small hyperparameter sweep. Current KL ~1e−3 and flat reward invite the reading “didn’t train,” not only “anchored by design.”

9. **Optional statistical honesty box** for students: with n=100, detecting a true +5 pt gain at 80% power needs roughly this order of sample size or paired design with more discordant items; n=9 for maybe cannot support strong per-class claims.

10. **Keep the structure** (same score → different failures → signal shapes behavior). That framing is the report’s strongest educational asset; the fixes above mostly add rigor around it rather than replace it.

---

## Bottom line

- **Numerical audit:** Full **PASS**. Headline metrics, vs-SFT 11/78 and 2/40, predicted distributions, per-class recall, and universal 0/9 on `maybe` all recompute exactly from `data/*.json`. Training-curve tables match `metrics.json`.
- **Critique:** Solid educational comparison of *behavior under different objectives*, with honest acknowledgment that SFT carries accuracy. Main weaknesses are **overgeneralized “RL is not magic” messaging**, **underpowered n=100 / n=9 inference**, **verdict-only metric**, **saturated disclaimer metric**, and **DPO rejected=base** making the accuracy comparison somewhat misaligned with DPO’s actual training signal. Strengthening uncertainty reporting, seeds, metrics, and preference construction would make the same narrative scientifically tighter without losing clarity.
