# REVIEW — DPO vs GRPO educational walkthrough

Sources checked: `docs/*.html`, `docs/assets/{charts.js,style.css}`, `data/{base,sft,dpo,grpo}.json`, `data/metrics.json`, `data/comparison.md`. All eval figures recomputed from the 100-row `rows` arrays (and `metrics.json` curves for training signals).

---

## (1) DATA ACCURACY

### Verdict accuracy (base / SFT / DPO / GRPO)

| Model | HTML claim | Recomputed `mean(correct)` | Result |
|---|---|---|---|
| Base | 0% (`index.html`, `04-comparison.html`) | 0/100 = **0.00** | **PASS** |
| SFT | 75% | 75/100 = **0.75** | **PASS** |
| DPO | 75% | 75/100 = **0.75** | **PASS** |
| GRPO | 75% | 75/100 = **0.75** | **PASS** |

Matches `data/*/json` `summary.verdict_accuracy`, `data/metrics.json` `eval.*`, and `data/comparison.md`. Base: all 100 `gen_verdict == "none"`, none correct.

### Difference from SFT (pair on same `instruction`)

| Measure | HTML (`04-comparison.html`) | Recomputed | Result |
|---|---|---|---|
| DPO answers reworded | 78 | **78** (`generation` ≠ SFT) | **PASS** |
| DPO verdict flips | 11 | **11** (`gen_verdict` ≠ SFT) | **PASS** |
| GRPO answers reworded | 40 | **40** | **PASS** |
| GRPO verdict flips | 2 | **2** | **PASS** |

Also matches `metrics.json` `behavior.vs_sft` (DPO identical 22 / rewritten 78 / flips 11; GRPO identical 60 / rewritten 40 / flips 2).

Prose claims in `04-comparison.html`:
- “GRPO’s correct-answer set is … identical to SFT’s” — **PASS** (same 75 correct instructions).
- “DPO … swapped 5 questions in for 5 others” — **PASS** (5 gained, 5 lost vs SFT; net accuracy unchanged).

### Predicted-verdict counts

| Source | yes | no | maybe | Result |
|---|---|---|---|---|
| Reference (truth) | 60 | 31 | 9 | **PASS** (from `ref_verdict`) |
| SFT | 59 | 38 | 3 | **PASS** |
| DPO | 70 | 30 | 0 | **PASS** |
| GRPO | 59 | 37 | 4 | **PASS** |

Matches HTML table/charts in `04-comparison.html`, `metrics.json` `behavior.pred_dist`, and `comparison.md`.

### Per-class recall (correct count for each `ref_verdict`)

| Model | yes (of 60) | no (of 31) | maybe (of 9) | Result |
|---|---|---|---|---|
| SFT | 49 | 26 | 0 | **PASS** |
| DPO | 54 | 21 | 0 | **PASS** |
| GRPO | 49 | 26 | 0 | **PASS** |

Matches `04-comparison.html` recall table, prose “26 → 21” / “49 → 54”, “0 of 9” / “0/9”, and `metrics.json` `behavior.recall`.

### DPO training curve (`02-dpo.html` vs `metrics.json` `curves.dpo`)

`metrics.json` stores per step: `step`, `loss`, `margin`, `acc` only — **not** separate chosen/rejected rewards.

| Check | Result |
|---|---|
| Qualitative: chosen stays ~0, rejected falls to ~−5.8 | **PASS** (chart series: chosen ∈ [−0.217, 0.012]; rejected ends **−5.785** at step 100; aria-label “about minus 5.8”) |
| Margin = chosen − rejected (chart points) | **PASS** (within ~0.001 of `metrics.margin` at every logged step) |
| Table margins vs metrics (2-decimal display) | **PASS** (e.g. step 100: table 5.57, metrics 5.568; step 80: 5.74 vs 5.737; step 5: −0.00 vs −0.0037) |
| “By step 15 … 100% of pairs” | **PASS** (`acc` = 1.0 from step 15 onward) |
| Tile “Reward margin 0 → 5.6” | **PASS** (step 5 ≈ 0, step 100 = 5.57) |
| Individual chosen/rejected series as ground truth | **CANNOT FULLY RECONCILE** from `metrics.json` (only `margin` present). Series are self-consistent with margin and with the table at 2 d.p. rounding. |

Table vs chart script (2 d.p.): all six table rows match the script series when rounded; step 5 table `0.00 − 0.01 ≠ −0.00` is independent-column rounding, not a wrong metric.

### GRPO windowed rewards (`03-grpo.html` vs `metrics.json` `curves.grpo`)

60 curve points → six windows of 10. Recomputed means of `reward_correct` and `kl`:

| Steps | HTML reward (table / chart) | Recomputed `reward_correct` | HTML KL | Recomputed KL | Result |
|---|---|---|---|---|---|
| 1–10 | 0.60 / 0.60 | **0.6000** | 0.0003 | **0.000312** | **PASS** |
| 11–20 | 0.70 / 0.70 | **0.7000** | **0.0006** | **0.000548** | Reward **PASS**; KL **MISMATCH** (should display **0.0005** if rounded to 4 d.p.; `comparison.md` correctly has 0.0005) |
| 21–30 | 0.67 / 0.669 | **0.66875** | 0.0007 | **0.000656** | **PASS** (display rounding) |
| 31–40 | 0.60 / 0.60 | **0.6000** | 0.0008 | **0.000750** | **PASS** |
| 41–50 | 0.64 / 0.644 | **0.64375** | 0.0008 | **0.000784** | **PASS** |
| 51–60 | 0.60 / 0.60 | **0.6000** | 0.0009 | **0.000892** | **PASS** |

User-cited chart points `0.60 / 0.70 / 0.669 / 0.60 / 0.644 / 0.60`: **PASS** vs recomputed. KL band “~0.0003 to ~0.0009” and tile “≈0.001”: **PASS** as order-of-magnitude (window means 0.0003–0.0009; max raw point ≈0.00143).

### Example answers

**04 — dialysis pH, reference `no`**

- Instruction exists in all four eval files: *“Does increasing blood pH stimulate protein synthesis in dialysis patients?”*
- `ref_verdict`: **no** — **PASS**
- SFT / GRPO: `gen_verdict` **no**, correct; generations **byte-identical** — **PASS** (matches “SFT = GRPO (identical)”)
- DPO: `gen_verdict` **no**, correct; reworded text matches the quoted HTML (including disclaimer) — **PASS**
- Quoted SFT body is a faithful truncation of the real generation (ellipsis after “nutritional parameters”)

**02 — CPAP preference-pair sample**

- Chosen text (“neither BMI nor neck circumference… autoCPAP…”) and rejected text (“[IMPACT] The study is a continuation… obesity and OSA…”) appear **only** in `02-dpo.html`.
- **Neither string appears** in `data/{base,sft,dpo,grpo}.json` (searched instructions + generations).
- **CANNOT RECONCILE** with the provided eval/metrics data. Plausible as an out-of-sample training-pair illustration, but not verifiable from this folder’s machine-readable artifacts.

### Other numbers flagged

| Claim | Location | Status |
|---|---|---|
| Mean lengths / disclaimer rates not shown as headline tiles but present in `comparison.md` | report only | Consistent with JSON summaries (e.g. base 461.8 chars → 462; SFT 245.2 → 245; DPO 229.2 → 229; GRPO 244.3 → 244) |
| “90 of 800 training examples (11%)” for `maybe` | `04-comparison.html` | **Not in** `data/*` — **cannot verify** from provided files |
| DPO 800 pairs; GRPO 240 prompts balanced 80/80/80; 4 answers/group; 60 steps | `02` / `03` tiles & prose | **Not in** eval JSON / metrics curves metadata — **cannot verify** here |
| Pref. accuracy table in `comparison.md` (23% at step 5, etc.) | not duplicated as full table in HTML | metrics `acc` at step 5 = 0.225 → 23% **PASS** if cited |

**Bottom line (data):** All headline eval numbers on the pages match recomputation. One clear training-table slip: **GRPO KL for steps 11–20 is 0.0006 in HTML but 0.000548 from data**. DPO chosen/rejected *levels* and the CPAP pair text are not independently present in `data/*.json` / `metrics.json` curves.

---

## (2) HTML / ACCESSIBILITY / CORRECTNESS

### Chart ↔ details tables

| Figure | File | Match? |
|---|---|---|
| Verdict accuracy bars | `index.html`, `04-comparison.html` | **OK** — table 0/75/75/75% equals script `bars` |
| DPO divergence bars | `04-comparison.html` `#chart-div` | **OK** — 78/40/11/2 |
| Yes-count bars | `04-comparison.html` `#chart-yes` | **OK** for yes column (60/59/70/59); table also lists no/maybe (not plotted — intentional) |
| DPO chosen/rejected line | `02-dpo.html` `#chart-dpo` | **OK** at 2 d.p. (see §1); script series and table agree under rounding |
| GRPO reward line | `03-grpo.html` `#chart-grpo` | **Mostly OK** — rewards align; **KL column is table-only** and has the **0.0006 vs 0.0005** error above |

### aria-labels

- Index lifecycle SVG, setup merge SVG, GRPO loop SVG: descriptive and consistent with the diagrams.
- Chart aria-labels state the same headline numbers as the data (0/75s; DPO 78 & 11 / GRPO 40 & 2; yes 60/59/70/59; DPO rejected ≈ −5.8; GRPO reward ~0.6–0.7).
- **No aria-label found that contradicts the numbers.**

### Links, IDs, containers, markup

- Internal nav + pager links among the five pages all resolve (`index`, `01`–`04`); no broken relative `.html` targets.
- Chart container IDs used by scripts (`chart-acc`, `chart-dpo`, `chart-grpo`, `chart-div`, `chart-yes`) exist on the same page; no missing `getElementById` targets.
- No duplicate IDs within a page; no unclosed tags detected by a stack-based HTML parse of all five files.
- `01-setup.html` loads `charts.js` but draws no charts (harmless).

### Prose vs numbers

- “Identical” headline accuracy / score: **consistent** with 75% for SFT/DPO/GRPO.
- “0 of 9” / “0/9” on `maybe`: **consistent**.
- “78”, “11”, “40”, “2”: **consistent**.
- DPO yes-bias and recall trade (49→54 yes, 26→21 no): **consistent**.
- GRPO correct set identical to SFT; dialysis example byte-identical: **consistent**.
- **Issue:** `03-grpo.html` table KL **0.0006** for steps 11–20 does not match recomputed **0.0005** (and disagrees with `comparison.md`).

### Colour encoding

- CSS: `--series-1` blue, `--series-2` green, `--series-3` pink/rose, `--de-emphasis` grey.
- `02-dpo.html`: legend Chosen = `series-1`, Rejected = `series-3` — matches script series colors.
- `04-comparison.html` `#chart-div`: legend DPO `series-1`, GRPO `series-3` — matches bar colors.
- `#chart-yes`: DPO `series-1`, GRPO `series-3`, reference/SFT de-emphasized — consistent.
- GRPO chart uses `series-2` (green) for the single reward series; no multi-series legend conflict.

### Issues worth fixing (concrete)

1. **`03-grpo.html` L130** — KL `0.0006` → **`0.0005`** (window mean 0.000548).
2. **`02-dpo.html` L63–71** — CPAP pair is not in shipped `data/*.json`; either source it from training artifacts or label as illustrative / not from the held-out set.
3. **`02-dpo.html`** — chosen/rejected levels are not in `metrics.json`; if reproducibility matters, store them in metrics or state that only the margin is logged and levels are reconstructed.

No broken nav, no chart/script ID failures, no colour-legend contradictions found.

---

## (3) CLARITY / PEDAGOGY

The core story is accurate and well-scoped for an educational demo:

- SFT creates verdict capability (0% → 75%); short DPO/GRPO runs do not move the headline.
- DPO optimizes an *easy style/format preference* (curated ≻ raw base), so it rewrites heavily and drifts toward “yes.”
- GRPO optimizes a *verifiable verdict reward under a KL leash*, so it stays near SFT (flat reward, tiny KL, nearly identical correctness set).
- Caveats (single seed, n=100 CI, apples-to-oranges signals, `maybe` n=9) are explicit and honest.

**Over/under-claims (mild):**

- Over: “the model grades its own answers” (`index` chapter blurb) is slightly looser than “a rule scores generated answers” (body is clearer).
- Under: the walkthrough could stress more that **GRPO’s reward already includes format shaping (+0.2)** (`03` code snippet) while eval is verdict-only — a subtle train/eval gap.
- The flat GRPO reward is framed as both budget limit and intentional leash — good dual reading.

**Prioritized improvements (≤6):**

1. **Fix GRPO KL 11–20 (0.0006 → 0.0005)** so the only numeric slip disappears.
2. **Source or qualify the CPAP pair** so every quoted example is traceable to `data/` (dialysis already is).
3. **Log or footnote DPO chosen/rejected rewards** in `metrics.json` (or say “reconstructed so margin matches”) for full auditability.
4. **Show the 5-for-5 DPO swap** as a tiny contingency / list of instructions — turns “net zero” into a concrete pedagogical moment.
5. **One diagram of signal mismatch** (DPO preference = format; eval metric = verdict) next to the 75% bars — the strongest lesson currently lives mostly in prose.
6. **Optional: plot full verdict distribution** (stacked or grouped bars for yes/no/maybe), not only the “yes” count, so the `maybe` blind spot is visual as well as tabular.

---

*End of review. No HTML/CSS/JS files were modified; only this `REVIEW.md` was written.*
