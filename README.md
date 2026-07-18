# Medical LLM — RL / Preference Alignment

Stage 2 of an educational LLM lifecycle series: take the QLoRA-finetuned medical model
and align it two different ways — **DPO** (learn from a preference) and **GRPO** (optimize
a verifiable reward) — then compare what each actually does. All on a single RTX 4090.

⚠️ **DISCLAIMER:** For educational/research purposes only. NOT for clinical use.

## Where this fits in the series

| Stage | Repo | What it teaches |
|---|---|---|
| 1 · Supervised finetuning | [medical-llm](https://github.com/jackkeane/medical-llm) | QLoRA SFT of BioMistral-7B, visualized end to end |
| **2 · Preference alignment (RL)** | **this repo** | DPO vs GRPO on the SFT checkpoint |
| 3 · Compression | [medical-llm-pdq](https://github.com/jackkeane/medical-llm-pdq) | Pruning → distillation → quantization |

Alignment sits *between* SFT and compression on purpose: you align first, then compress,
so pruning/distillation preserve the aligned behavior.

## 📖 Visual walkthrough (start here)

Open **`docs/index.html`** in any browser — no server needed. Five self-contained pages, every
chart drawn from this repo's real runs, in the style of the
[finetuning walkthrough](https://github.com/jackkeane/medical-llm):

| Chapter | What it shows |
|---|---|
| [Overview](docs/index.html) | Where alignment sits in the lifecycle + the headline result |
| [1 · Setup](docs/01-setup.html) | The shared SFT starting point and the two kinds of signal |
| [2 · DPO](docs/02-dpo.html) | Preference pairs, and how DPO improves by pushing the *rejected* answer down |
| [3 · GRPO](docs/03-grpo.html) | The generate→score→update loop and the verifiable verdict reward |
| [4 · Comparison](docs/04-comparison.html) | Same 75%, different models: divergence, yes-bias drift, the `maybe` blind spot |

Static HTML/SVG, no dependencies, light + dark theme — also publishes via GitHub Pages
(Settings → Pages → deploy from `docs/`).

## Headline result

Both methods start from the **same merged SFT model** and train a fresh rank-16 LoRA
adapter. Evaluated on 100 held-out PubMedQA questions (greedy decode, verdict = yes/no/maybe):

| Model | Verdict accuracy | No-verdict rate | Reworded vs SFT | Verdict flips vs SFT |
|---|---|---|---|---|
| Base (BioMistral-7B) | 0% | 100% | — | — |
| SFT | 75% | 0% | — | — |
| DPO | 75% | 0% | 78 / 100 | 11 / 100 |
| GRPO | 75% | 0% | 40 / 100 | 2 / 100 |

The interesting part is **not** the identical 75%. It's that the two methods reach it
differently: **DPO** rewrote most answers and drifted toward a yes-bias, while **GRPO**
(KL-anchored) stayed close to SFT. And a shared blind spot — **every model scores 0/9 on
the `maybe` class**. Full analysis, per-class recall, training curves, and example outputs:

### 👉 [reports/comparison.md](reports/comparison.md)

## What each stage does

1. **Merge SFT → shared start policy.** `llamafactory-cli export` folds the Stage-1 adapter
   into the base weights (`models/sft-merged`). This is both the RL starting point and the
   frozen reference DPO measures preference against.
2. **DPO** (LLaMA-Factory, QLoRA). Preference pairs: *chosen* = curated PubMedQA answer,
   *rejected* = raw base-model generation for the same prompt (produced with vLLM). The
   model learns to prefer the verdict-first curated style over raw base output.
3. **GRPO** (TRL). The model generates its own answers; a **verifiable reward** scores each
   one — `+1` if its yes/no/maybe verdict matches the reference, `+0.2` for leading with a
   verdict at all. Trained on a class-balanced 240-prompt subset.
4. **Evaluate + compare.** Merge each adapter, generate on the test set with vLLM, score
   verdict accuracy / disclaimer rate / length, and analyze how the models differ.

## Environment

Two environments (the training and serving stacks don't co-exist cleanly):

- **`py312` conda env** — training: LLaMA-Factory, TRL 0.24, PEFT, bitsandbytes, torch 2.10.
  Prepend `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` so conda's `libstdc++` wins.
- **`~/.venv`** — inference: vLLM 0.15, used for base-answer generation and evaluation.

## Layout

```
medical-llm-RL/
├── configs/
│   ├── dataset_info.json        # registers the DPO preference set
│   ├── export_{sft,dpo,grpo}.yaml
│   └── dpo.yaml                 # DPO (stage: dpo) QLoRA config
├── scripts/
│   ├── gen_base_vllm.py         # base-model "rejected" answers (vLLM)
│   ├── build_dpo.py             # assemble preference pairs
│   ├── train_grpo.py            # TRL GRPO with verifiable reward
│   ├── eval_vllm.py             # score one variant on the test set
│   └── make_report.py           # consolidate everything → comparison.md
├── data/                        # preference pairs + base generations
├── reports/
│   ├── comparison.md            # the writeup
│   ├── metrics.json             # all numbers + training curves
│   └── eval/*.json              # per-question generations, all 4 variants
└── (models/, outputs/ are gitignored — regenerable)
```

## Reproduce

See the command block at the bottom of [reports/comparison.md](reports/comparison.md).

## Why compression wasn't re-run on the aligned models

The compression stage ([medical-llm-pdq](https://github.com/jackkeane/medical-llm-pdq)) ran on the
**SFT checkpoint**, and we deliberately did *not* re-run it on the DPO or GRPO models:

- Compression preserves what the input model does; it doesn't improve it. GRPO's model is nearly
  indistinguishable from SFT (identical correct-answer set, 2/100 verdict flips), so its
  compression run would reproduce the existing PDQ numbers within noise.
- DPO scores the same 75% but with a yes-bias — compressing it would only bake a known
  distribution drift into the deployment artifact.
- **The lesson:** don't re-run a pipeline whose input hasn't meaningfully changed. A re-run
  becomes worthwhile only after an alignment run that actually moves accuracy (e.g. wrong-verdict
  DPO rejections, or longer GRPO at a looser KL) — then "does compression preserve the alignment
  gain?" is a real question.

## Caveats

- Small by design: one 4090, 100-eval samples, a 240-prompt GRPO run. Treat trends as
  illustrative, not benchmark-grade. Longer GRPO at lower KL, or DPO pairs built from
  *wrong-verdict* rejections, would likely move the numbers.
- The finetuned model is an educational artifact; never use it for real medical decisions.

## License

MIT — see [LICENSE](LICENSE).
