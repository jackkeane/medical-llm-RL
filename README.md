# Medical LLM — RL / Preference Alignment

Stage 2 of an educational LLM lifecycle series: teach the finetuned medical model *which* answers
are preferred, not just how to format them.

⚠️ **DISCLAIMER:** For educational/research purposes only. NOT for clinical use.

## Where this fits in the series

| Stage | Repo | What it teaches |
|---|---|---|
| 1 · Supervised finetuning | [medical-llm](https://github.com/jackkeane/medical-llm) | QLoRA SFT of BioMistral-7B, visualized end to end |
| **2 · Preference alignment (RL)** | **this repo** | DPO / GRPO on the SFT checkpoint |
| 3 · Compression | [medical-llm-pdq](https://github.com/jackkeane/medical-llm-pdq) | Pruning → distillation → quantization |

Alignment sits *between* SFT and compression on purpose: you align first, then compress, so
pruning/distillation preserve the aligned behavior.

## Planned approach

- **DPO (Direct Preference Optimization)** via LLaMA-Factory — same tooling as stage 1, similar
  VRAM footprint to the QLoRA run. Preference pairs built from this series' own artifacts:
  *chosen* = reference PubMedQA answer, *rejected* = base-model answer.
- **GRPO (optional follow-up)** — PubMedQA's yes/no/maybe labels give a verifiable exact-match
  reward, making it a clean small-scale reasoning-RL exercise.
- Every run's logs and metrics become material for a visual walkthrough, in the style of
  [medical-llm/docs](https://github.com/jackkeane/medical-llm).

## Status

🚧 Scaffold — no training runs yet.

## License

MIT — see [LICENSE](LICENSE).
