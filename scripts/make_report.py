"""Consolidate eval + training curves into reports/comparison.md and metrics.json."""
import ast
import json
from collections import Counter
from pathlib import Path

RL = Path(__file__).resolve().parent.parent
EVAL = RL / "reports" / "eval"
VARIANTS = ["base", "sft", "dpo", "grpo"]
LABELS = {"base": "Base (BioMistral-7B)", "sft": "SFT", "dpo": "DPO", "grpo": "GRPO"}
CLASSES = ["yes", "no", "maybe"]


def load_eval():
    out = {}
    for v in VARIANTS:
        p = EVAL / f"{v}.json"
        if p.exists():
            out[v] = json.load(open(p))
    return out


def dpo_curve():
    st = json.load(open(RL / "outputs" / "dpo" / "trainer_state.json"))
    return [{"step": e["step"], "loss": e.get("loss"),
             "margin": e.get("rewards/margins"), "acc": e.get("rewards/accuracies")}
            for e in st["log_history"] if "loss" in e]


def grpo_curve():
    rows = []
    # progress-bar text prefixes each metric dict on the same line, and lines may
    # contain carriage returns, so locate the "{'loss'" dict substring explicitly.
    for raw in open(RL / "outputs" / "grpo_train.log"):
        for chunk in raw.split("\r"):
            idx = chunk.find("{'loss'")
            if idx == -1:
                continue
            end = chunk.rfind("}")
            if end <= idx:
                continue
            try:
                d = ast.literal_eval(chunk[idx:end + 1])
            except (ValueError, SyntaxError):
                continue
            if "reward" in d:  # skip the final train_runtime summary dict
                rows.append({"epoch": d.get("epoch"),
                             "reward": d.get("reward"),
                             "reward_correct": d.get("rewards/reward_correct/mean"),
                             "kl": d.get("kl")})
    return rows


def fmt_pct(x):
    return f"{x*100:.0f}%" if x is not None else "—"


def behavior(ev):
    """Per-question behavioural comparison vs the SFT model."""
    rows = {v: {r["instruction"]: r for r in ev[v]["rows"]} for v in ev}
    instrs = list(rows["sft"].keys())
    ref = {i: rows["sft"][i]["ref_verdict"] for i in instrs}
    V = {v: {i: rows[v][i]["gen_verdict"] for i in instrs} for v in ev}
    T = {v: {i: rows[v][i]["generation"] for i in instrs} for v in ev}
    b = {"pred_dist": {}, "vs_sft": {}, "recall": {}}
    for v in ev:
        b["pred_dist"][v] = dict(Counter(V[v][i] for i in instrs))
        b["recall"][v] = {
            c: [sum(1 for i in instrs if ref[i] == c and V[v][i] == c),
                sum(1 for i in instrs if ref[i] == c)]
            for c in CLASSES}
    for v in ev:
        if v == "sft":
            continue
        changed = sum(1 for i in instrs if V[v][i] != V["sft"][i])
        identical = sum(1 for i in instrs if T[v][i] == T["sft"][i])
        b["vs_sft"][v] = {"verdicts_changed": changed,
                          "answers_identical": identical,
                          "answers_rewritten": len(instrs) - identical}
    return b


def main():
    ev = load_eval()
    metrics = {v: ev[v]["summary"] for v in VARIANTS if v in ev}
    curves = {}
    try:
        curves["dpo"] = dpo_curve()
    except FileNotFoundError:
        pass
    try:
        curves["grpo"] = grpo_curve()
    except FileNotFoundError:
        pass
    beh = behavior(ev) if all(v in ev for v in VARIANTS) else None
    (RL / "reports" / "metrics.json").write_text(
        json.dumps({"eval": metrics, "curves": curves, "behavior": beh}, indent=2))

    # ---- markdown ----
    L = []
    L.append("# DPO vs GRPO — comparison on the medical SFT model\n")
    L.append("Both methods start from the **same merged SFT checkpoint** "
             "(`models/sft-merged`) and train a fresh rank-16 LoRA adapter on one "
             "RTX 4090. They differ in *what signal they optimize*.\n")

    L.append("## Headline metrics (100-sample held-out test set)\n")
    L.append("| Model | Verdict accuracy | Disclaimer rate | No-verdict rate | Mean length (chars) |")
    L.append("|---|---|---|---|---|")
    for v in VARIANTS:
        if v not in metrics:
            continue
        m = metrics[v]
        L.append(f"| {LABELS[v]} | {fmt_pct(m['verdict_accuracy'])} | "
                 f"{fmt_pct(m['disclaimer_rate'])} | {fmt_pct(m['verdict_none_rate'])} | "
                 f"{m['mean_length_chars']:.0f} |")
    L.append("")

    if "sft" in metrics:
        base_acc = metrics.get("base", {}).get("verdict_accuracy")
        sft_acc = metrics["sft"]["verdict_accuracy"]
        dpo_acc = metrics.get("dpo", {}).get("verdict_accuracy")
        grpo_acc = metrics.get("grpo", {}).get("verdict_accuracy")
        L.append("## What moved\n")
        if dpo_acc is not None:
            d = (dpo_acc - sft_acc) * 100
            L.append(f"- **DPO** shifted verdict accuracy by **{d:+.0f} pts** vs SFT "
                     f"({fmt_pct(sft_acc)} → {fmt_pct(dpo_acc)}). DPO optimizes a "
                     f"*preference* (curated answer ≻ raw base answer), so its main effect "
                     f"is on answer **style/consistency**, not necessarily verdict correctness.")
        if grpo_acc is not None:
            d = (grpo_acc - sft_acc) * 100
            L.append(f"- **GRPO** shifted verdict accuracy by **{d:+.0f} pts** vs SFT "
                     f"({fmt_pct(sft_acc)} → {fmt_pct(grpo_acc)}). GRPO optimizes a "
                     f"*verifiable reward* (does the verdict match?), so it targets "
                     f"**correctness** directly.")
        L.append("")

    # behavioural comparison — the real story behind an identical headline number
    if beh:
        L.append("## Same score, different models\n")
        L.append("All three post-SFT models score the same overall, so the averages hide "
                 "what each method actually did. Measured against the SFT model, question by "
                 "question:\n")
        L.append("| vs SFT | Verdicts changed /100 | Answers reworded /100 |")
        L.append("|---|---|---|")
        for v in ["dpo", "grpo"]:
            d = beh["vs_sft"][v]
            L.append(f"| {LABELS[v]} | {d['verdicts_changed']} | {d['answers_rewritten']} |")
        L.append("")
        L.append("- **GRPO stayed close to SFT.** Its KL penalty anchors the policy to the "
                 "starting model, so it reworded answers but rarely overturned a verdict.")
        L.append("- **DPO drifted further.** With no answer it was told to stay near, the "
                 "preference signal rewrote most answers and flipped more verdicts — gaining on "
                 "some classes, losing on others, for no net accuracy change.\n")

        L.append("### Predicted-verdict distribution (reference: 60 yes / 31 no / 9 maybe)\n")
        L.append("| Model | yes | no | maybe |")
        L.append("|---|---|---|---|")
        for v in ["sft", "dpo", "grpo"]:
            pd = beh["pred_dist"][v]
            L.append(f"| {LABELS[v]} | {pd.get('yes', 0)} | {pd.get('no', 0)} | {pd.get('maybe', 0)} |")
        L.append("")
        L.append("DPO's preference optimization pushed the model toward answering **yes** more "
                 "often (a common DPO distribution-shift effect); GRPO kept SFT's balance.\n")

        L.append("### Per-class recall (correct / total for each reference class)\n")
        L.append("| Model | yes | no | maybe |")
        L.append("|---|---|---|---|")
        for v in ["sft", "dpo", "grpo"]:
            r = beh["recall"][v]
            L.append(f"| {LABELS[v]} | {r['yes'][0]}/{r['yes'][1]} | "
                     f"{r['no'][0]}/{r['no'][1]} | {r['maybe'][0]}/{r['maybe'][1]} |")
        L.append("")
        L.append("> **The shared blind spot:** every model scores **0/9 on `maybe`**. It is only "
                 "9% of the training data, and none of SFT, DPO, or GRPO learned to produce it. "
                 "This — not the yes/no split — is where a targeted reward or rebalanced data "
                 "would help most.\n")

    # training curves
    if "dpo" in curves:
        c = curves["dpo"]
        L.append("## DPO training signal\n")
        L.append("Preference margin (reward for *chosen* − reward for *rejected*) and "
                 "implicit accuracy over training:\n")
        L.append("| Step | Loss | Reward margin | Pref. accuracy |")
        L.append("|---|---|---|---|")
        for e in c[:: max(1, len(c) // 8)]:
            L.append(f"| {e['step']} | {e['loss']:.3f} | {e['margin']:.2f} | {fmt_pct(e['acc'])} |")
        L.append("")
    if "grpo" in curves:
        c = curves["grpo"]
        L.append("## GRPO training signal\n")
        L.append(f"Per-step reward is noisy (only 4 prompts per optimizer step over {len(c)} "
                 "steps), so this shows the verdict-match reward averaged over consecutive "
                 "windows. It hovers near its starting level — consistent with GRPO staying "
                 "anchored to SFT rather than climbing:\n")
        nwin = 6
        win = max(1, len(c) // nwin)
        L.append("| Steps | Avg. verdict-correct reward | Avg. total reward | Avg. KL |")
        L.append("|---|---|---|---|")
        for s in range(0, len(c), win):
            grp = c[s:s + win]
            if not grp:
                continue
            rc = sum(e["reward_correct"] for e in grp) / len(grp)
            rt = sum(e["reward"] for e in grp) / len(grp)
            kl = sum(e["kl"] for e in grp) / len(grp)
            L.append(f"| {s + 1}–{s + len(grp)} | {rc:.2f} | {rt:.2f} | {kl:.4f} |")
        L.append("")

    # qualitative examples
    if all(v in ev for v in VARIANTS):
        by_instr = {}
        for v in VARIANTS:
            for r in ev[v]["rows"]:
                by_instr.setdefault(r["instruction"], {})[v] = r
        # pick 2 examples where models disagree
        picks = []
        for instr, d in by_instr.items():
            verds = {v: d[v]["gen_verdict"] for v in VARIANTS}
            if len({verds[v] for v in VARIANTS}) > 1:
                picks.append(instr)
            if len(picks) >= 2:
                break
        if picks:
            L.append("## Example outputs (where the models disagree)\n")
            for instr in picks:
                d = by_instr[instr]
                ref = d["sft"]["ref_verdict"]
                L.append(f"**Q:** {instr}  \n**Reference verdict:** `{ref}`\n")
                for v in VARIANTS:
                    g = d[v]["generation"].replace("\n", " ")[:180]
                    mark = "✓" if d[v]["correct"] else "✗"
                    L.append(f"- **{LABELS[v]}** `{d[v]['gen_verdict']}` {mark}: {g}")
                L.append("")

    L.append("## Takeaways\n")
    L.append("1. **SFT did the heavy lifting.** Base → SFT is 0% → 75% verdict accuracy; the "
             "base model never even emits a verdict. Alignment starts from a capable model.")
    L.append("2. **RL is not magic.** Neither DPO nor GRPO raised headline accuracy here — on a "
             "7B model with a small preference/reward budget, 75% is near what SFT already "
             "extracted. RL redistributes and stabilizes existing behavior; it does not inject "
             "knowledge the model lacks.")
    L.append("3. **The signal shapes the change.** DPO (learn a *preference*) rewrote answer "
             "style aggressively and shifted the output distribution; GRPO (optimize a "
             "*verifiable reward* under a KL leash) stayed anchored to SFT. Same score, "
             "different failure modes.")
    L.append("4. **Where to push next:** the `maybe` class (0/9) is the clearest target — more "
             "`maybe` data, a class-balanced reward, more GRPO steps at lower KL, or DPO pairs "
             "built from *wrong-verdict* rejections rather than raw-base rejections.\n")
    L.append("## Reproduce\n")
    L.append("```bash\n"
             "# 1. merge SFT adapter -> shared start policy (py312 / LLaMA-Factory)\n"
             "llamafactory-cli export configs/export_sft.yaml\n"
             "# 2. DPO preference data (rejected = base generations, vLLM) + train\n"
             "~/.venv/bin/python scripts/gen_base_vllm.py && python scripts/build_dpo.py\n"
             "llamafactory-cli train configs/dpo.yaml\n"
             "# 3. GRPO with verifiable verdict reward (TRL)\n"
             "python scripts/train_grpo.py\n"
             "# 4. merge both, evaluate 4 variants, build this report\n"
             "llamafactory-cli export configs/export_dpo.yaml\n"
             "llamafactory-cli export configs/export_grpo.yaml\n"
             "for n in base sft dpo grpo; do ~/.venv/bin/python scripts/eval_vllm.py "
             "--model models/$n-merged --name $n; done\n"
             "python scripts/make_report.py\n"
             "```\n")

    (RL / "reports" / "comparison.md").write_text("\n".join(L))
    print("wrote reports/comparison.md and reports/metrics.json")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
