"""Table and chart of curve_results.jsonl: quality vs teacher labels, per model; the staircase beside it."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "curve_results.jsonl")]
g = defaultdict(list)
for r in rows:
    g[(r["model"], r["tag"] or "direct", r["n"])].append(r)
print(f"{'model':18} {'how':10} {'N':>6} {'runs':>4} | {'test AUC':>8} {'recall':>6} {'prec':>5} | {'refs AUC':>8} {'recall':>6} {'title':>6} | speed/s  train s")
summary = []
for (m, how, n), rs in sorted(g.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    f = lambda k, s: np.mean([r[s][k] for r in rs if r[s].get(k) is not None]) if any(r[s].get(k) is not None for r in rs) else float("nan")
    sd = np.std([r["test"]["auc"] for r in rs]) if len(rs) > 1 else 0
    line = dict(model=m, how=how, n=n, runs=len(rs), test_auc=f("auc", "test"), test_auc_sd=sd, test_recall=f("recall", "test"),
                test_precision=f("precision", "test"), refs_auc=f("auc", "refs"), refs_recall=f("recall", "refs"),
                refs_title=f("recall_title_only", "refs"), speed=np.mean([r.get("papers_per_s", 0) for r in rs]),
                train_s=np.mean([r.get("train_s", 0) for r in rs]))
    summary.append(line)
    print(f"{m:18} {how:10} {n:6} {len(rs):4} | {line['test_auc']:8.3f} {line['test_recall']:6.3f} {line['test_precision']:5.3f} | "
          f"{line['refs_auc']:8.3f} {line['refs_recall']:6.3f} {line['refs_title']:6.3f} | {line['speed']:7.0f} {line['train_s']:7.0f}")
(HERE / "curve_summary.json").write_text(json.dumps(summary, indent=1, default=float))
if "--chart" in sys.argv:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, (metric, title) in enumerate((("test_auc", "Held-out papers (Kimi's labels): AUC"), ("refs_recall", "Meta-analyses' studies found (recall at 0.5)"))):
        for m in sorted({s["model"] for s in summary}):
            for how, style in (("direct", "-o"), ("staircase", "--s")):
                pts = sorted((s["n"], s[metric]) for s in summary if s["model"] == m and s["how"] == how)
                if pts:
                    ax[k].plot(*zip(*pts), style, label=f"{m.replace('ettin-encoder-', 'Ettin ')}{' (staircase)' if how == 'staircase' else ''}")
        ax[k].set_xscale("log"); ax[k].set_xlabel("teacher (Kimi K3) labels"); ax[k].set_title(title, fontsize=11); ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "curve.png", dpi=130)
    print("chart:", HERE / "curve.png")
