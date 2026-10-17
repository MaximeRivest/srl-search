"""How good is a big whole-field map? A larger model (Kimi K3, the v5 teacher) judges stratified samples.

Precision: a random sample of the map's papers, and a sample of its well-cited papers (what reading lists show).
Recall: the candidates the map left out, sampled by score band (just under the cut-off, lower, lowest); each
band's share on topic times its size estimates the papers on topic that were missed. Papers that never became
candidates are not counted (the estimate is an upper bound on recall).

  python experiments/bigfields/check.py STORE KEY [--n 100] [--budget 1.0]
Writes STORE/KEY/check.json. Needs a Kimi K3 key (srl_search/judges.py, KIMI).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return round(m - c, 3), round(m + c, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("key")
    ap.add_argument("--n", type=int, default=100, help="papers per sample")
    ap.add_argument("--budget", type=float, default=1.0, help="USD at most")
    a = ap.parse_args()
    d = Path(a.store) / a.key
    sel = json.loads((d / "field.json").read_text())
    z = np.load(d / "work" / "scores.npz")
    ids, final, cut = z["ids"], z["final"], float(z["cut"])
    ctx = F.Ctx()
    cit = ctx.cit
    rng = np.random.default_rng(7)
    inmap = np.array(sorted(int(w) for w in sel["papers"]), np.int64)
    nodes = cit.nodes(inmap)
    cites = np.where(nodes >= 0, np.asarray(cit.cites)[np.maximum(nodes, 0)], 0)
    out_ = ~np.isin(ids, inmap)
    bands = [("just under the cut-off", (final >= max(0.0, cut - 0.25)) & (final < cut) & out_),
             ("lower", (final >= 0.1) & (final < max(0.1, cut - 0.25)) & out_),
             ("lowest", (final < 0.1) & out_)]
    samples = {"map, random": rng.choice(inmap, a.n, replace=False),
               "map, cited 50+ times": rng.choice(inmap[cites >= 50], min(a.n, int((cites >= 50).sum())), replace=False)}
    for name, m in bands:
        pool = ids[m]
        samples[f"left out, {name}"] = rng.choice(pool, min(a.n, len(pool)), replace=False) if len(pool) else np.zeros(0, np.int64)
    det = F._rows(ctx, np.concatenate(list(samples.values())), "title, abstract")

    def text(w):
        r = det.get(int(w))
        if r:
            return J.paper_text(r.get("title"), r.get("abstract"))
        nd = int(cit.nodes([int(w)])[0])
        return J.paper_text(cit.title(nd) if nd >= 0 else "", None)
    teacher = J.Teacher(a.budget)
    res = {"topic": sel["topic"], "papers": len(inmap), "cut": cut, "judge": "Kimi K3", "samples": {}}
    for name, ws in samples.items():
        lab = teacher.label([text(w) for w in ws], sel["topic"], sel.get("definition"))
        ok = ~np.isnan(lab)
        k, n = int((lab[ok] >= 0.5).sum()), int(ok.sum())
        examples = [{"id": f"W{int(w)}", "title": (det.get(int(w)) or {}).get("title"), "on_topic": bool(v >= 0.5)}
                    for w, v in zip(ws[ok][:12], lab[ok][:12])]
        res["samples"][name] = {"judged": n, "on_topic": k, "share": round(k / max(n, 1), 3), "ci95": wilson(k, n),
                                "examples": examples}
        print(f"{name:28s} {k}/{n} on topic ({100 * k / max(n, 1):.0f}%, 95% CI {wilson(k, n)})", flush=True)
    s = res["samples"]
    found = s["map, random"]["share"] * len(inmap)
    missed = sum(s[f"left out, {name}"]["share"] * int(m.sum()) for name, m in bands)
    res["estimate"] = {"precision": s["map, random"]["share"], "on_topic_in_map": round(found),
                       "on_topic_left_out": round(missed), "recall_of_candidates": round(found / max(found + missed, 1), 3),
                       "band_sizes": {name: int(m.sum()) for name, m in bands}, "usd": round(teacher.usd, 3)}
    print(json.dumps(res["estimate"]), flush=True)
    (d / "check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
