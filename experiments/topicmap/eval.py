"""Which way of linking papers gives the best topic-map groups? Measured on 30 questions from many fields.

    search/.venv/bin/python search-dev/experiments/topicmap/eval.py [--questions N] [--out eval.json]

Same papers per question (the service's own selection: the 1,000 closest in meaning), several linkings:
  hybrid      shared references (idf-weighted cosine) + text neighbours, half the weight each   (the default)
  hybrid-noidf  the same with Salton's cosine (every reference counts the same)
  hybrid-top30  hybrid, keeping each paper's 30 strongest reference links (faster)
  refs        shared references only
  text        text neighbours only

Accuracy, from facts the grouping never sees (as in Boyack & Klavans 2010):
  mesh_lift   pairs in the same group share MeSH headings (assigned by NLM indexers from the full text) how many
              times more than random pairs (mean Jaccard of the heading sets). Biomedical questions only.
  venue_lift  pairs in the same group were published in the same journal how many times more than random pairs.
Also: groups, share of papers in groups, agreement of the groups made with another random seed (ARI), time.
Lifts depend on group sizes (smaller groups are easier to make coherent), so group counts are reported next to them.

Loads the release's search index read-only in its own process (~10 GB); changes nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import igraph
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from srl_search import topics as T  # noqa: E402
from srl_search.citations import Citations  # noqa: E402
from srl_search.engine import Engine  # noqa: E402
from srl_search.quality import Quality  # noqa: E402
from srl_search.topic_service import TopicService  # noqa: E402

QUESTIONS = [  # (question, biomedical?)
    ("intermittent fasting for weight loss", True), ("coral reef bleaching and recovery", False),
    ("graph neural networks for molecular property prediction", False), ("effects of remote work on productivity", False),
    ("gut microbiome and depression", True), ("CRISPR base editing for genetic diseases", True),
    ("microplastics in freshwater ecosystems", False), ("perovskite solar cell stability", False),
    ("sleep deprivation and memory consolidation", True), ("antibiotic resistance in hospital infections", True),
    ("social media use and adolescent mental health", True), ("large language models for code generation", False),
    ("urban heat islands and green infrastructure", False), ("minimum wage effects on employment", False),
    ("long COVID symptoms and mechanisms", True), ("pollinator decline and pesticides", False),
    ("dark matter direct detection experiments", False), ("immunotherapy for melanoma", True),
    ("bilingualism and cognitive control", True), ("blockchain in supply chain management", False),
    ("exercise as treatment for depression", True), ("glacier retreat and sea level rise", False),
    ("misinformation on social media", False), ("Alzheimer's disease amyloid hypothesis", True),
    ("soil carbon sequestration in agriculture", False), ("autism early diagnosis", True),
    ("medieval monasteries and literacy", False), ("lithium-ion battery degradation", False),
    ("air pollution and cardiovascular mortality", True), ("teacher feedback and student achievement", False),
]
VARIANTS = {
    "hybrid": T.Options(),
    "hybrid-noidf": T.Options(idf=False),
    "hybrid-top30": T.Options(refs_top=30),
    "refs": T.Options(links="refs"),
    "text": T.Options(links="text"),
}


def lift(groups: np.ndarray, labels: list, sim, rng, pairs=3000) -> float | None:
    """Mean similarity of same-group pairs / mean similarity of random pairs (papers with a label only)."""
    ok = np.array([bool(x) for x in labels])
    if ok.sum() < 50:
        return None
    idx = np.flatnonzero(ok & (groups > 0))
    a, b = rng.choice(np.flatnonzero(ok), pairs), rng.choice(np.flatnonzero(ok), pairs)
    m = a != b
    rand = np.mean([sim(labels[i], labels[j]) for i, j in zip(a[m], b[m])])
    same = []
    for g in set(groups[idx].tolist()):
        mem = idx[groups[idx] == g]
        if len(mem) < 2:
            continue
        k = max(20, int(pairs * len(mem) / len(idx)))
        x, y = rng.choice(mem, k), rng.choice(mem, k)
        m = x != y
        same += [sim(labels[i], labels[j]) for i, j in zip(x[m], y[m])]
    return round(float(np.mean(same) / rand), 3) if rand > 0 and same else None


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=len(QUESTIONS))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "eval.json")
    args = ap.parse_args()
    rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
    P = {k: Path(v) for k, v in rel["paths"].items()}
    t = time.perf_counter()
    quality = Quality(P["corpus"].parent / "quality-2026-10-05")
    eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=8, quality=quality)
    cit = Citations(P["citations"])
    api = SimpleNamespace(engine=eng, KEYWORDS=None, RELEASE={"id": "eval"}, ATLAS=None,
                          looks_boolean=lambda q: False, readable=lambda w: None, place_work=lambda w: None)
    svc = TopicService(api, Path("/tmp/topicmap-eval-cache"), cit)
    print(f"loaded in {time.perf_counter() - t:.0f}s; authors index: {cit.authors is not None}", flush=True)
    rows = []
    for q, bio in QUESTIONS[:args.questions]:
        t = time.perf_counter()
        seed, _ = svc.seed_semantic(q, 1000, {})
        t_seed = time.perf_counter() - t
        mesh = [set(x.strip() for x in (d.get("mesh") or "").split("|") if x.strip()) for d in seed.details]
        venue = [(d.get("venue") or "").strip().lower() for d in seed.details]
        row = {"question": q, "biomedical": bio, "papers": len(seed.wids), "seed_s": round(t_seed, 1),
               "with_mesh": sum(bool(m) for m in mesh), "variants": {}}
        for name, opt in VARIANTS.items():
            rng = np.random.default_rng(0)
            t = time.perf_counter()
            out = T.build(seed, cit, q, opt)
            secs = time.perf_counter() - t
            g = np.array(out["papers"]["group"])
            alt = T.build(seed, cit, q, T.Options(**{**opt.__dict__, "seed": opt.seed + 1}))
            ari = igraph.compare_communities(g.tolist(), alt["papers"]["group"], method="adjusted_rand")
            row["variants"][name] = {
                "groups": len(out["groups"]), "in_groups": round(float((g > 0).mean()), 3),
                "resolution": out["method"].get("resolution"),
                "mesh_lift": lift(g, mesh, jaccard, rng) if bio else None,
                "venue_lift": lift(g, venue, lambda a, b: float(a == b), rng),
                "seed_ari": round(float(ari), 3), "seconds": round(secs, 2),
                "names": [x["name"] for x in out["groups"]],
                "timing": out["timing"]}
            v = row["variants"][name]
            print(f"{q[:45]:45} {name:13} groups {v['groups']:2} in {v['in_groups']:.2f} mesh {v['mesh_lift']} "
                  f"venue {v['venue_lift']} ari {v['seed_ari']} {v['seconds']}s", flush=True)
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    # summary: mean of each measure per variant, and how often each beats the default on the same question
    summary = {}
    for name in VARIANTS:
        vals = lambda k: [r["variants"][name][k] for r in rows if r["variants"][name][k] is not None]
        summary[name] = {k: round(float(np.mean(vals(k))), 3) for k in ("groups", "in_groups", "mesh_lift", "venue_lift", "seed_ari", "seconds")}
    for name in VARIANTS:
        for k in ("mesh_lift", "venue_lift", "seed_ari"):
            d = [r["variants"][name][k] - r["variants"]["hybrid"][k] for r in rows
                 if r["variants"][name][k] is not None and r["variants"]["hybrid"][k] is not None]
            if d and name != "hybrid":
                boot = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
                summary[name][f"{k}_vs_hybrid"] = [round(float(np.mean(d)), 3), round(float(np.percentile(boot, 2.5)), 3),
                                                   round(float(np.percentile(boot, 97.5)), 3)]
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "questions": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
