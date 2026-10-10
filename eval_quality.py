"""Does search return junk? A fixed question set, results judged by an independent model, compared per system.

    python eval_quality.py run    --name baseline [--api URL] [--param k=v ...]   # top 100 per question
    python eval_quality.py judge  NAME [NAME ...]                                  # judge what is not judged yet
    python eval_quality.py report NAME [NAME ...]                                  # metrics, paired

Questions: experiments/quality/queries.jsonl (60: everyday advice, curiosity, research phrasing, terse
keywords), each with the information need the judge reads. Runs: experiments/quality/runs/NAME.json.
Judgments: experiments/quality/judgments.jsonl, one per (question, paper), reused by every run, so a new
system only costs the papers nobody has judged yet (pooling, as in TREC).

The judge sees what a reader sees (title, type, year, venue, the start of the abstract), never citation
counts or venue flags: those are what the rankers use, and a judge that saw them would grade the rankers'
own inputs. It gives each paper:
  rel   0 off-topic · 1 same area, does not answer · 2 partly answers · 3 directly answers the need
  kind  research (a real scholarly work, even a weak one) · non_research (real but not scholarship: magazine
        piece, handout, newsletter, front matter, chapter fragment, class assignment) · spam (advertising,
        product "reviews", SEO/clickbait, filler posted to look like research)

Metrics, over the top 10 (and 20) a reader sees:
  spam@10      share of results that are spam
  junk@10      share that are spam or non_research
  useful nDCG@10  graded relevance, counting only research; ideal ranking from every judged paper
  useful@10    results with rel >= 2 that are research
Paired differences against the first run named, with a bootstrap 95% interval over questions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent / "experiments" / "quality"
QUERIES = HERE / "queries.jsonl"
RUNS = Path(os.environ.get("SRL_QUALITY_RUNS", "/mnt/fast/scholarsreadinglist/search/quality-eval/runs"))  # ~20 MB per run (abstracts)
JUDGMENTS = HERE / "judgments.jsonl"
JUDGE_MODEL = "sonnet"
PROMPT_VERSION = "v1"

PROMPT = """You grade results of a search engine over scholarly literature, used by the general public.
The engine should return genuine scholarly works that answer the searcher's need.

Search: {q}
Need: {need}

For EACH numbered result below, judge only from what is shown (do not reward fame or popularity):
- rel: 0 = off-topic; 1 = same general area but does not address the need; 2 = partly addresses the need;
  3 = directly addresses the need.
- kind: "research" = a real scholarly work (study, review, thesis, scholarly book or chapter, report of a
  credible body, conference paper), even if weak or small; "non_research" = a real document but not
  scholarship (magazine or trade article, patient handout, newsletter, editorial note without content,
  front/back matter, a chapter fragment like "Final thoughts", course notes, a class assignment);
  "spam" = advertising, product or supplement "reviews", affiliate or SEO content, clickbait listicles,
  filler text made to look like research.

Answer with ONLY a JSON array, one object per result, in order: [{{"n": 1, "rel": 0, "kind": "research"}}, ...]

Results:
{items}"""


def queries() -> list[dict]:
    """The evaluation questions, or with SRL_QSET=train the disjoint set used to fit rankers."""
    f = HERE / "train_queries.jsonl" if os.environ.get("SRL_QSET") == "train" else QUERIES
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def run(name: str, api: str, params: list[str], rewrites: Path | None = None, field: str = "rewrite") -> None:
    """Top 100 per question; with --rewrites, search the rewritten text (offline experiments)."""
    extra = dict(p.split("=", 1) for p in params)
    out = {"api": api, "params": extra, "results": {}}
    rw = json.loads(rewrites.read_text()) if rewrites else None
    if rw:
        out["rewrites"] = {"file": str(rewrites), "field": field}

    def one(q):
        text = rw[q["qid"]][field][:1900] if rw else q["q"]
        args = {"q": text, "k": "100", **extra}
        with urllib.request.urlopen(f"{api}/search?{urllib.parse.urlencode(args)}", timeout=120) as r:
            d = json.load(r)
        keep = ("id", "title", "type", "year", "venue", "abstract", "score", "citations")
        return q["qid"], [{k: x.get(k) for k in keep} for x in d["results"]], d.get("timing", {}), d.get("rewritten")

    with ThreadPoolExecutor(4) as ex:
        for qid, res, timing, rew in ex.map(one, queries()):
            out["results"][qid] = res
            out.setdefault("timing", {})[qid] = timing
            if rew:
                out.setdefault("rewritten", {})[qid] = rew
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: {len(out['results'])} questions", file=sys.stderr)


def fuse(names: list[str], out_name: str, rrf_k: int = 60) -> None:
    """Reciprocal-rank fusion: a paper's score is the sum of 1/(rrf_k + rank) over the runs."""
    runs = [json.loads((RUNS / f"{n}.json").read_text())["results"] for n in names]
    out = {"fused": names, "rrf_k": rrf_k, "results": {}}
    for qid in runs[0]:
        score, doc = {}, {}
        for r in runs:
            for rank, x in enumerate(r.get(qid, [])):
                score[x["id"]] = score.get(x["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
                doc.setdefault(x["id"], x)
        out["results"][qid] = [doc[i] for i in sorted(score, key=lambda i: -score[i])][:100]
    (RUNS / f"{out_name}.json").write_text(json.dumps(out))


def rerank(name: str, out_name: str, alpha: float, beta: float, quality: Path, hard: float | None) -> None:
    """Offline: re-order a run's candidates by score - alpha * P(junk) + beta * log1p(citations)/10
    (or drop P(junk) > hard). Same arithmetic as the service, on the run's top 100."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fit_quality import vectors_of
    from srl_search.quality import Quality
    import numpy as np
    q = Quality(quality)
    src = json.loads((RUNS / f"{name}.json").read_text())
    out = {"reranked": name, "alpha": alpha, "beta": beta, "hard": hard, "results": {}}
    for qid, res in src["results"].items():
        ws = np.array([int(x["id"][1:]) for x in res], dtype=np.int64)
        vec, ok = vectors_of(ws)
        full = np.zeros((len(ws), vec.shape[1] if len(vec) else 768), np.float32)
        full[ok] = vec
        p = q.junk(ws, full)
        if (~ok).any():
            p[~ok] = q.junk(ws[~ok])
        s = np.array([x.get("score") or 0 for x in res]) - alpha * p
        s = s + beta * np.log1p([x.get("citations") or 0 for x in res]) / 10
        order = [i for i in np.argsort(-s, kind="stable") if hard is None or p[i] <= hard]
        out["results"][qid] = [dict(res[i], junk=round(float(p[i]), 3)) for i in order]
    (RUNS / f"{out_name}.json").write_text(json.dumps(out))


def load_judgments() -> dict:
    j = {}
    if JUDGMENTS.exists():
        for l in JUDGMENTS.read_text().splitlines():
            d = json.loads(l)
            j[(d["qid"], d["id"])] = d
    return j


def show(i: int, x: dict) -> str:
    abstract = " ".join((x.get("abstract") or "").split())[:900]
    return (f"[{i}] {(x.get('title') or '').strip()}\n    type: {x.get('type')} · year: {x.get('year')} · "
            f"venue: {x.get('venue') or '-'}\n    abstract: {abstract or '(none)'}")


def ask(prompt: str) -> str:
    r = subprocess.run(["claude", "-p", "--model", JUDGE_MODEL, "--tools", "", "--no-session-persistence"],
                       input=prompt, capture_output=True, text=True, timeout=600, cwd="/tmp")  # no project context
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return r.stdout


def judge(names: list[str], depth: int, batch: int = 25) -> None:
    done = load_judgments()
    qs = {q["qid"]: q for q in queries()}
    todo: dict[str, dict[str, dict]] = {}
    for n in names:
        for qid, res in json.loads((RUNS / f"{n}.json").read_text())["results"].items():
            for x in res[:depth]:
                if (qid, x["id"]) not in done:
                    todo.setdefault(qid, {})[x["id"]] = x
    jobs = []
    for qid, docs in todo.items():
        docs = list(docs.values())
        for s in range(0, len(docs), batch):
            jobs.append((qid, docs[s:s + batch]))
    print(f"judging {sum(len(d) for _, d in jobs)} papers in {len(jobs)} calls", file=sys.stderr)

    def one(job):
        qid, docs = job
        q = qs[qid]
        prompt = PROMPT.format(q=q["q"], need=q["need"], items="\n".join(show(i + 1, x) for i, x in enumerate(docs)))
        for attempt in range(3):
            try:
                txt = ask(prompt)
                arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
                by_n = {int(a["n"]): a for a in arr}
                if set(by_n) != set(range(1, len(docs) + 1)):
                    raise ValueError("wrong count")
                return [{"qid": qid, "id": x["id"], "rel": int(by_n[i + 1]["rel"]), "kind": by_n[i + 1]["kind"],
                         "judge": JUDGE_MODEL, "prompt": PROMPT_VERSION} for i, x in enumerate(docs)]
            except Exception as e:  # noqa: BLE001 — retried, then reported
                err = e
        print(f"  {qid}: gave up ({err})", file=sys.stderr)
        return []

    with ThreadPoolExecutor(6) as ex, JUDGMENTS.open("a") as f:
        for rows in ex.map(one, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()


def metrics(res: list[dict], qid: str, j: dict, ideal: list[int], k: int) -> dict | None:
    top = res[:k]
    js = [j.get((qid, x["id"])) for x in top]
    if not top or any(x is None for x in js):
        return None
    gain = [x["rel"] if x["kind"] == "research" else 0 for x in js]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gain))
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return {"spam": sum(x["kind"] == "spam" for x in js) / len(js),
            "junk": sum(x["kind"] != "research" for x in js) / len(js),
            "ndcg": dcg / idcg if idcg else 0.0,
            "useful": sum(1 for x in js if x["kind"] == "research" and x["rel"] >= 2)}


def report(names: list[str], k: int = 10) -> None:
    j = load_judgments()
    qs = queries()
    ideal = {}
    for q in qs:
        g = sorted((d["rel"] if d["kind"] == "research" else 0 for (qid, _), d in j.items() if qid == q["qid"]),
                   reverse=True)
        ideal[q["qid"]] = g
    runs = {n: json.loads((RUNS / f"{n}.json").read_text())["results"] for n in names}
    per = {n: {q["qid"]: metrics(runs[n].get(q["qid"], []), q["qid"], j, ideal[q["qid"]], k) for q in qs} for n in names}
    styles = ["all"] + sorted({q["style"] for q in qs})
    print(f"top {k}; each cell: spam / junk (spam + non-research) / useful nDCG / useful results")
    print(f"{'run':28s}" + "".join(f"{s:>26s}" for s in styles))
    for n in names:
        row = f"{n:28s}"
        for s in styles:
            ms = [per[n][q["qid"]] for q in qs if s in ("all", q["style"])]
            if any(m is None for m in ms):
                row += f"{'(not judged)':>26s}"
                continue
            a = lambda key: sum(m[key] for m in ms) / len(ms)
            row += f"{a('spam'):>7.0%} {a('junk'):>5.0%} {a('ndcg'):>5.3f} {a('useful'):>5.1f}"
        print(row)
    base = names[0]
    rng = random.Random(0)
    for n in names[1:]:
        ok = [q["qid"] for q in qs if per[n][q["qid"]] and per[base][q["qid"]]]
        for key in ("junk", "ndcg"):
            d = [per[n][x][key] - per[base][x][key] for x in ok]
            boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
            better = sum(v < 0 if key == "junk" else v > 0 for v in d)
            worse = sum(v > 0 if key == "junk" else v < 0 for v in d)
            print(f"  {n} vs {base}: {key} {sum(d)/len(d):+.3f} (95% {boots[50]:+.3f} … {boots[1949]:+.3f}); "
                  f"better on {better}, worse on {worse} of {len(d)} questions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--name", required=True)
    r.add_argument("--api", default="http://127.0.0.1:8810"); r.add_argument("--param", action="append", default=[])
    r.add_argument("--rewrites", type=Path); r.add_argument("--field", default="rewrite")
    rr = sub.add_parser("rerank"); rr.add_argument("name"); rr.add_argument("--out", required=True)
    rr.add_argument("--alpha", type=float, default=0.1); rr.add_argument("--beta", type=float, default=0.0)
    rr.add_argument("--hard", type=float); rr.add_argument("--quality", type=Path,
                    default=Path("/mnt/fast/scholarsreadinglist/search/quality-2026-10-05"))
    fu = sub.add_parser("fuse", help="reciprocal-rank fusion of runs"); fu.add_argument("names", nargs="+")
    fu.add_argument("--out", required=True); fu.add_argument("--rrf-k", type=int, default=60)
    jd = sub.add_parser("judge"); jd.add_argument("names", nargs="+"); jd.add_argument("--depth", type=int, default=20)
    rp = sub.add_parser("report"); rp.add_argument("names", nargs="+"); rp.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.name, a.api, a.param, a.rewrites, a.field)
    elif a.cmd == "rerank":
        rerank(a.name, a.out, a.alpha, a.beta, a.quality, a.hard)
    elif a.cmd == "fuse":
        fuse(a.names, a.out, a.rrf_k)
    elif a.cmd == "judge":
        judge(a.names, a.depth)
    else:
        report(a.names, a.k)
