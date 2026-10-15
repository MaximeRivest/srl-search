"""Step 2: for each topic (topics_raw.json), the papers the teacher will judge, chosen as the whole-field pipeline
chooses its teacher's papers (srl_search/field.py, select_field stage 1): ~60 from the 300,000 closest in meaning,
spread from nearest to farthest over the same 16 buckets (where neighbouring subjects are), and ~30 works the 5,000
nearest papers cite (two thirds with a title only: the classics without abstracts). Junk records and non-publications
out, as there.

Leakage guards: near-duplicate topics (cosine > 0.9) are dropped; topics close to the two test topics (cosine > 0.55
to "habitat fragmentation" or to "LLM post-training: <definition>", listed in the output) are dropped; papers of the
test sets (Kimi yardsticks, meta-analyses' references, landmarks) are never sampled.

    ../../.venv/bin/python sample.py        (~30 min; writes samples.jsonl, resumable; topics.json)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import faiss
from srl_search import field as F, judges as J
from srl_search.engine import passes
from srl_search.scope import NON_PUBLICATION_TYPES

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
EXP = Path(__file__).resolve().parents[1] / "topicmap" / "teacher"
N_POOL, N_CITED, POOL = 60, 30, 300_000
EDGES = (.002, .004, .007, .01, .017, .025, .035, .05, .07, .1, .14, .2, .27, .4, .55, .75)

ctx = F.Ctx()
eng, cit = ctx.engine, ctx.cit
faiss.omp_set_num_threads(24)
bad_types = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]
rng = np.random.default_rng(2026)

# the test sets' papers: never in training
test_ids = set()
for f in ("eval_population.json", "eval_classics.json", "eval_population_llm_posttraining.json", "llm_posttraining_survey_refs.json"):
    test_ids |= {int(w) for w in json.loads((EXP / f).read_text())["ids"]}
for l in open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/refs_labelled.jsonl"):
    test_ids.add(int(json.loads(l)["id"][1:]))
for l in open(EXP / "round2" / "landmarks.tsv"):
    if l.startswith("W"):
        test_ids.add(int(l.split("\t")[0][1:]))
print("test papers excluded:", len(test_ids), flush=True)

raw = json.loads((D / "topics_raw.json").read_text())
for r in raw:
    if str(r.get("definition")).strip().lower() in ("", "none", "null"):
        r["definition"] = None
qtext = lambda t, d: f"{t}: {d}" if d else t
V = np.stack([eng.encoder.encode(qtext(r["topic"], r["definition"]))[0] for r in raw])
TESTS = [("habitat fragmentation", None), ("LLM post-training", "training large language models after pretraining: "
         "supervised fine-tuning and instruction tuning, reinforcement learning from human or AI feedback, preference "
         "optimization such as DPO, reinforcement learning for reasoning, and alignment training")]
TV = np.stack([eng.encoder.encode(qtext(t, d))[0] for t, d in TESTS])
near_test = (V @ TV.T).max(1)
keep, dropped = [], []
for i, r in enumerate(raw):
    if near_test[i] > 0.55:
        dropped.append((r["topic"], round(float(near_test[i]), 3), "close to a test topic"))
    elif keep and (V[keep] @ V[i]).max() > 0.9:
        dropped.append((r["topic"], round(float((V[keep] @ V[i]).max()), 3), "near-duplicate"))
    else:
        keep.append(i)
topics = [{**raw[i], "id": n} for n, i in enumerate(keep)]
(D / "topics.json").write_text(json.dumps({"topics": topics, "dropped": dropped}, indent=1, ensure_ascii=False))
print(len(topics), "topics kept;", len(dropped), "dropped:", dropped, flush=True)
print("closest kept to the tests:", sorted(((round(float(near_test[i]), 3), raw[i]["topic"]) for i in keep), reverse=True)[:8], flush=True)


def details(ws):
    return F._details(ctx, ws)


def text_and_ok(ws):
    det = details(ws)
    out = []
    for w in ws:
        d = det.get(int(w))
        if d:
            out.append((J.paper_text(d.get("title"), d.get("abstract")), passes(d, {}), True))
        else:
            nd = int(cit.nodes([int(w)])[0])
            t = cit.title(nd) if nd >= 0 else ""
            out.append((J.paper_text(t, None), nd >= 0 and int(cit.kind[nd]) not in bad_types and len(t) >= 10, False))
    return out


done = set()
out_path = D / "samples.jsonl"
if out_path.exists():
    done = {json.loads(l)["topic_id"] for l in open(out_path)}
fout = open(out_path, "a")
for tp, qv in zip(topics, V[keep]):
    if tp["id"] in done:
        continue
    t0 = time.time()
    dist, rows = eng.index.search(qv.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=512))
    rows = rows[0][rows[0] >= 0]                      # nearest first (inner product, descending)
    wa = np.asarray(eng.work_ids[rows], np.int64)
    n = len(wa)
    edges = sorted({0, *[int(n * f) for f in EDGES], n})
    per = math.ceil(N_POOL * 2 / (len(edges) - 1))   # oversampled: junk and non-publications go below
    pick, bucket = [], []
    for b, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if hi > lo:
            s = rng.choice(np.arange(lo, hi), size=min(per, hi - lo), replace=False)
            pick += s.tolist()
            bucket += [b] * len(s)
    pick, bucket = np.array(pick), np.array(bucket)
    keep_ = ~np.isin(wa[pick], list(test_ids))
    VA = F.unit(eng.vectors.get(rows[pick]))
    keep_ &= ctx.quality.junk(wa[pick], VA) < 0.5
    pick, bucket = pick[keep_], bucket[keep_]
    # an even share per bucket
    order = rng.permutation(len(pick))
    taken, per_b = [], {}
    for j in order:
        if per_b.get(bucket[j], 0) < math.ceil(N_POOL / (len(edges) - 1)):
            taken.append(j)
            per_b[bucket[j]] = per_b.get(bucket[j], 0) + 1
    pool_w = wa[pick[taken]][:N_POOL * 2]
    pool_b = bucket[taken][:N_POOL * 2]
    # works the 5,000 nearest cite (3+ times), not in the pool
    nodes = cit.nodes(np.unique(wa[:5000]))
    nodes = nodes[nodes >= 0]
    refs = np.concatenate([cit.refs(int(x)) for x in nodes]) if len(nodes) else np.zeros(0, np.uint32)
    u, c = np.unique(refs, return_counts=True)
    cn = u[c >= 3]
    cn = cn[~np.isin(np.asarray(cit.kind[cn]), bad_types)]
    cw = np.array([cit.work(int(x)) for x in cn], np.int64)
    cw = cw[~np.isin(cw, wa) & ~np.isin(cw, list(test_ids))]
    _, hv = F._vectors(ctx, cw) if len(cw) else (None, np.zeros(0, bool))
    t_only = rng.choice(cw[~hv], size=min(int((~hv).sum()), N_CITED), replace=False) if len(cw) else np.zeros(0, np.int64)
    w_vec = rng.choice(cw[hv], size=min(int(hv.sum()), N_CITED), replace=False) if len(cw) else np.zeros(0, np.int64)
    rows_out = []
    for kind, ws, bs in (("pool", pool_w, pool_b), ("cited_title_only", t_only, None), ("cited", w_vec, None)):
        lim = {"pool": N_POOL, "cited_title_only": round(N_CITED * 2 / 3), "cited": N_CITED - round(N_CITED * 2 / 3)}[kind]
        got = 0
        for k, (w, (text, ok, has_abs)) in enumerate(zip(ws, text_and_ok(ws))):
            if ok and got < lim:
                rows_out.append({"topic_id": tp["id"], "id": int(w), "kind": kind, "bucket": None if bs is None else int(bs[k]),
                                 "text": text})
                got += 1
    for r in rows_out:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout.flush()
    print(f"{tp['id']:3d} {tp['topic'][:50]:50s} {len(rows_out):3d} papers  {time.time() - t0:5.1f}s", flush=True)
print("SAMPLING DONE", flush=True)
