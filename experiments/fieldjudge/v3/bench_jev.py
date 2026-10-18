"""Our round-3 classifiers on jevarticle's 200-article expert benchmark, written as jevarticle runs
(results/<run>/responses.jsonl + meta.json) so jevarticle/scripts/score.py scores them next to Jev and Qwen 27B.

Runs per model: abs (title, venue, year, abstract; one question, 176 options), title (title, venue, year only),
abs-3orders (the abs question asked in 3 option orders, probabilities averaged). Speed: steady state of 3 repeats.

    SRL_FIELDS_GPU=0 ../../../.venv/bin/python bench_jev.py
"""
import json, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import judges as J
import fmt, eval3 as E

RES = Path.home() / "Projects/jevarticle/results"
W = E.V3 / "bench-jev"; W.mkdir(exist_ok=True)
MODELS = {"08": E.V3 / "models/0.8B/merged", "2b": E.V3 / "models/2B/merged"}
NAMES = {"08": "fieldjudge-v3 0.8B (ours)", "2b": "fieldjudge-v3 2B (ours)"}
Q = "Which subfield best describes the research content of this paper?"
opts = [{"name": t["subfield"], "parent": t["field"]} for t in E.tax]


def text(a, title_only):
    s = f"Title: {a['title']}\nVenue: {a['journal']}\nYear: {a['year']}"
    return s + (f"\nAbstract: {a['abstract'][:1500]}" if a["abstract"] and not title_only else "\n(no abstract)")


def rows(title_only, perm_seed=None):
    out = []
    for i, a in enumerate(E.arts):
        perm = list(range(len(opts)))
        if perm_seed is not None:
            random.Random(perm_seed * 1000 + i).shuffle(perm)
        codes = fmt.assign_codes(len(opts))
        out.append({"user": fmt.pick_prompt(text(a, title_only), E.SM, Q, [opts[j] for j in perm], codes),
                    "answers": codes, "perm": perm})
    return out


def score(m, name, rs, repeat):
    src, dst = W / f"{m}-{name}.jsonl", W / f"{m}-{name}.json"
    if not dst.exists():
        src.write_text("".join(json.dumps({"user": r["user"], "answers": r["answers"]}) + "\n" for r in rs))
        J.run([E.VPY, str(HERE / "vllm_classify.py"), str(MODELS[m]), str(src), str(dst), J.GPU, "--repeat", str(repeat)],
              W / f"{m}.log", env={"HF_HUB_OFFLINE": "1"})
    P = []
    for r, p in zip(rs, json.loads(dst.read_text())):
        q = np.zeros(len(opts)); q[np.array(r["perm"])] = p; P.append(q)
    return np.array(P), json.loads(Path(str(dst) + ".speed.json").read_text())


def write(run, m, P, speed_list, input_kind, questions):
    d = RES / run; d.mkdir(parents=True, exist_ok=True)
    with open(d / "responses.jsonl", "w") as f:
        for a, p in zip(E.arts, P):
            j = int(p.argmax())
            f.write(json.dumps({"scopus_id": a["scopus_id"], "subset": a["subset"], "has_abstract": bool(a["abstract"]),
                                "state": text(a, input_kind == "title"),
                                "prediction": E.subs[j], "confidence": round(float(p[j]), 4),
                                "top5": [[E.subs[k], round(float(p[k]), 4)] for k in np.argsort(-p)[:5]]}) + "\n")
    secs = sum(s["seconds"][-1] for s in speed_list); toks = sum(s["prompt_tokens"] for s in speed_list)
    json.dump({"run": run, "model": NAMES[m], "design": "llm", "mode": "one-token", "input": input_kind,
               "workers": "batched", "n_articles": len(E.arts), "wall_clock_s": round(secs, 2),
               "latency_median_s": "n/a", "latency_p95_s": "n/a", "input_tokens": toks, "output_tokens": 0,
               "cost_usd": None, "questions_per_request": questions,
               "cost_note": "local vLLM 0.29 on one RTX 3090 to itself, eager mode, prefix cache; steady state of 3 repeats",
               "seconds_per_article": round(secs / len(E.arts), 4)}, open(d / "meta.json", "w"), indent=1)


for m in MODELS:
    Pa, sa = score(m, "abs", rows(False), 3)
    Pt, st = score(m, "title", rows(True), 3)
    Ps = [score(m, f"abs-perm{k}", rows(False, k + 1), 3) for k in range(2)]
    write(f"srl3-{m}-abs", m, Pa, [sa], "abstract", 1)
    write(f"srl3-{m}-title", m, Pt, [st], "title", 1)
    write(f"srl3-{m}-abs-3orders", m, (Pa + Ps[0][0] + Ps[1][0]) / 3, [sa, Ps[0][1], Ps[1][1]], "abstract", 3)
    print("done", m, flush=True)
print("BENCH DONE", flush=True)
