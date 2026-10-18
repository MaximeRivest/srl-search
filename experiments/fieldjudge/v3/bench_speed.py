"""Speed of the round-3 classifiers (fast path, vllm_hidden.py) on one RTX 3090 to themselves (GPU 0).

  quality   the 200 expert articles (176-option question), as jevarticle run srl3-<m>-abs-fast
  sweep     engine limits (max_num_seqs x max_num_batched_tokens, CUDA graphs on; plus eager) on 600 questions:
            throughput, and one question at a time (latency)
  yes/no    one topic question at a time (latency) and all at once

    SRL_FIELDS_GPU=0 ../../../.venv/bin/python bench_speed.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import judges as J
import eval3 as E, fmt

W = E.V3 / "bench-speed"; W.mkdir(exist_ok=True)
B = E.V3 / "bench-jev"
RES = Path.home() / "Projects/jevarticle/results"
MODELS = {"08": E.V3 / "models/0.8B/merged", "2b": E.V3 / "models/2B/merged"}
CONFIGS = [(16, 16384, False), (64, 32768, False), (256, 32768, False), (256, 65536, False), (256, 32768, True)]

pick600 = W / "pick600.jsonl"
pick600.write_text("".join(open(B / f"08-{n}.jsonl").read() for n in ("abs", "abs-perm0", "abs-perm1")))
yes = W / "yesno200.jsonl"
rows = []
for f in sorted((E.V2 / "benchmark").glob("*.json"))[:8]:
    for r in json.loads(f.read_text())["pairs"][:25]:
        rows.append({"user": fmt.yesno_prompt(r["text"], r["topic"], r["definition"]), "answers": ["yes", "no"]})
yes.write_text("".join(json.dumps(r) + "\n" for r in rows))


def call(m, src, dst, *extra):
    if not Path(dst).exists():
        J.run([E.VPY, str(HERE / "vllm_hidden.py"), str(MODELS[m]), str(src), str(dst), J.GPU, *map(str, extra)],
              W / f"{m}.log", env={"HF_HUB_OFFLINE": "1"})
    return json.loads(Path(dst).read_text())


for m in MODELS:
    P = call(m, B / f"{m}-abs.jsonl", W / f"{m}-abs-fast.json", "--repeat", 3)
    sp = json.loads((W / f"{m}-abs-fast.json.speed.json").read_text())
    d = RES / f"srl3-{m}-abs-fast"; d.mkdir(exist_ok=True)
    with open(d / "responses.jsonl", "w") as f:
        for a, p in zip(E.arts, P):
            p = np.array(p); j = int(p.argmax())
            f.write(json.dumps({"scopus_id": a["scopus_id"], "subset": a["subset"], "has_abstract": bool(a["abstract"]),
                                "state": f"Title: {a['title']}", "prediction": E.subs[j], "confidence": round(float(p[j]), 4)}) + "\n")
    json.dump({"run": f"srl3-{m}-abs-fast", "model": f"fieldjudge-v3 {'0.8B' if m == '08' else '2B'} (ours), fast path",
               "design": "llm", "mode": "one-token", "input": "abstract", "workers": "batched", "n_articles": 200,
               "wall_clock_s": round(sp["seconds"][-1], 2), "latency_median_s": "n/a", "latency_p95_s": "n/a",
               "input_tokens": sp["prompt_tokens"], "output_tokens": 0, "cost_usd": None, "questions_per_request": 1},
              open(d / "meta.json", "w"), indent=1)
    for seqs, toks, eager in CONFIGS:
        call(m, pick600, W / f"{m}-sweep-{seqs}-{toks}{'-eager' if eager else ''}.json", "--sweep", "--seqs", seqs,
             "--tokens", toks, *(["--eager"] if eager else []))
    call(m, yes, W / f"{m}-yesno-sweep.json", "--sweep")
    print("done", m, flush=True)
print("SPEED DONE", flush=True)
