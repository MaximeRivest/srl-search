"""Step 7: the general judge inside the real pipeline. The two test topics' whole-field selections are replayed
(srl_search.field.select_field, today's code) with the run's own cached Kimi answers; only the judge that labels
10,000 papers for the students changes:
  A per-topic  the run's own decider (its cached scores: the v5 runs)
  B general-2B / C general-0.8B  the general judge, the topic's question in each prompt, no per-topic training
Students, citation combiner, cut-off and Jev's check run for real in every arm (Jev ~$0.5 per fragmentation arm).
Measured against Kimi: tests.jsonl (population sample, papers cited 50+, meta-analyses' studies, landmarks).

    ../../.venv/bin/python replay.py        (GPU 1; keys in the environment; writes replay.json)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
OUT = Path(os.environ.get("REPLAY_OUT", "/mnt/fast/scholarsreadinglist/search/fields-r3"))
RESULTS = Path(__file__).parent / os.environ.get("REPLAY_RESULTS", "replay.json")
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
RUNS = {"frag": ("habitat fragmentation", None, "fields-test5/a699318c9a209d289e4c3702"),
        "llm": ("LLM post-training", "training large language models after pretraining: supervised fine-tuning and instruction "
                "tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement "
                "learning for reasoning, and alignment training", "fields-test6/1a0c3446c53e7f1da081def4")}
ARMS = {"per-topic": None, "general-2B": D / "m2b" / "merged", "general-0.8B": D / "m08" / "merged",
        "ensemble": "ensemble",      # Kimi and Jev as now; the 10k labels: mean logit of the per-topic decider and the 2B
        "2B-only": "2B-only",        # no Kimi, no Jev: the 2B is the teacher, the 10k labeller and the final checker
        "new08-only": "new08-only",  # the same, with the round-2 0.8B (Sol + Opus labels)
        "new2B-only": "new2B-only",  # the same, with the round-2 2B (skipped until it exists)
        "old08-only": "old08-only"}  # the same, with the round-1 0.8B (Kimi labels): the matched baseline
M2B = D / "m2b" / "merged"
ONLY = {"2B-only": M2B, "new08-only": D / "v2" / "models" / "0.8B" / "merged",
        "new2B-only": D / "v2" / "models" / "2B" / "merged", "old08-only": D / "m08" / "merged"}
if os.environ.get("REPLAY_ARMS"):          # run only these arms, in this order
    ARMS = {a: ARMS[a] for a in os.environ["REPLAY_ARMS"].split(",")}
VPY = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python"
lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def score_2b(texts, question, work: Path, model_path: Path = M2B) -> np.ndarray:
    """The general 2B on these papers, the topic's question in each prompt (vLLM, srl_search/vllm_decider.py)."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "score.jsonl").write_text("".join(json.dumps({"text": x, "system": question}) + "\n" for x in texts))
    J.wait_for_gpu(10000, print)
    J.run([VPY, str(Path(J.__file__).with_name("vllm_decider.py")), str(model_path), str(work / "score.jsonl"),
           str(work / "scores.json"), J.GPU], work / "log.txt")
    return np.array(json.loads((work / "scores.json").read_text()))
tests = [json.loads(l) for l in open(D / "tests.jsonl")]
ctx = F.Ctx()


def text_of(ws):
    det = F._details(ctx, ws)
    out = []
    for w in ws:
        d = det.get(int(w))
        if d:
            out.append(J.paper_text(d.get("title"), d.get("abstract")))
        else:
            nd = int(ctx.cit.nodes([int(w)])[0])
            out.append(J.paper_text(ctx.cit.title(nd) if nd >= 0 else "", None))
    return out


def measure(t, chosen):
    m = {"papers": len(chosen)}
    for s in ("population", "cited50"):
        rs = [r for r in tests if r["test"] == t and r["set"] == s and r["kimi"] is not None]
        if rs:
            y = np.array([r["kimi"] >= .5 for r in rs]); w = np.array([r["weight"] for r in rs])
            yes = np.array([r["id"] in chosen for r in rs])
            tp = (w * yes * y).sum(); P, R = tp / (w * yes).sum(), tp / (w * y).sum()
            m[s] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R), 3)}
    for s in ("meta_refs", "landmarks"):
        rs = [r for r in tests if r["test"] == t and r["set"] == s]
        if rs:
            m[s] = round(float(np.mean([r["id"] in chosen for r in rs])), 3)
            to = [r for r in rs if r["title_only"]]
            if to and s == "meta_refs":
                m["meta_refs_title_only"] = round(float(np.mean([r["id"] in chosen for r in to])), 3)
    return m


res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
saved = (J.Teacher, J.train_decider, J.decider_scores, J.Jev)
for t, (q, defin, run) in RUNS.items():
    f = json.loads((Path("/mnt/fast/scholarsreadinglist/search") / run / "field.json").read_text())
    teach = {int(w): v for w, v in f["labels"]["teacher"].items()}
    dec = {int(w): v for w, v in f["labels"]["decider"].items()}
    by_text_t = dict(zip(text_of(list(teach)), teach.values()))
    by_text_d = dict(zip(text_of(list(dec)), dec.values()))
    question = J.instruction(q, defin) + ANSWER

    class CachedTeacher:
        def __init__(self, b): self.usd, self.failed, self.slowest = 0.0, 0, {}
        def label(self, texts, q_, d_):
            miss = sum(x not in by_text_t for x in texts)
            assert miss == 0, f"{miss} teacher texts not cached"
            return np.array([by_text_t[x] for x in texts])

    for arm, model in ARMS.items():
        key = f"{t}/{arm}"
        if key in res:
            continue
        if model in ONLY and not (ONLY[model] / "model.safetensors").exists():
            print(key, "skipped: model not trained yet", flush=True)
            continue
        timing = {}

        def scores(model_dir, texts, say, model=model):
            t0 = time.time()
            if model == "ensemble":                     # the 2B's scores of these same papers, from its own arm
                prev = OUT / f"{t}-general-2B" / "work" / "decider"
                cache = dict(zip((json.loads(l)["text"] for l in open(prev / "score.jsonl")), json.loads((prev / "scores.json").read_text())))
                miss = sum(x not in cache or x not in by_text_d for x in texts)
                assert miss == 0, f"{miss} papers not scored by both"
                return 1 / (1 + np.exp(-(lgt(np.array([by_text_d[x] for x in texts])) + lgt(np.array([cache[x] for x in texts]))) / 2))
            if model in ONLY:
                p = score_2b(texts, question, model_dir, ONLY[model])
                timing["judge_s"] = round(time.time() - t0)
                return p
            if model is None:
                miss = sum(x not in by_text_d for x in texts)
                assert miss == 0, f"{miss} decider texts not cached"
                return np.array([by_text_d[x] for x in texts])
            model_dir.mkdir(parents=True, exist_ok=True)
            src = model_dir / "score.jsonl"
            src.write_text("".join(json.dumps({"text": x, "system": question}) + "\n" for x in texts))
            J.wait_for_gpu(10000, print)
            vpy = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python"
            J.run([vpy, str(Path(J.__file__).with_name("vllm_decider.py")), str(model), str(src), str(model_dir / "scores.json"), J.GPU],
                  model_dir / "log.txt")
            timing["judge_s"] = round(time.time() - t0)
            return np.array(json.loads((model_dir / "scores.json").read_text()))

        work = OUT / f"{t}-{arm}" / "work"

        class Teacher2B:                                # the 2B in Kimi's place
            def __init__(self, b): self.usd, self.failed, self.slowest = 0.0, 0, {}
            def label(self, texts, q_, d_): return score_2b(texts, question, work / "teacher2b", ONLY[model])

        class Check2B:                                  # the 2B in Jev's place
            def __init__(self, b): self.usd, self.failed, self.budget = 0.0, 0, b
            def label(self, pairs, q_, d_): return score_2b([J.paper_text(a_, b_) for a_, b_ in pairs], question, work / "check2b", ONLY[model])

        J.Teacher = Teacher2B if model in ONLY else CachedTeacher
        J.train_decider, J.decider_scores = (lambda *a, **k: None), scores
        if model in ONLY:
            J.Jev = Check2B
        job = F.Job(OUT, f"{t}-{arm}")
        job.dir.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.time()
            sel = F.select_field(ctx, job, q, defin, None)
        finally:
            J.Teacher, J.train_decider, J.decider_scores, J.Jev = saved
        chosen = {int(w) for w in sel["papers"]}
        res[key] = {**measure(t, chosen), "jev_usd": sel["report"]["check"]["usd"], "jev_checked": sel["report"]["check"]["papers"],
                    "cut": sel["report"]["judging"]["cut"], "seconds": round(time.time() - t0), **timing}
        print(key, json.dumps(res[key]), flush=True)
        RESULTS.write_text(json.dumps(res, indent=1))
print("REPLAY DONE", flush=True)
