"""Step 5: the judges on the two held-out topics (tests.jsonl, build_tests.py), Kimi K3 as the yardstick.

Judges: general models (train.py: the question in each prompt), the per-topic deciders of the real whole-field runs
(each trained on 400 Kimi answers about its own topic: the baseline to beat), and Jev (its answers are in tests.jsonl).
For each topic:
  population  stratified sample of the whole-field candidates, weighted to the population: AUC; agreement with Kimi;
              precision and recall if the judge decided every candidate alone (at 0.5)
  cited50     papers cited 50+ times (what tops reading lists; habitat fragmentation): AUC, precision, recall
  meta_refs   the studies 19 meta-analyses cite that Kimi calls on topic: share the judge says yes to (title only too)
  landmarks   44 landmark papers of LLM post-training: share it says yes to

    ../../.venv/bin/python eval.py NAME=MODEL_DIR ...     (GPU 1; writes results.json, adds to it)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J  # noqa: E402

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
HERE = Path(__file__).resolve().parent
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
BASELINES = {"frag": "/mnt/fast/scholarsreadinglist/search/fields-test5/a699318c9a209d289e4c3702/work/decider/merged",
             "llm": "/mnt/fast/scholarsreadinglist/search/fields-test6/1a0c3446c53e7f1da081def4/work/decider/merged"}
rows = [json.loads(l) for l in open(D / "tests.jsonl")]


def vllm_scores(model_dir: str, rs: list[dict], general: bool) -> np.ndarray:
    work = D / "eval_work"
    work.mkdir(exist_ok=True)
    src, dst = work / "in.jsonl", work / "out.json"
    src.write_text("".join(json.dumps({"text": r["text"], **({"system": J.instruction(r["topic"], r["definition"]) + ANSWER}
                                                             if general else {})}) + "\n" for r in rs))
    J.wait_for_gpu(9000, print)
    env = {**J.gpu_env(), "HF_HUB_OFFLINE": "1"}
    vpy = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python"
    with open(work / "log.txt", "a") as f:
        r = subprocess.run([vpy, str(HERE.parents[1] / "srl_search" / "vllm_decider.py"), model_dir, str(src), str(dst), J.GPU],
                           env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"vLLM failed: {(work / 'log.txt').read_text()[-1500:]}")
    return np.array(json.loads(dst.read_text()))


def metrics(rs: list[dict], p: np.ndarray) -> dict:
    out = {}
    for test in ("frag", "llm"):
        m = {}
        for s in ("population", "cited50"):
            idx = [i for i, r in enumerate(rs) if r["test"] == test and r["set"] == s and r["kimi"] is not None
                   and p[i] == p[i]]                    # a judge that did not answer a paper: left out
            if not idx:
                continue
            y = np.array([rs[i]["kimi"] >= 0.5 for i in idx])
            w = np.array([rs[i]["weight"] for i in idx])
            q = p[idx]
            yes = q >= 0.5
            tp = (w * yes * y).sum()
            prec, rec = tp / max(1e-9, (w * yes).sum()), tp / max(1e-9, (w * y).sum())
            m[s] = {"auc": round(float(roc_auc_score(y, q, sample_weight=w)), 4),
                    "agree": round(float((w * (yes == y)).sum() / w.sum()), 4),
                    "agree_sample": round(float((yes == y).mean()), 4),
                    "precision": round(float(prec), 3), "recall": round(float(rec), 3),
                    "f1": round(float(2 * prec * rec / max(1e-9, prec + rec)), 3)}
        for s in ("meta_refs", "landmarks"):
            idx = [i for i, r in enumerate(rs) if r["test"] == test and r["set"] == s and p[i] == p[i]]
            if idx:
                yes = p[idx] >= 0.5
                to = np.array([rs[i]["title_only"] for i in idx])
                m[s] = {"found": round(float(yes.mean()), 3), "n": len(idx)}
                if to.any():
                    m[s]["found_title_only"] = round(float(yes[to].mean()), 3)
        out[test] = m
    return out


results = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else {}
if "jev" not in results:
    results["jev"] = metrics(rows, np.array([np.nan if r["jev"] is None else r["jev"] for r in rows]))
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    if name == "per-topic":                 # each topic's own decider, on its own topic's rows
        p = np.full(len(rows), np.nan)
        for test, d in BASELINES.items():
            idx = [i for i, r in enumerate(rows) if r["test"] == test]
            p[idx] = vllm_scores(d, [rows[i] for i in idx], general=False)
    else:
        p = vllm_scores(str(Path(path) / "merged") if (Path(path) / "merged").exists() else path, rows, general=True)
    np.save(D / f"scores_{name}.npy", p)
    results[name] = metrics(rows, p)
    print(name, json.dumps(results[name]), flush=True)
(HERE / "results.json").write_text(json.dumps(results, indent=1))
print("jev", json.dumps(results["jev"]))
