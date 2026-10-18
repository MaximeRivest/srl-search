"""Eight judges of "is this paper about the topic?", asked the same question (srl_search/judges.py instruction):
the reference model and Jev (answers already in tests.jsonl), and six more through two accounts (subs.py; which
models: the private settings, teachers.py JUDGES; here they are judge-1 .. judge-6).

Papers: 300 per held-out topic, drawn from each topic's stratified population sample in proportion (so weights still
stand for the whole-field candidates), and LLM post-training's 44 landmark papers (on topic by construction: the only
reference independent of every judge).
No judge is the truth here. For each judge: agreement with the majority of the other seven (leave-one-out), and its
precision and recall against that majority, weighted to the population; landmarks found; agreement between every pair.

    sciencemadereadable/.venv/bin/python compare.py run       (resumable; stops a judge at 90% of its account's allowance)
    sciencemadereadable/.venv/bin/python compare.py report
"""
import json, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import teachers  # noqa: E402
D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
ANS = D / "compare_answers.jsonl"
PER_TOPIC = 300


def judges():
    """("reference", "jev", judge-1 ...), and {name in compare_answers.jsonl: public name} for the private ones."""
    private = teachers.judges()
    return ("reference", "jev") + tuple(private), {j.get("stored", n): n for n, j in private.items()}


def sample():
    rows = [json.loads(l) for l in open(D / "tests.jsonl")]
    rng = random.Random(7)
    out = []
    for t in ("frag", "llm"):
        pop = [r for r in rows if r["test"] == t and r["set"] == "population" and r["reference"] is not None]
        by = defaultdict(list)
        for r in pop:
            by[r["stratum"]].append(r)
        for s, rs in by.items():
            k = max(2, round(PER_TOPIC * len(rs) / len(pop)))
            pick = rng.sample(rs, min(k, len(rs)))
            for r in pick:
                out.append({**r, "weight": r["weight"] * len(rs) / len(pick)})   # N_s / n'_s
    out += [r for r in rows if r["set"] == "landmarks"]
    for i, r in enumerate(out):
        r["key"] = f"{r['test']}:{r['set']}:{r['id']}"
    return out


def run():
    import lm15
    import subs
    _, public = judges()
    stored = {n: s for s, n in public.items()}
    papers = sample()
    done = set()
    if ANS.exists():
        done = {(a["key"], public.get(a["judge"], a["judge"])) for a in map(json.loads, open(ANS))}
    out, lock = open(ANS, "a"), threading.Lock()
    stopped = {}

    def allowance_ok(judge):
        if judge in stopped:
            return False
        j = subs.MODELS[judge]
        try:
            u = teachers.windows(j["account"], teachers.usage(j["account"]))
            checked = [w for w in u["windows"] if j.get("stop_windows") is None or w["window"] in j["stop_windows"]]
            bad = any(w["used"] >= 90 for w in checked) or bool(u["blocked"])
        except Exception:  # noqa: BLE001  (usage endpoint down: keep going, rate limits still stop us)
            return True
        if bad:
            stopped[judge] = f"allowance: {u}"
            print(judge, "STOPPED", u, flush=True)
        return not bad

    def one(job):
        judge, p = job
        for attempt in range(6):
            t0 = time.time()
            try:
                y = subs.ask(judge, p["topic"], p["definition"], p["text"])
                with lock:
                    out.write(json.dumps({"key": p["key"], "judge": stored[judge], "y": y, "s": round(time.time() - t0, 2)}) + "\n")
                    out.flush()
                return
            except lm15.RateLimitError:
                print(judge, "rate limited; waiting", flush=True)
                time.sleep(60 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                print(judge, "error", type(e).__name__, str(e)[:160], flush=True)
                time.sleep(5)

    def judge_all(judge):
        todo = [(judge, p) for p in papers if (p["key"], judge) not in done]
        print(judge, len(todo), "to ask", flush=True)
        for s in range(0, len(todo), 100):
            if not allowance_ok(judge):
                return
            with ThreadPoolExecutor(6) as ex:
                list(ex.map(one, todo[s:s + 100]))
        print(judge, "done", flush=True)

    # one model at a time per account, the accounts side by side
    threads = [threading.Thread(target=lambda js=js: [judge_all(j) for j in js]) for js in teachers.judge_run_order().values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("COMPARE DONE", stopped, flush=True)


def report():
    JUDGES, public = judges()
    labels = {n: j.get("label", n) for n, j in teachers.judges().items()}   # console only; compare.json: public names
    papers = sample()
    ans = defaultdict(dict)
    secs = defaultdict(list)
    for a in map(json.loads, open(ANS)):
        name = public.get(a["judge"], a["judge"])
        ans[a["key"]][name] = a["y"]
        secs[name].append(a["s"])
    for p in papers:
        ans[p["key"]]["reference"] = None if p["reference"] is None else float(p["reference"] >= 0.5)
        ans[p["key"]]["jev"] = None if p["jev"] is None else float(p["jev"] >= 0.5)
    res = {}
    for j in JUDGES:
        r = {"seconds_median": round(float(np.median(secs[j])), 2) if secs[j] else None}
        for t in ("frag", "llm"):
            ps = [p for p in papers if p["test"] == t and p["set"] == "population"]
            agree, w_, tp, yes_w, pos_w = 0.0, 0.0, 0.0, 0.0, 0.0
            unread = 0
            for p in ps:
                a = ans[p["key"]]
                mine = a.get(j)
                if mine is None:
                    unread += j in a
                    continue
                others = [v for k, v in a.items() if k != j and v is not None]
                if len(others) < 5 or sum(others) * 2 == len(others):
                    continue
                c = float(sum(others) * 2 > len(others))
                w = p["weight"]
                agree += w * (mine == c); w_ += w
                tp += w * mine * c; yes_w += w * mine; pos_w += w * c
            if w_:
                P, R = tp / max(1e-9, yes_w), tp / max(1e-9, pos_w)
                r[t] = {"agree_with_others": round(agree / w_, 3), "precision": round(P, 3), "recall": round(R, 3),
                        "f1": round(2 * P * R / max(1e-9, P + R), 3), "unreadable": unread}
        lm = [ans[p["key"]].get(j) for p in papers if p["set"] == "landmarks"]
        lm = [v for v in lm if v is not None]
        r["landmarks"] = round(float(np.mean(lm)), 3) if lm else None
        res[j] = r
    pair = {}
    pop = [p for p in papers if p["set"] == "population"]
    for a_ in JUDGES:
        for b_ in JUDGES:
            if a_ < b_:
                num = den = 0.0
                for p in pop:
                    x, y = ans[p["key"]].get(a_), ans[p["key"]].get(b_)
                    if x is not None and y is not None:
                        num += p["weight"] * (x == y); den += p["weight"]
                pair[f"{a_} ~ {b_}"] = round(num / den, 3) if den else None
    (HERE / "compare.json").write_text(json.dumps({"judges": res, "pairs": pair}, indent=1))
    print(f"{'judge':11s} {'frag agree':>10s} {'P/R/F1':>16s} {'llm agree':>10s} {'P/R/F1':>16s} {'landmarks':>9s} {'s/answer':>8s}")
    for j, r in res.items():
        f = r.get("frag", {}); l = r.get("llm", {})
        fmt = lambda m: f"{m.get('precision', 0):.2f}/{m.get('recall', 0):.2f}/{m.get('f1', 0):.2f}" if m else "-"
        print(f"{labels.get(j, j):11s} {f.get('agree_with_others', 0):10.3f} {fmt(f):>16s} {l.get('agree_with_others', 0):10.3f} {fmt(l):>16s} "
              f"{(r['landmarks'] or 0):9.2f} {r['seconds_median'] or 0:8.2f}")
    print("closest pairs:", sorted(pair.items(), key=lambda x: -(x[1] or 0))[:6])
    print("farthest pairs:", sorted(pair.items(), key=lambda x: (x[1] or 1))[:6])


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
