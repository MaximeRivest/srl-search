"""Can our field judges still say WHICH field a paper belongs to, and did our training hurt that?

Benchmark: ~/Projects/jevarticle (Rivest et al. 2021, PLOS ONE 0251493, Benchmark 2): 200 articles classified by 5-6
experts each into the 176 Science-Metrix subfields. Score = agreement with the experts' modal first choice (a tie: any
tied subfield counts), disciplinary and multidisciplinary halves separately, as in jevarticle/scripts/score.py.

Models: the base Qwen3.5-0.8B and 2B (what we fine-tuned), round 1 (Kimi labels) and round 2 (Sol + Opus labels, live).
Two ways to ask (vllm_pick.py):
  pick    the jevarticle prompt, same wording as its Jev and Qwen 27B runs (title, journal, year, abstract; all 176 names
          listed), every name scored as a full answer: a distribution over the 176
  yesno   our judge's own question, once per subfield ("is <subfield> a central subject of this paper?", title + abstract
          as in training), the 176 P(yes) normalised into a distribution

    ../../../.venv/bin/python fieldpick.py [run|score]      (SRL_FIELDS_GPU: card; run skips what exists)
"""
import csv, collections, json, math, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import judges as J

JA = Path.home() / "Projects/jevarticle"
sys.path.insert(0, str(JA / "scripts"))
import run_qwen as RQ                                      # the prompt wording of the Qwen 27B runs

VPY = os.environ.get("SRL_VLLM_PYTHON", "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python")
D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
MODELS = {"base-0.8B": "Qwen/Qwen3.5-0.8B", "round1-0.8B": D / "m08/merged", "round2-0.8B": D / "v2/models/0.8B/merged",
          "base-2B": "Qwen/Qwen3.5-2B", "round1-2B": D / "m2b/merged", "round2-2B": D / "v2/models/2B/merged"}
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'

tax = list(csv.DictReader(open(JA / "data/taxonomy.csv")))
names = [t["subfield"] for t in tax]
field_of = {t["subfield"]: t["field"] for t in tax}
fields = sorted(set(field_of.values()))
F_IDX = np.array([fields.index(field_of[n]) for n in names])
arts = list(csv.DictReader(open(JA / "data/articles_resolved.csv")))
tax_text = "Subfields (subfield — field (domain)):\n" + "\n".join(f"{t['subfield']} — {t['field']} ({t['domain']})" for t in tax)


def inputs():
    (OUT / "names.json").write_text(json.dumps(names))
    with open(OUT / "pick.jsonl", "w") as f:
        for a in arts:
            user = f"{tax_text}\n\nArticle:\n{RQ.build_state(a, 'abstract')}\n\n{RQ.QUESTION}"
            f.write(json.dumps({"system": RQ.SYSTEM_DIRECT, "user": user}) + "\n")
    with open(OUT / "yesno.jsonl", "w") as f:
        for a in arts:
            text = J.paper_text(a["title"], a["abstract"])
            for t in tax:
                defin = f"a subfield of {t['field']}, in {t['domain']}"
                f.write(json.dumps({"system": J.instruction(t["subfield"], defin) + ANSWER, "text": text}) + "\n")


def gate(minutes):
    """Never compete with a live whole-field map for the GPU: wait while one is judging or due within `minutes`.
    (Map-only rebuilds use no GPU.)"""
    from srl_search import field as LF
    while not os.environ.get("FIELDPICK_NO_GATE"):       # GPU 0 (run_gpu0.sh): the live maps use GPU 1
        plan = LF.FieldStore().plan()
        run_ = plan["running"]
        soon = any(j["st"].get("kind") == "full" and j["start"] - plan["now"] < 60 * minutes for j in plan["waiting"][:1])
        if not (run_ and run_["st"].get("kind") == "full") and not soon:
            break
        print("a live whole-field map is being made or due; waiting", flush=True)
        time.sleep(60)
    J.wait_for_gpu(9000, print)


CHUNK = 40                                                 # papers per yes/no call: ~7,000 prompts, a few minutes


def run():
    inputs()
    order = [(n, "pick") for n in MODELS] + [(n, "yesno") for n in ("base-0.8B", "round2-0.8B", "base-2B", "round2-2B",
                                                                     "round1-0.8B", "round1-2B")]
    yesno = open(OUT / "yesno.jsonl").readlines()
    for name, mode in order:
        m = MODELS[name]
        dst = OUT / f"{name}-{mode}.json"
        if dst.exists():
            continue
        spent = 0.0
        parts = [(0, len(arts))] if mode == "pick" else [(i, min(i + CHUNK, len(arts))) for i in range(0, len(arts), CHUNK)]
        out = []
        for a, b in parts:
            part = OUT / f"{name}-{mode}-{a}.json"
            if not part.exists():
                gate(5 if mode == "pick" else 12)
                src = OUT / f"{mode}.jsonl"
                if mode == "yesno":
                    src = OUT / f"yesno-{a}.jsonl"
                    src.write_text("".join(yesno[a * len(names):b * len(names)]))
                t = time.time()
                J.run([VPY, str(HERE / "vllm_pick.py"), str(m), mode, str(src), str(OUT / "names.json"), str(part), J.GPU],
                      OUT / f"{name}-{mode}.log", env={"HF_HUB_OFFLINE": "1"})
                spent += time.time() - t
                print(f"{name} {mode} papers {a}-{b}: {time.time() - t:.0f} s", flush=True)
            out += json.loads(part.read_text())
        dst.write_text(json.dumps(out))
        sec = OUT / f"{name}-{mode}.seconds"
        if not sec.exists():
            sec.write_text(f"{spent:.0f}")


# ------------------------------------------------------------------------------------------------ scoring
votes = collections.defaultdict(lambda: collections.defaultdict(list))
subset = {}
for r in csv.DictReader(open(JA / "data/gold_standard_long.csv")):
    votes[r["scopus_id"]][r["evaluator"]].append(r["subfield"])
    subset[r["scopus_id"]] = r["subset"]


def modes(xs):
    c = collections.Counter(xs); top = max(c.values())
    return {s for s, n in c.items() if n == top}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def load(name, mode):
    f = OUT / f"{name}-{mode}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    if mode == "pick":
        return np.array([r["p"] for r in d]), np.array([r["coverage"] for r in d])
    y = np.array(d).reshape(len(arts), len(names))
    return y / y.sum(1, keepdims=True), y


def score():
    firsts = [[v[0] for v in votes[a["scopus_id"]].values()] for a in arts]
    gold = [modes(f) for f in firsts]
    gold_field = [modes([field_of[x] for x in f]) for f in firsts]
    sub = np.array([subset[a["scopus_id"]] for a in arts])
    has_abs = np.array([bool(a["abstract"]) for a in arts])
    rows, hits = {}, {}
    for mode in ("pick", "yesno"):
        for name in MODELS:
            P, extra = load(name, mode)
            if P is None:
                continue
            top = P.argmax(1)
            hit = np.array([names[top[i]] in gold[i] for i in range(len(arts))])
            Pf = np.zeros((len(arts), len(fields)))
            np.add.at(Pf.T, F_IDX, P.T)
            fhit = np.array([fields[Pf[i].argmax()] in gold_field[i] for i in range(len(arts))])
            top3 = np.array([bool({names[j] for j in np.argsort(-P[i])[:3]} & gold[i]) for i in range(len(arts))])
            idx = {n: j for j, n in enumerate(names)}
            p_expert = np.array([np.mean([P[i, idx[x]] for x in firsts[i]]) for i in range(len(arts))])
            loss = np.array([np.mean([-math.log(max(P[i, idx[x]], 1e-6)) for x in firsts[i]]) for i in range(len(arts))])
            conf = P.max(1)
            r = {"disciplinary": wilson(hit[sub == "disciplinary"].sum(), (sub == "disciplinary").sum()),
                 "multidisciplinary": wilson(hit[sub == "multidisciplinary"].sum(), (sub == "multidisciplinary").sum()),
                 "all": 100 * hit.mean(), "with_abstract": 100 * hit[has_abs].mean(), "title_only": 100 * hit[~has_abs].mean(),
                 "field": 100 * fhit.mean(), "top3": 100 * top3.mean(), "p_on_expert_choice": float(p_expert.mean()),
                 "log_loss": float(loss.mean()), "mean_confidence": float(conf.mean()),
                 "hit_when_confident": 100 * float(hit[conf >= np.quantile(conf, 2 / 3)].mean()),
                 "hit_when_unsure": 100 * float(hit[conf <= np.quantile(conf, 1 / 3)].mean())}
            if mode == "pick":
                r["coverage"] = float(extra.mean())
            sec = OUT / f"{name}-{mode}.seconds"
            r["seconds"] = int(sec.read_text()) if sec.exists() else None
            rows[f"{name}/{mode}"] = r
            hits[f"{name}/{mode}"] = hit.astype(float)
    # paired bootstrap over articles: trained minus base, same mode and size
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(arts), (4000, len(arts)))
    paired = {}
    for k in hits:
        name, mode = k.split("/")
        size = name.split("-")[1]
        b = f"base-{size}/{mode}"
        if not name.startswith("base") and b in hits:
            d = hits[k] - hits[b]
            paired[k] = {"minus_base": 100 * d.mean(), "ci95": (100 * np.percentile(d[B].mean(1), [2.5, 97.5])).tolist()}
    (HERE / "fieldpick.json").write_text(json.dumps({"rows": rows, "paired": paired}, indent=1))
    print(f"{'model/mode':22} {'discipl.':>16} {'multidisc.':>16} {'field':>6} {'top3':>5} {'p(exp)':>6} {'loss':>5} {'cov':>5} {'conf':>5} {'sec':>5}")
    for k, r in rows.items():
        f = lambda t: f"{t[0]:3.0f}% [{t[1]:2.0f}-{t[2]:2.0f}]"
        print(f"{k:22} {f(r['disciplinary']):>16} {f(r['multidisciplinary']):>16} {r['field']:5.0f}% {r['top3']:4.0f}% "
              f"{r['p_on_expert_choice']:6.2f} {r['log_loss']:5.2f} {r.get('coverage', float('nan')):5.2f} "
              f"{r['mean_confidence']:5.2f} {r['seconds'] or 0:5}")
    for k, v in paired.items():
        print(f"{k:22} vs base: {v['minus_base']:+5.1f} points [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]")


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
    if (sys.argv[1] if len(sys.argv) > 1 else "run") == "run":
        score()
