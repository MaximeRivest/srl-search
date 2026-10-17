"""Acceptance tests for a round-3 classifier (and the round-2 judges as the starting point), one vLLM run per model.

  expert   the 200 expert-classified articles (jevarticle; Science-Metrix, a scheme never trained on):
           flat     one question, all 176 subfields, taxonomy order            -> agreement with the experts' mode
           shuffled the same with 2 random option orders                       -> does the pick depend on order?
           field    one question, the 22 fields                                -> field-level agreement
           tree     P(subfield) = P(field) x P(subfield | field): 1 + 22 questions per paper (prefix-cached)
  yesno    the 8 held-out topics of round 2 (4,800 pairs, Kimi reference): macro average precision and AUC
  unseen   eval units (JEL, held-out invented schemes, held-out papers on training schemes), against the teachers:
           agreement with each teacher's top pick, soft log loss, calibration (ECE); teacher-vs-teacher as ceiling

    ../../../.venv/bin/python eval3.py NAME MODEL_DIR [legacy]     (SRL_FIELDS_GPU; legacy = round-2 yes/no prompt)
"""
import collections, csv, json, math, os, random, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import judges as J
import fmt

V3 = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v3")
V2 = V3.parent / "v2"
JA = Path.home() / "Projects/jevarticle"
VPY = os.environ.get("SRL_VLLM_PYTHON", "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python")
OUT = V3 / "evaluation"; OUT.mkdir(exist_ok=True)
ANSWER_V2 = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'

tax = list(csv.DictReader(open(JA / "data/taxonomy.csv")))
arts = list(csv.DictReader(open(JA / "data/articles_resolved.csv")))
subs = [t["subfield"] for t in tax]
field_of = {t["subfield"]: t["field"] for t in tax}
fields = list(dict.fromkeys(t["field"] for t in tax))
domain_of_field = {t["field"]: t["domain"] for t in tax}
SM = "Science-Metrix classification of science (176 subfields within 22 fields and 6 domains)"


def art_text(a):
    s = f"Title: {a['title']}\nVenue: {a['journal']}\nYear: {a['year']}"
    return s + (f"\nAbstract: {a['abstract'][:1500]}" if a["abstract"] else "\n(no abstract)")


def pick_row(tag, paper, scheme, question, options):
    codes = fmt.assign_codes(len(options))
    return {"tag": tag, "user": fmt.pick_prompt(paper, scheme, question, options, codes), "answers": codes}


def build(legacy):
    rows = []
    sub_opts = [{"name": t["subfield"], "parent": t["field"]} for t in tax]
    for i, a in enumerate(arts):
        p = art_text(a)
        rows.append(pick_row(("flat", i, None), p, SM, "Which subfield best describes the research content of this paper?", sub_opts))
        for k in range(2):
            perm = list(range(len(sub_opts))); random.Random(1000 * k + i).shuffle(perm)
            r = pick_row(("shuffled", i, perm), p, SM, "Which subfield best describes the research content of this paper?",
                         [sub_opts[j] for j in perm])
            rows.append(r)
        rows.append(pick_row(("field", i, None), p, SM, "Which field best describes the research content of this paper?",
                             [{"name": f, "parent": domain_of_field[f]} for f in fields]))
        for f in fields:
            ch = [s for s in subs if field_of[s] == f]
            rows.append(pick_row(("tree", i, f), p, SM + f" — within: {f}",
                                 "Which subfield best describes the research content of this paper?", [{"name": s} for s in ch]))
    # yes/no on the 8 held-out topics
    labels = {}
    for f in (V2 / "benchmark-kimi").glob("wave-*.json"):
        for r in json.loads(f.read_text())["answers"]:
            if r["y"] is not None:
                labels.setdefault(r["id"], set()).add(r["y"])
    for f in sorted((V2 / "benchmark").glob("*.json")):
        for r in json.loads(f.read_text())["pairs"]:
            ys = labels.get(r["id"], set())
            if len(ys) == 1:
                user = fmt.yesno_prompt(r["text"], r["topic"], r["definition"])
                row = {"tag": ("yesno", r["topic"], next(iter(ys)), r["weight"]), "user": user, "answers": ["yes", "no"]}
                if legacy:                                  # round-2 judge: its own prompt
                    row["legacy_system"] = J.instruction(r["topic"], r["definition"]) + ANSWER_V2; row["text"] = r["text"]
                rows.append(row)
    # held-out eval units, against the teachers
    for f in sorted((V3 / "queue" / "eval").glob("*.json")):
        u = json.loads(f.read_text())
        ans = {}
        for m in ("sol", "opus"):
            a = V3 / "answers" / m / f.name
            if a.exists():
                for x in json.loads(a.read_text())["answers"]:
                    ans.setdefault(x["work_id"], {})[m] = x["dist"]
        group = "jel" if u["scheme"] == "jel" else "invented" if u["scheme"].startswith("inv-") else "papers"
        for p in u["papers"]:
            if p["work_id"] in ans:
                r = pick_row(("unseen", group, u["id"], p["work_id"]), p["text"], u["scheme_text"], u["question"],
                             [{k: o.get(k) for k in ("name", "about", "parent")} for o in u["options"]] if u["describe"]
                             else [{"name": o["name"], "parent": o.get("parent")} for o in u["options"]])
                r["nodes"] = [o["node"] for o in u["options"]]; r["teachers"] = ans[p["work_id"]]
                rows.append(r)
    return rows


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)]


def ece(conf, hit, bins=10):
    conf, hit = np.array(conf), np.array(hit, float); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return round(float(e), 4)


def score(rows, P):
    votes = collections.defaultdict(lambda: collections.defaultdict(list))
    subset = {}
    for r in csv.DictReader(open(JA / "data/gold_standard_long.csv")):
        votes[r["scopus_id"]][r["evaluator"]].append(r["subfield"]); subset[r["scopus_id"]] = r["subset"]
    def modes(xs):
        c = collections.Counter(xs); m = max(c.values()); return {s for s, n in c.items() if n == m}
    gold = [modes([v[0] for v in votes[a["scopus_id"]].values()]) for a in arts]
    gold_f = [modes([field_of[v[0]] for v in votes[a["scopus_id"]].values()]) for a in arts]
    sub = [subset[a["scopus_id"]] for a in arts]
    res = {}
    flat = {}; shuf = collections.defaultdict(list); fieldp = {}; tree = collections.defaultdict(dict)
    yes = collections.defaultdict(list); unseen = collections.defaultdict(list)
    for r, p in zip(rows, P):
        t = r["tag"]; p = np.array(p)
        if t[0] == "flat":
            flat[t[1]] = p
        elif t[0] == "shuffled":
            q = np.zeros(len(p)); q[np.array(t[2])] = p; shuf[t[1]].append(q)
        elif t[0] == "field":
            fieldp[t[1]] = p
        elif t[0] == "tree":
            tree[t[1]][t[2]] = p
        elif t[0] == "yesno":
            yes[t[1]].append((float(p[0]), t[2], t[3]))
        else:
            unseen[t[1]].append((r, p))
    def agree(top_names):
        hits = [top_names[i] in gold[i] for i in range(len(arts))]
        return {"disciplinary": wilson(sum(h for h, s in zip(hits, sub) if s == "disciplinary"), sub.count("disciplinary")),
                "multidisciplinary": wilson(sum(h for h, s in zip(hits, sub) if s == "multidisciplinary"), sub.count("multidisciplinary"))}, hits
    a_flat, hits = agree([subs[int(flat[i].argmax())] for i in range(len(arts))])
    conf = [float(flat[i].max()) for i in range(len(arts))]
    top3 = np.mean([bool({subs[j] for j in np.argsort(-flat[i])[:3]} & gold[i]) for i in range(len(arts))])
    res["expert_flat"] = {**a_flat, "top3": round(100 * float(top3), 1), "ece_vs_experts": ece(conf, hits),
                          "mean_confidence": round(float(np.mean(conf)), 3)}
    same = [np.mean([int(q.argmax() == flat[i].argmax()) for q in shuf[i]]) for i in range(len(arts))]
    a_sh, _ = agree([subs[int(shuf[i][0].argmax())] for i in range(len(arts))])
    res["expert_shuffled"] = {**a_sh, "same_pick_as_unshuffled": round(100 * float(np.mean(same)), 1)}
    fh = [fields[int(fieldp[i].argmax())] in gold_f[i] for i in range(len(arts))]
    res["expert_field"] = {"field_agreement": wilson(sum(fh), len(fh))}
    tp = []
    for i in range(len(arts)):
        v = {}
        for fi, f in enumerate(fields):
            ch = [s for s in subs if field_of[s] == f]
            for s, q in zip(ch, tree[i][f]):
                v[s] = fieldp[i][fi] * q
        tp.append(max(v, key=v.get))
    res["expert_tree"], _ = agree(tp)
    aps, aucs = [], []
    for topic, xs in sorted(yes.items()):
        p = np.array([x[0] for x in xs]); y = np.array([x[1] for x in xs]); w = np.array([x[2] for x in xs])
        aps.append(average_precision_score(y, p, sample_weight=w)); aucs.append(roc_auc_score(y, p, sample_weight=w))
    res["yesno_heldout_topics"] = {"macro_average_precision": round(float(np.mean(aps)), 4),
                                   "macro_auc": round(float(np.mean(aucs)), 4), "topics": len(aps)}
    for g, xs in sorted(unseen.items()):
        out = {}
        for m in ("sol", "opus"):
            hit, conf, ll = [], [], []
            for r, p in xs:
                d = r["teachers"].get(m)
                if not d:
                    continue
                t = np.array([d.get(n, 0.0) for n in r["nodes"]]); t = t / t.sum()
                hit.append(int(p.argmax() == t.argmax())); conf.append(float(p.max()))
                ll.append(float(-(t * np.log(np.clip(p, 1e-6, 1))).sum()))
            if hit:
                out[m] = {"agree_top": round(100 * float(np.mean(hit)), 1), "log_loss": round(float(np.mean(ll)), 3),
                          "ece": ece(conf, hit), "n": len(hit)}
        both = [(r, p) for r, p in xs if len(r["teachers"]) == 2]
        if both:
            tt = [int(max(r["teachers"]["sol"], key=r["teachers"]["sol"].get) == max(r["teachers"]["opus"], key=r["teachers"]["opus"].get))
                  for r, _ in both]
            out["sol_vs_opus_agree_top"] = round(100 * float(np.mean(tt)), 1)
        res[f"unseen_{g}"] = out
    return res


def main(name, model, legacy=False):
    rows = build(legacy)
    src, dst = OUT / f"{name}-rows.jsonl", OUT / f"{name}-scores.json"
    if not dst.exists():
        new = [r for r in rows if "legacy_system" not in r]
        src.write_text("".join(json.dumps({"user": r["user"], "answers": r["answers"]}) + "\n" for r in new))
        J.wait_for_gpu(9000, print)
        J.run([VPY, str(HERE / "vllm_classify.py"), str(model), str(src), str(dst), J.GPU], OUT / f"{name}.log",
              env={"HF_HUB_OFFLINE": "1"})
        if legacy:                                         # round-2 yes/no in its own prompt
            leg = [r for r in rows if "legacy_system" in r]
            lsrc, ldst = OUT / f"{name}-legacy.jsonl", OUT / f"{name}-legacy.json"
            lsrc.write_text("".join(json.dumps({"system": r["legacy_system"], "text": r["text"]}) + "\n" for r in leg))
            J.run([VPY, str(Path(J.__file__).with_name("vllm_decider.py")), str(model), str(lsrc), str(ldst), J.GPU],
                  OUT / f"{name}.log")
            P, L = iter(json.loads(dst.read_text())), iter(json.loads(ldst.read_text()))
            dst.write_text(json.dumps([[next(L), None] if "legacy_system" in r else next(P) for r in rows]))
    P = json.loads(dst.read_text())
    res = score(rows, P)
    sp = Path(str(OUT / f"{name}-scores.json") + ".speed.json")
    if sp.exists():
        res["speed"] = json.loads(sp.read_text())
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(name, json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "legacy")
