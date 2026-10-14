ev_ids = np.array(EV["ids"]); ev_y = np.array([np.nan if y is None else y for y in EV["y"]]); ev_st = np.array(EV["stratum"])
ok_e = ~np.isnan(ev_y); w_e = np.array([EV["N"][s] / EV["n"][s] for s in ev_st])
refs = [json.loads(l) for l in open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/refs_labelled.jsonl")]
meta = json.load(open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/meta.json"))
ref_pos = [int(r["id"][1:]) for r in refs if r["label"] == "on topic"]
ref_ab = np.array([bool(meta["W%d" % w]["abstract"]) for w in ref_pos])
def _pr(sel, idx):
    y, w, s = ev_y[idx], w_e[idx], sel[idx]
    tp = (w * s * y).sum(); return tp / max(1e-9, (w * s).sum()), tp / max(1e-9, (w * y).sum())
def evaluate(chosen, name="", boot=300):
    """chosen: set of work ids selected (the field). Population precision/recall over the candidate universe."""
    sel = np.array([int(w) in chosen for w in ev_ids], float)
    base = np.flatnonzero(ok_e)
    p, r = _pr(sel, base)
    rb = np.random.default_rng(0); bs = []
    groups = {s: base[ev_st[base] == s] for s in set(ev_st)}
    for _ in range(boot):
        idx = np.concatenate([rb.choice(g, len(g)) for g in groups.values()]); bs.append(_pr(sel, idx))
    bs = np.array(bs); lo, hi = np.percentile(bs, [5, 95], axis=0)
    found = np.array([w in chosen for w in ref_pos])
    f1 = 2 * p * r / max(1e-9, p + r)
    out = {"papers": len(chosen), "precision": round(p, 3), "p90": [round(lo[0], 3), round(hi[0], 3)],
           "pop_recall": round(r, 3), "r90": [round(lo[1], 3), round(hi[1], 3)], "F1": round(f1, 3),
           "meta_recall": round(found.mean(), 3), "meta_title_only": round(found[~ref_ab].mean(), 3)}
    print(f"{name:28s}", json.dumps(out)); return out
est_pos = sum(EV["N"][s] * np.nanmean(ev_y[ev_st == s]) for s in EV["N"])
print("estimated on-topic papers among the candidates:", round(est_pos))
for s in sorted(EV["N"]): print(f"  {s:14s} N={EV['N'][s]:7,d} on topic {np.nanmean(ev_y[ev_st == s]):.2f}")
v3_set = set(int(w) for w in f3["papers"]); evaluate(v3_set, "v3")
v1 = json.load(open("/mnt/fast/scholarsreadinglist/search/fields/a699318c9a209d289e4c3702/field.json"))["papers"]
evaluate(set(int(w) for w in v1), "v1 Jev (not same universe)")
