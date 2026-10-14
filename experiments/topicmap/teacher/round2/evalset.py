alloc = {"U|0.9+|in": 60, "U|.5-.9|in": 200, "U|.5-.9|out": 150, "U|.2-.5|in": 120, "U|.2-.5|out": 200,
         "U|.05-.2|in": 20, "U|.05-.2|out": 80, "U|<.05|out": 20, "C|title|in": 120, "C|title|out": 200,
         "C|.5-.9|in": 25, "C|.5-.9|out": 30, "C|.2-.5|in": 40, "C|.2-.5|out": 120, "C|.05-.2|out": 50}
st2 = strat.copy()
for k, v in cnt.items():
    if k.startswith("H|"): st2[strat == k] = "H|" + k.split("|")[2]
    elif v < 60: st2[strat == k] = "small"
alloc.update({"H|in": 60, "H|out": 60, "small": 30})
rng_e = np.random.default_rng(12345)
ev_w, ev_s = [], []
for k, n in alloc.items():
    idx = np.flatnonzero(st2 == k)
    pick = rng_e.choice(idx, size=min(n, len(idx)), replace=False)
    ev_w += univ[pick].tolist(); ev_s += [k] * len(pick)
Nst = collections.Counter(st2.tolist()); nst = collections.Counter(ev_s)
assert set(Nst) == set(alloc), set(Nst) ^ set(alloc)
ev_w = np.array(ev_w, np.int64)
detE = Fm._details(ctx, ev_w.tolist())
def text_any(w):
    d = detE.get(int(w))
    if d: return J.paper_text(d.get("title"), d.get("abstract"))
    nd = int(ctx.cit.nodes([int(w)])[0]); return J.paper_text(ctx.cit.title(nd) if nd >= 0 else "", None)
ev_texts = [text_any(w) for w in ev_w]
tE = realT(4.0); t = time.time()
ev_y = tE.label(ev_texts, "habitat fragmentation", None)
print(len(ev_w), "labelled in", round(time.time() - t, 1), "s, usd", round(tE.usd, 3), "failed", tE.failed, "slowest", tE.slowest)
EV = {"ids": ev_w.tolist(), "stratum": ev_s, "y": [None if y != y else float(y) for y in ev_y],
      "N": dict(Nst), "n": dict(nst), "note": "Kimi K3 labels of a stratified sample of the v3 candidate universe (habitat fragmentation)"}
out = Path("/home/maxime/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher/eval_population.json")
out.write_text(json.dumps(EV))
