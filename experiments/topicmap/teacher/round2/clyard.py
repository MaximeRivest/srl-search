rngc = np.random.default_rng(2026)
def fbin(p):
    return 0 if p < .01 else 1 if p < .1 else 2 if p < cut4 else 3 if p < .7 else 4 if p < .95 else 5
cb = np.array([fbin(p) for p in final]); hb = (cites >= 200).astype(int)
mask = cites >= 50
alloc_c = {0: 60, 1: 120, 2: 180, 3: 140, 4: 120, 5: 100}
cl_ids, cl_st = [], []
for b, n_ in alloc_c.items():
    for h in (0, 1):
        idx = np.flatnonzero(mask & (cb == b) & (hb == h))
        k_ = min(len(idx), n_ // 2); pick = rngc.choice(idx, size=k_, replace=False)
        cl_ids += univ[pick].tolist(); cl_st += [f"{b}|{h}"] * k_
CLN = collections.Counter([f"{b}|{h}" for b, h in zip(cb[mask], hb[mask])]); cln = collections.Counter(cl_st)
cl_ids = np.array(cl_ids, np.int64)
detc = Fm._details(ctx, cl_ids.tolist())
def pair(w, det):
    d = det.get(int(w))
    if d: return d.get("title"), d.get("abstract")
    nd = int(ctx.cit.nodes([int(w)])[0]); return (ctx.cit.title(nd) if nd >= 0 else ""), None
cl_pairs = [pair(w, detc) for w in cl_ids]
cl_texts = [J.paper_text(*p) for p in cl_pairs]
tK = J.Teacher(1.5); t = time.time(); cl_y = tK.label(cl_texts, "habitat fragmentation", None)
print(len(cl_ids), "Kimi", round(time.time() - t, 1), "s $", round(tK.usd, 3), "failed", tK.failed)
CL = {"ids": cl_ids.tolist(), "stratum": cl_st, "y": [None if y != y else float(y) for y in cl_y], "N": dict(CLN), "n": dict(cln),
      "note": "Kimi K3 labels of papers cited >= 50 times among selection v4's candidates (habitat fragmentation), stratified by v4 score and citations (>=200)"}
Path("/home/maxime/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher/eval_classics.json").write_text(json.dumps(CL))
for s_ in sorted(CLN): print(f"  {s_}  N={CLN[s_]:7,d}  n={cln[s_]:3d}  Kimi on topic {np.nanmean(cl_y[np.array(cl_st) == s_]):.2f}")
