D6 = "/mnt/fast/scholarsreadinglist/search/fields-test6/1a0c3446c53e7f1da081def4"
f6 = json.load(open(D6 + "/field.json")); Z = np.load(D6 + "/work/scores.npz")
univ, final, before, cut6 = Z["ids"], Z["final"], Z["before_check"], float(Z["cut"])
tl = {int(w): v for w, v in f6["labels"]["teacher"].items()}
nodes = ctx.cit.nodes(univ); cites = np.where(nodes >= 0, np.asarray(ctx.cit.cites)[np.maximum(nodes, 0)], 0).astype(int)
v5 = set(int(w) for w in f6["papers"]); pre = set(univ[before >= cut6].tolist())
lm = [int(l.strip()[1:]) for l in open("/tmp/r2/landmark_ids.txt")]
fin_by = dict(zip(univ.tolist(), final.tolist())); pre_by = dict(zip(univ.tolist(), before.tolist())); cit_by = dict(zip(univ.tolist(), cites.tolist()))
print("landmarks among candidates:", sum(w in fin_by for w in lm), "of", len(lm), "| in map: before Jev", sum(w in pre for w in lm), " after Jev", sum(w in v5 for w in lm))
for w in lm:
    if w not in v5:
        n = int(ctx.cit.nodes([w])[0]); print(f"   missed: score {pre_by.get(w, float('nan')):.2f} cites {cit_by.get(w, '-')!s:>5}  {ctx.cit.title(n)[:80] if n >= 0 else w}")
for c in (5, 10, 20, 50): print(f"candidates cited >= {c}: {(cites >= c).sum():,}; in band 0.1-0.7: {((cites >= c) & (before >= .1) & (before < .7)).sum():,}")
# population yardstick
def fb(p): return 0 if p < .02 else 1 if p < .1 else 2 if p < cut6 else 3 if p < .7 else 4 if p < .95 else 5
bins = np.array([fb(p) for p in before]); hc = (cites >= 10).astype(int)
alloc = {0: 100, 1: 150, 2: 220, 3: 200, 4: 160, 5: 140}
rng6 = np.random.default_rng(6); P_ids, P_st = [], []
for b, n_ in alloc.items():
    for h in (0, 1):
        idx = np.flatnonzero((bins == b) & (hc == h) & ~np.isin(univ, list(tl)))
        k_ = min(len(idx), n_ // 2); pk = rng6.choice(idx, size=k_, replace=False); P_ids += univ[pk].tolist(); P_st += [f"{b}|{h}"] * k_
PN = collections.Counter(f"{b}|{h}" for b, h, w in zip(bins, hc, univ.tolist()) if w not in tl); Pn = collections.Counter(P_st)
detp = Fm._details(ctx, P_ids)
def pair2(w):
    d = detp.get(int(w))
    if d: return d.get("title"), d.get("abstract")
    n = int(ctx.cit.nodes([int(w)])[0]); return (ctx.cit.title(n) if n >= 0 else ""), None
P_pairs = [pair2(w) for w in P_ids]
tK = J.Teacher(2.0); P_y = tK.label([J.paper_text(*p) for p in P_pairs], TOPIC, DEF)
jv = J.Jev(0.5); P_j = jv.label(P_pairs, TOPIC, DEF)
print(len(P_ids), "yardstick papers: Kimi $", round(tK.usd, 3), " Jev $", round(jv.usd, 3), " Jev agrees with Kimi", round(float(np.mean((P_j >= .5) == (P_y >= .5))), 3))
Path("/home/maxime/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher/eval_population_llm_posttraining.json").write_text(json.dumps(
    {"ids": P_ids, "stratum": P_st, "y": [None if y != y else float(y) for y in P_y], "jev": [None if y != y else float(y) for y in P_j], "N": dict(PN), "n": dict(Pn)}))
for s_ in sorted(PN): print(f"  {s_} N={PN[s_]:7,d} n={Pn[s_]:3d} Kimi on topic {np.nanmean(P_y[np.array(P_st) == s_]):.2f}")
