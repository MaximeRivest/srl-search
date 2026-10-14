from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
dec_ids = [int(w) for w in f6["labels"]["decider"]]
teach_ids = dec_ids + list(tl)
dt_ = Fm._details(ctx, teach_ids)
def pr_(w, det):
    d = det.get(int(w))
    if d: return d.get("title"), d.get("abstract")
    n = int(ctx.cit.nodes([int(w)])[0]); return (ctx.cit.title(n) if n >= 0 else ""), None
jt = J.Jev(1.0); t = time.time(); jy = jt.label([pr_(w, dt_) for w in teach_ids], TOPIC, DEF)
print(len(teach_ids), "Jev labels in", round(time.time() - t), "s $", round(jt.usd, 3), "failed", jt.failed)
JL = {w: float(v) for w, v in zip(teach_ids, jy) if v == v}
print("Jev on Kimi's 400: agree", round(np.mean([(JL[w] >= .5) == (tl[w] >= .5) for w in tl if w in JL]), 3), "| decider (0.8B) agreement with Jev on its 8,719:",
      round(np.mean([(JL[w] >= .5) == (f6["labels"]["decider"][str(w)] >= .5) for w in dec_ids if w in JL]), 3))
# students learn Jev's probabilities (the same recipe as v5's students learn the decider's)
X, h = Fm._vectors(ctx, list(JL)); p_ = np.array(list(JL.values()))
lrj = LogisticRegression(C=2.0, max_iter=4000).fit(np.concatenate([X[h], X[h]]), np.r_[np.ones(h.sum()), np.zeros(h.sum())], sample_weight=np.r_[p_[h], 1 - p_[h]])
Vu, hu = Fm._vectors(ctx, univ); pv = np.full(len(univ), np.nan); pv[hu] = lrj.predict_proba(Vu[hu])[:, 1]; del Vu
is_t = np.isin(univ, list(tl)); em = np.isnan(pv) | ((pv > .1) & (pv < .9)) | is_t | np.isin(univ, P_ids_a)
E = univ[em]; tw = list(JL); dE = Fm._details(ctx, E.tolist()); dT = Fm._details(ctx, tw)
txt = lambda w, d: J.paper_text(*pr_(w, d))
t = time.time(); pe = J.student_scores([(txt(w, dT), JL[w]) for w in tw], [txt(w, dE) for w in E], Path(D6) / "work" / "student_jev", print)
print("text student on", len(E), "papers", round(time.time() - t), "s")
pt = np.full(len(univ), np.nan); pt[em] = pe
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))); sg = lambda z: 1 / (1 + np.exp(-z))
both = ~np.isnan(pv) & ~np.isnan(pt)
stud = np.nan_to_num(np.where(both, sg((lg(np.nan_to_num(pv, nan=.5)) + lg(np.nan_to_num(pt, nan=.5))) / 2), np.where(np.isnan(pv), pt, pv)))
pos = {w: i for i, w in enumerate(univ.tolist())}
s1 = stud.copy(); fixed = np.zeros(len(univ), bool)
for w, v in JL.items():
    if w in pos and w not in tl: s1[pos[w]] = v; fixed[pos[w]] = True
feats = Fm._citation_features(ctx.cit, univ, s1, lg(stud))
tix = np.array([pos[w] for w in tl if w in pos]); ty = np.array([tl[int(univ[i])] >= .5 for i in tix])
comb = LogisticRegression(C=1.0, max_iter=2000).fit(feats[tix], ty); z = feats @ comb.coef_[0] + comb.intercept_[0]
free = ~fixed & ~is_t; shift = brentq(lambda s: sg(z[free] + s).sum() - stud[free].sum(), -20, 20)
fj = np.where(fixed, s1, sg(z + shift))
for w, v in tl.items():
    if w in pos: fj[pos[w]] = v
cutj, ef = Fm._plugin_cut(fj)
print(f"expected on topic {fj.sum():,.0f} (Kimi yardstick estimate {sum(PN[s] * np.nanmean(P_y[P_st_a == s]) for s in PN):,.0f}); cut {cutj}")
fj_by = dict(zip(univ.tolist(), fj.tolist()))
for name, ch in (("v5 run", v5), ("Jev teaches, cut %.2f" % cutj, {w for w, v in fj_by.items() if v >= cutj}), ("Jev teaches, cut 0.5", {w for w, v in fj_by.items() if v >= .5})):
    p, r = pr6(ch); print(f"{name:22s} papers {len(ch):6,d}  population P {p:.2f} R {r:.2f}  landmarks {sum(w in ch for w in lm)}/44")
