route = Z["route"]
Xt, ht = Fm._vectors(ctx, list(tl)); lr0 = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(Xt[ht], np.array([tl[w] >= .5 for w in tl])[ht])
Vu, hu = Fm._vectors(ctx, univ); p0u = np.full(len(univ), np.nan); p0u[hu] = lr0.predict_proba(Vu[hu])[:, 1]
U = route == 0; Cc = route == 2
strata = [(U & (p0u >= .5), 3000), (U & (p0u > .1) & (p0u < .9), 2500), (U, 1500), (Cc & ~hu, 2000), (Cc & hu, 1000)]
pi = 1 - np.prod([np.where(m, 1 - min(1, k / max(1, m.sum())), 1.0) for m, k in strata], axis=0)
pi = np.maximum(pi, 1e-4)
print("inclusion probabilities of the decider's papers: median %.4f, range %.4f-%.3f" % (np.median(pi[np.isin(univ, dec_ids)]), pi[np.isin(univ, dec_ids)].min(), pi[np.isin(univ, dec_ids)].max()))
dec_p = {int(w): v for w, v in f6["labels"]["decider"].items()}
for nm, lab in (("decider", dec_p), ("Jev", {w: JL[w] for w in dec_ids if w in JL})):
    ii = np.array([pos[w] for w in lab]); yy = np.array(list(lab.values()))
    print(f"estimated on-topic papers in the pool from the {nm}'s labels (weighted): {(yy / pi[ii]).sum():,.0f}   unweighted students: {'%.0f' % Z['student'].sum() if nm == 'decider' else '%.0f' % stud.sum()}")
def weighted_students(lab, tag):
    ws_ = list(lab); ii = np.array([pos[w] for w in ws_]); p_ = np.array([lab[w] for w in ws_]); wt = 1 / pi[ii]; wt = wt / wt.mean()
    X, h = Fm._vectors(ctx, ws_)
    m = LogisticRegression(C=2.0, max_iter=4000).fit(np.concatenate([X[h], X[h]]), np.r_[np.ones(h.sum()), np.zeros(h.sum())], sample_weight=np.r_[p_[h] * wt[h], (1 - p_[h]) * wt[h]])
    pv_ = np.full(len(univ), np.nan); pv_[hu] = m.predict_proba(Vu[hu])[:, 1]
    print(f"  {tag}: weighted vector student expects {np.nansum(pv_):,.0f} on topic")
    return pv_
pv_dec = weighted_students(dec_p, "decider-taught")
pv_jev = weighted_students({w: JL[w] for w in dec_ids if w in JL}, "Jev-taught")
