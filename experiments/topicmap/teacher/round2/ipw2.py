r1 = list(tl)[:272]
assert all(w in tl for w in r1)
Xt, ht = Fm._vectors(ctx, r1); lr0 = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(Xt[ht], np.array([tl[w] >= .5 for w in r1])[ht])
p0u = np.full(len(univ), np.nan); p0u[hu] = lr0.predict_proba(Vu[hu])[:, 1]
U = route == 0
print("likely by first classifier:", int((U & (p0u >= .5)).sum()), "(run said", f6["report"]["candidates"]["likely_by_first_classifier"], ")")
dset = set(dec_ids)
# the run's take(): sequential, each stratum minus what is already taken; reconstruct each decided paper's stratum
seq = [(U & (p0u >= .5), 3000), (U & (p0u > .1) & (p0u < .9), 2500), (U, 1500), (Cc & ~hu, 2000), (Cc & hu, 1000)]
taken = np.isin(univ, list(tl)); pi = np.zeros(len(univ))
for m, k in seq:
    avail = m & ~taken; got = avail & np.isin(univ, dec_ids)
    pi[avail] = 1 - (1 - pi[avail]) * (1 - got.sum() / max(1, avail.sum())) if False else np.maximum(pi[avail], got.sum() / max(1, avail.sum()))
    print(f"  stratum: available {avail.sum():7,d}  decided {got.sum():5,d}")
    taken |= got
ii = np.array([pos[w] for w in dec_ids]); print("decided papers with no stratum:", int((pi[ii] == 0).sum()))
for nm, lab in (("decider", dec_p), ("Jev", JL)):
    yy = np.array([lab.get(w, np.nan) for w in dec_ids]); ok_ = ~np.isnan(yy) & (pi[ii] > 0)
    print(f"weighted estimate of on-topic papers, {nm}'s labels: {(yy[ok_] / pi[ii][ok_]).sum() + sum(v >= .5 for v in tl.values()):,.0f}  (Kimi yardstick {est(ky6):,.0f})")
