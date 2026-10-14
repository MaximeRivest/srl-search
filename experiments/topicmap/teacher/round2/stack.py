def feats(ix):
    return np.column_stack([lg(np.clip(stud[ix], 1e-4, 1 - 1e-4)), np.log1p(fc[ix]), in_share[ix], out_share[ix], (nref[ix] == 0).astype(float)])
clean = {int(w): y for w, y in zip(AL["spr"], AL["y_spr"]) if y == y}
clean.update({int(w): labels_v for w, labels_v in S["labels"].items() if w in S["hold"]})
cw_ = [w for w in clean if w in pos_of]; cix = np.array([pos_of[w] for w in cw_]); cy = np.array([clean[w] >= .5 for w in cw_])
print("clean labels for the combiner:", len(cw_), "on topic", cy.sum())
for Cc in (0.3, 1.0):
    stk = LogisticRegression(C=Cc, max_iter=2000).fit(feats(cix), cy)
    print("C", Cc, "coefs (logit, log fc, in_share, out_share, no refs):", np.round(stk.coef_[0], 2), "b", round(stk.intercept_[0], 2))
    comb = stk.predict_proba(feats(np.arange(len(univ))))[:, 1]
    # teacher / decider answers keep priority, as before
    fin2 = comb.copy()
    for w, v in tl400.items():
        if w in pos_of: fin2[pos_of[w]] = v
    for w, v in decided.items():
        if w in pos_of: fin2[pos_of[w]] = max(v, comb[pos_of[w]]) if False else v
    print("   combiner AUC on eval", wauc(fin2[ev_u]), "(students:", wauc(fin[ev_u]), ")")
    fb2 = dict(zip(univ.tolist(), fin2.tolist()))
    for t in (0.4, 0.5, 0.6):
        evaluate({w for w, v in fb2.items() if v >= t}, f"   C={Cc} cut {t}", boot=100)
