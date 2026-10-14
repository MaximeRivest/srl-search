# students taught by the decider only (teacher answers unseen by them), then the combiner fitted on the 400 teacher answers
lr1b = lr_soft(dict(decided))
pv_b = np.full(len(univ), np.nan); pv_b[:len(U_w)] = lr1b.predict_proba(S["U_V"])[:, 1]
for off, ws, hh in ((len(U_w), H, hH), (len(U_w) + len(H), C, hC)):
    X, _ = Fm._vectors(ctx, ws[hh]); tmp = np.full(len(ws), np.nan); tmp[hh] = lr1b.predict_proba(X)[:, 1]; pv_b[off:off + len(ws)] = tmp
Eb_mask = np.isnan(pv_b) | ((pv_b > BAND2[0]) & (pv_b < BAND2[1])) | np.isin(univ, ev_ids) | np.isin(univ, np.array(list(tl400), np.int64))
Eb = univ[Eb_mask]; dw = list(decided)
t = time.time(); peb = J.student_scores(list(zip(texts_any(dw), [decided[w] for w in dw])), texts_any(Eb), W / "student_dec_only", print)
print("text student (decider labels only):", round(time.time() - t), "s on", len(Eb))
ptb = np.full(len(univ), np.nan); ptb[Eb_mask] = peb
bb = ~np.isnan(pv_b) & ~np.isnan(ptb)
stud_b = np.where(bb, sg((lg(np.nan_to_num(pv_b, nan=.5)) + lg(np.nan_to_num(ptb, nan=.5))) / 2), np.where(np.isnan(pv_b), ptb, pv_b))
print("students AUC: with teacher rows", wauc(stud[ev_u]), " decider labels only", wauc(stud_b[ev_u]))
fin_b = stud_b.copy()
for w, v in decided.items():
    if w in pos_of: fin_b[pos_of[w]] = v
fc_b, tot_b, in_b, out_b, nref_b = cite_features(np.nan_to_num(fin_b))
def feats_b(ix):
    return np.column_stack([lg(np.clip(stud_b[ix], 1e-4, 1 - 1e-4)), np.log1p(fc_b[ix]), in_b[ix], out_b[ix], (nref_b[ix] == 0).astype(float)])
tw_ = [w for w in tl400 if w in pos_of]; tix = np.array([pos_of[w] for w in tw_]); ty = np.array([tl400[w] >= .5 for w in tw_])
for Cc in (0.3, 1.0):
    stk = LogisticRegression(C=Cc, max_iter=2000).fit(feats_b(tix), ty)
    print("C", Cc, "coefs:", np.round(stk.coef_[0], 2), "b", round(stk.intercept_[0], 2))
    comb = stk.predict_proba(feats_b(np.arange(len(univ))))[:, 1]
    fin3 = comb.copy()
    for w, v in decided.items():
        if w in pos_of: fin3[pos_of[w]] = v
    for w, v in tl400.items():
        if w in pos_of: fin3[pos_of[w]] = v
    fb3 = dict(zip(univ.tolist(), fin3.tolist()))
    o3 = np.sort(fin3)[::-1]; cs3 = np.cumsum(o3); k3 = np.arange(1, len(o3) + 1); kb3 = int(np.argmax(2 * cs3 / (k3 + o3.sum())))
    print(f"   expected on topic {o3.sum():,.0f}; plug-in cut {o3[kb3]:.2f}")
    for t in (0.4, 0.5, 0.6, round(float(o3[kb3]), 2)):
        evaluate({w for w, v in fb3.items() if v >= t}, f"   C={Cc} cut {t}", boot=100)
fb_s = dict(zip(univ.tolist(), np.where(np.isin(univ, np.array(list(tl400), np.int64)), [tl400.get(w, 0) for w in univ.tolist()], fin_b).tolist()))
for t in (0.5, 0.6): evaluate({w for w, v in fb_s.items() if v >= t}, f"   no combiner cut {t}", boot=100)
