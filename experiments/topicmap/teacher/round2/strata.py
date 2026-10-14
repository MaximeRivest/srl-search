import collections
U_w, p0, H, C, hC, lr0 = S["U_w"], S["p0"], S["H"], S["C"], S["hC"], S["lr0"]
inv3 = set(int(w) for w in f3["papers"])
XC, _ = Fm._vectors(ctx, C[hC]); pC0 = np.full(len(C), np.nan); pC0[hC] = lr0.predict_proba(XC)[:, 1]
_, hH = Fm._vectors(ctx, H); XH, _ = Fm._vectors(ctx, H[hH]); pH0 = np.full(len(H), np.nan); pH0[hH] = lr0.predict_proba(XH)[:, 1]
univ = np.concatenate([U_w, H, C]); src = np.array(["U"] * len(U_w) + ["H"] * len(H) + ["C"] * len(C))
s0 = np.concatenate([p0, pH0, pC0])
print("universe", len(univ), "unique", len(np.unique(univ)))
def binof(p):
    if p != p: return "title"
    return "0.9+" if p >= .9 else ".5-.9" if p >= .5 else ".2-.5" if p >= .2 else ".05-.2" if p >= .05 else "<.05"
inv = np.array([int(w) in inv3 for w in univ])
strat = np.array([f"{s}|{binof(p)}|{'in' if i else 'out'}" for s, p, i in zip(src, s0, inv)])
cnt = collections.Counter(strat.tolist())
for k in sorted(cnt): print(f"{k:22s} {cnt[k]:8,d}")
print("v3 field in universe:", int(inv.sum()), "of", len(inv3))
