from scipy.optimize import brentq
stk = LogisticRegression(C=1.0, max_iter=2000).fit(feats_b(tix), ty)
Fall = feats_b(np.arange(len(univ))); z = Fall @ stk.coef_[0] + stk.intercept_[0]
fixed = np.array([w in decided or w in tl400 for w in univ.tolist()])
target = np.nansum(stud_b[~fixed])                       # what the students expect among the papers the combiner judges
shift = brentq(lambda s: sg(z[~fixed] + s).sum() - target, -10, 10)
print(f"students expect {target:,.0f} on topic among the combiner's papers; intercept shift {shift:.2f}")
comb = sg(z + shift); fin4 = comb.copy()
for w, v in decided.items(): fin4[pos_of[w]] = v
for w, v in tl400.items():
    if w in pos_of: fin4[pos_of[w]] = v
o4 = np.sort(fin4)[::-1]; cs4 = np.cumsum(o4); k4 = np.arange(1, len(o4) + 1); f14 = 2 * cs4 / (k4 + o4.sum()); kb4 = int(np.argmax(f14))
print(f"expected on topic {o4.sum():,.0f}; plug-in cut {o4[kb4]:.2f} (expected F1 {f14[kb4]:.3f})")
fb4 = dict(zip(univ.tolist(), fin4.tolist()))
for t in (round(float(o4[kb4]), 2), 0.5):
    evaluate({w for w, v in fb4.items() if v >= t}, f"combiner+shift cut {t}", boot=200)
