pre_y = np.array([pre_by[int(w)] for w in P_ids])          # each yardstick paper's score before Jev's check
jyes = (np.nan_to_num(P_j) >= .5).astype(float)
res = []
for seed in range(6):
    rs = np.random.default_rng(100 + seed); half = rs.random(len(P_ids)) < .5
    for fit_on, test_on in ((half, ~half), (~half, half)):
        f_ = fit_on & ok6
        cal = LogisticRegression(C=10.0, max_iter=1000).fit(lg(pre_y[f_]).reshape(-1, 1), jyes[f_], sample_weight=w6[f_])
        a, b = cal.coef_[0][0], cal.intercept_[0]
        fc = sg(a * lg(before) + b)
        for w, v in tl.items():
            if w in pos: fc[pos[w]] = v
        cutc, _ = Fm._plugin_cut(fc)
        sel = np.array([sg(a * lg(pre_by[int(w)]) + b) >= cutc for w in P_ids], float)
        t_ = test_on & ok6; tp = (w6 * sel * y6)[t_].sum()
        res.append((tp / (w6 * sel)[t_].sum(), tp / (w6 * y6)[t_].sum(), fc.sum(), (fc >= cutc).sum(), cutc, a, b))
r_ = np.array(res)
print(f"calibrated by Jev (half fit, half test, 12 splits): P {r_[:,0].mean():.2f} ± {r_[:,0].std():.2f}  R {r_[:,1].mean():.2f} ± {r_[:,1].std():.2f}  expected on topic {r_[:,2].mean():,.0f}  map {r_[:,3].mean():,.0f} papers  (a {r_[:,5].mean():.2f}, b {r_[:,6].mean():.2f})")
selv5 = np.array([w in v5 for w in P_ids], float); tp = (w6 * selv5 * y6)[ok6].sum()
print(f"v5 run for comparison: P {tp / (w6 * selv5)[ok6].sum():.2f}  R {tp / (w6 * y6)[ok6].sum():.2f}")
a, b = r_[:, 5].mean(), r_[:, 6].mean(); fc = sg(a * lg(before) + b); cutc, _ = Fm._plugin_cut(fc)
print("landmarks with calibration:", sum(sg(a * lg(pre_by[w]) + b) >= cutc for w in lm if w in pre_by), "/44")
