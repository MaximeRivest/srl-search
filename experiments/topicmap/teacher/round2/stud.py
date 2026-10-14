from sklearn.linear_model import LogisticRegression
decided3 = {int(w): v for w, v in f3["labels"]["decider"].items()}
train_labels = S["train_labels"]
def lr_soft(soft: dict, C=2.0):
    ws = np.array(list(soft), np.int64); X, h = Fm._vectors(ctx, ws); X = X[h]; q = np.array([soft[int(w)] for w in ws])[h]
    XX = np.concatenate([X, X]); yy = np.r_[np.ones(len(X)), np.zeros(len(X))]; sw = np.r_[q, 1 - q]
    return LogisticRegression(C=C, max_iter=4000).fit(XX, yy, sample_weight=sw)
soft = {**decided3, **train_labels}
lrs = lr_soft(soft)
Xe, he = Fm._vectors(ctx, ev_ids); v_ev = np.full(len(ev_ids), np.nan); v_ev[he] = lrs.predict_proba(Xe[he])[:, 1]
print("vector student AUC (has vector):", wauc(v_ev), " decider on same papers:", wauc(np.where(he, d_ev, np.nan)))
curve(v_ev, "vector student (soft)")
curve(np.where(he, d_ev, np.nan), "decider, same papers")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for a in (0.5, 0.7):
    ens = 1 / (1 + np.exp(-(a * lg(d_ev) + (1 - a) * lg(v_ev))))
    print("ensemble a=%.1f AUC" % a, wauc(ens)); curve(ens, f"ensemble {a}")
