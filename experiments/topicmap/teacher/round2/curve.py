def pr_at(score, t):
    sel = (score >= t).astype(float); idx = np.flatnonzero(ok_e & ~np.isnan(score))
    return _pr(sel, idx)
def curve(score, name, ts=(.05, .1, .2, .3, .5, .7, .9, .95, .98, .99, .995)):
    rows = []
    for t in ts:
        p, r = pr_at(score, t); rows.append((t, p, r, 2 * p * r / max(1e-9, p + r)))
    best = max(rows, key=lambda x: x[3])
    print(f"{name:26s} " + "  ".join(f"{t:g}:{p:.2f}/{r:.2f}" for t, p, r, f in rows) + f"   best F1 {best[3]:.3f} at {best[0]:g}")
curve(d_ev, "decider judges all")
