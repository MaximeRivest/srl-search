dec_ev_act = res_al["act"]
ev_idx = {int(w): i for i, w in enumerate(ev_ids)}
def final_scores(K, cut=0.5, mode="decider"):
    """score per eval paper: teacher > decider (10k pass, plus round 2 on the K papers nearest the cut) > students"""
    cand = np.array([i for i, w in enumerate(univ.tolist()) if w not in tl400 and w not in decided])
    unc = np.abs(lg(stud[cand]) - lg(np.array(cut)))
    r2 = set(univ[cand[np.argsort(unc, kind="stable")[:K]]].tolist()) if K else set()
    out = np.zeros(len(ev_ids)); src_ = []
    for i, w in enumerate(ev_ids.tolist()):
        s_ = st_ev[i]
        if w in tl400: out[i] = tl400[w]; src_.append("teacher"); continue
        dv = decided.get(w, dec_ev_act[i] if w in r2 else None)
        if dv is None: out[i] = s_; src_.append("students"); continue
        out[i] = dv if mode == "decider" else sg((lg(dv) + lg(s_)) / 2); src_.append("decider")
    return out, collections.Counter(src_), r2
for K in (0, 10000, 20000, 40000):
    for mode in ("decider", "avg"):
        sc, c, r2 = final_scores(K, mode=mode)
        curve(sc, f"K={K//1000}k {mode}", ts=(.2, .3, .4, .5, .6, .7, .8, .9))
    print("   sources on eval:", dict(c))
