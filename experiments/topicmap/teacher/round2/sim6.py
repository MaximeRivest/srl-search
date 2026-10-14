Pj_by = dict(zip(P_ids, P_j.tolist()))
lm_j = J.Jev(0.1).label([pair2(w) if w in detp else ((Fm._details(ctx, [w]).get(w) or {}).get("title"), (Fm._details(ctx, [w]).get(w) or {}).get("abstract")) for w in lm], TOPIC, DEF)
lmj_by = dict(zip(lm, lm_j.tolist()))
print("Jev says yes to", int((lm_j >= .5).sum()), "of 44 landmarks")
pjc6 = 0.03 / 970
def sim(lo, hi, c, cut=cut6):
    def dec(w):
        f = pre_by.get(w)
        if f is None: return False
        if w in tl: return tl[w] >= .5
        if lo <= f < hi and cit_by[w] >= c:
            j = Pj_by.get(w, lmj_by.get(w)); return j is not None and j == j and j >= .5
        return f >= cut
    S = int(((before >= lo) & (before < hi) & (cites >= c)).sum())
    ch_y = {w for w in P_ids if dec(w)}
    p, r = pr6(ch_y)
    n_map = int(((before >= cut) & ~((before >= lo) & (before < hi) & (cites >= c))).sum()) + 0
    lmk = sum(dec(w) for w in lm)
    print(f"Jev [{lo},{hi}) cites>={c:3d}: checks {S:6,d} (${S * 0.027 / 1000:.2f})  population P {p:.2f} R {r:.2f}  landmarks {lmk}/44")
sim(0.1, 0.7, 20)
for c in (0, 5, 20):
    for lo, hi in ((0.1, 0.7), (0.1, 0.95), (0.1, 1.01), (0.2, 1.01)):
        sim(lo, hi, c)
