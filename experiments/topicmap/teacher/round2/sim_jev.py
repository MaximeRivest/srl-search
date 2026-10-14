fin_by = dict(zip(univ.tolist(), final.tolist())); cit_by = dict(zip(univ.tolist(), cites.tolist()))
teach = set(t4)
ev_st = np.array(EV["stratum"]); w_e = np.array([EV["N"][s] / EV["n"][s] for s in ev_st]); ok_e = ~np.isnan(ev_y)
cl_ids_a = np.array(CL["ids"]); cl_st_a = np.array(CL["stratum"]); w_c = np.array([CL["N"][s] / CL["n"][s] for s in cl_st_a]); ok_c = ~np.isnan(cy)
pjc = 0.079 / 2903 * 1000
def decide(w, lo, hi, c, cut):
    f = fin_by.get(w)
    if f is None: return False, False
    if w not in teach and lo <= f < hi and cit_by[w] >= c:
        j = JEV.get(w); return (j is not None and j == j and j >= .5), True
    return f >= cut, False
def wpr(ids, y, w, ok, rule):
    s = np.array([decide(int(x), *rule)[0] for x in ids], float)
    tp = (w * s * y)[ok].sum(); return tp / max(1e-9, (w * s)[ok].sum()), tp / max(1e-9, (w * y)[ok].sum())
def run(name, lo, hi, c, cut):
    rule = (lo, hi, c, cut)
    S = int(sum(1 for w, f in fin_by.items() if w not in teach and lo <= f < hi and cit_by[w] >= c)) if hi > lo else 0
    sel_n = sum(decide(w, *rule)[0] for w in fin_by)  # papers in the map
    p, r = wpr(ev_ids, np.nan_to_num(ev_y), w_e, ok_e, rule); pc, rc = wpr(cl_ids_a, np.nan_to_num(cy), w_c, ok_c, rule)
    mr = np.array([decide(w, *rule)[0] for w in ref_pos])
    print(f"{name:34s} Jev checks {S:6,d} (${S * pjc / 1000:.2f})  map {sel_n:6,d}  population P {p:.2f} R {r:.2f}  cited>=50 P {pc:.2f} R {rc:.2f}  meta {mr.mean():.2f} (title only {mr[~ref_ab].mean():.2f})")
run("round 2 as it is", 0, 0, 0, cut4)
run("round 2, strict cut 0.6", 0, 0, 0, 0.6)
for c in (20, 50, 100):
    for lo, hi in ((0.1, 0.7), (0.05, 0.9), (0.02, 0.95)):
        for cut in (cut4, 0.6):
            run(f"Jev [{lo},{hi}) cites>={c} cut {cut:.2f}", lo, hi, c, cut)
