_dec0 = decide
for jt in (0.3, 0.5, 0.7):
    def decide(w, lo, hi, c, cut, jt=jt):
        f = fin_by.get(w)
        if f is None: return False, False
        if w not in teach and lo <= f < hi and cit_by[w] >= c:
            j = JEV.get(w); return (j is not None and j == j and j >= jt), True
        return f >= cut, False
    for c in (10, 20, 30, 50):
        run(f"Jev>={jt} [0.1,0.7) cites>={c} cut {cut4:.2f}", 0.1, 0.7, c, cut4)
decide = _dec0
