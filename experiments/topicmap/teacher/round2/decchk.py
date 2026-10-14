t = time.time(); P_d = J.decider_scores(Path(D6) / "work" / "decider", [J.paper_text(*p) for p in P_pairs], print); el = time.time() - t
print(f"decider on {len(P_ids)} papers: {el:.0f} s (incl. ~40 s start-up)")
Pd_by = dict(zip(P_ids, P_d.tolist()))
for nm, src in (("Jev", Pj_by), ("decider", Pd_by)):
    for lo in (0.1, 0.2):
        sel = np.array([(src[w] >= .5) if (pre_by[w] >= lo and w not in tl) else (pre_by[w] >= cut6) for w in P_ids], float)
        tp = (w6 * sel * y6)[ok6].sum(); n_ = int(((before >= lo)).sum())
        print(f"{nm:8s} checks every paper scoring >= {lo}: {n_:,} papers  P {tp / (w6 * sel)[ok6].sum():.2f}  R {tp / (w6 * y6)[ok6].sum():.2f}")
