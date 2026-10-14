nodes = ctx.cit.nodes(univ); cites = np.where(nodes >= 0, np.asarray(ctx.cit.cites)[np.maximum(nodes, 0)], 0).astype(int)
sel_ = final >= cut4
for c in (50, 100, 200, 500, 1000):
    m = cites >= c
    print(f"cites>={c:5d}: candidates {m.sum():7,d}  with score>=0.01 {(m & (final >= .01)).sum():6,d}  selected {(m & sel_).sum():6,d}  near the cut (0.01..cut) {(m & (final >= .01) & ~sel_).sum():6,d}")
print("selected papers' citations: median", int(np.median(cites[sel_])), " 90th pct", int(np.percentile(cites[sel_], 90)), " 99th", int(np.percentile(cites[sel_], 99)))
