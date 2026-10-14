cut = 0.6
miss = [w for w in ref_pos if w in fin_by and fin_by[w] < cut]
found_ = [w for w in ref_pos if w in fin_by and fin_by[w] >= cut]
print("on-topic refs in candidates:", len(miss) + len(found_), "missed at", cut, ":", len(miss))
ms = np.array([fin_by[w] for w in miss]); print("their scores: <0.1:", (ms < .1).sum(), " 0.1-0.4:", ((ms >= .1) & (ms < .4)).sum(), " 0.4-0.6:", (ms >= .4).sum())
src_of = {}
for w in miss: src_of[w] = "U" if w in set(U_w.tolist()) else "H" if w in set(H.tolist()) else "C"
print(collections.Counter(src_of.values()))
rtext = {int(r["id"][1:]): r["text"] for r in refs}
mt = texts_any(miss)
same = sum(a.endswith("(no abstract)") == b.endswith("(no abstract)") for a, b in zip(mt, [rtext[w] for w in miss]))
print("pipeline sees title-only:", sum(t.endswith("(no abstract)") for t in mt), "of", len(mt), "; label-time text title-only:", sum(rtext[w].endswith("(no abstract)") for w in miss))
for w, t in list(zip(miss, mt))[:25]: print(f"  {fin_by[w]:.2f} {src_of[w]} {t.splitlines()[0][7:120]}")
