# final score over the whole universe: teacher answer > decider (10k pass) > students
fin = stud.copy()
for i, w in enumerate(univ.tolist()):
    if w in tl400: fin[i] = tl400[w]
    elif w in decided: fin[i] = decided[w]
fin = np.nan_to_num(fin, nan=0.0)
o = np.sort(fin)[::-1]; cs = np.cumsum(o); P = o.sum(); k = np.arange(1, len(o) + 1)
f1 = 2 * cs / (k + P); kb = int(np.argmax(f1))
print(f"expected on-topic papers among candidates (sum of probabilities): {P:,.0f}   (eval-set estimate: {est_pos:,.0f})")
print(f"plug-in F1 rule: cut {o[kb]:.3f}, {kb+1:,} papers, expected F1 {f1[kb]:.3f}")
be = int(np.argmin(np.abs(k - P))); print(f"break-even rule (as many papers as expected on topic): cut {o[be]:.3f}")
fin_by = dict(zip(univ.tolist(), fin.tolist()))
inU = np.array([w in fin_by for w in ref_pos]); print("meta-analysis on-topic refs inside the candidates:", round(inU.mean(), 3))
for t in (0.4, o[kb], 0.5, 0.6, 0.7):
    chosen = {w for w, v in fin_by.items() if v >= t}
    evaluate(chosen, f"cut {t:.2f}", boot=200)
