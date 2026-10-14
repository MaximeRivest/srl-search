base_w = list(train_labels); base_tx = texts_any(base_w); base_y = [train_labels[w] for w in base_w]
res_al = {}
for arm in ("act", "spr"):
    ws, ys = AL[arm], AL["y_" + arm]
    rows = list(zip(base_tx, base_y)) + [(AL["tx"][w], y) for w, y in zip(ws, ys) if y == y]
    d = W / f"decider_{arm}"; t = time.time()
    J.train_decider(rows, "habitat fragmentation", None, d, print); tt = time.time() - t
    sc = J.decider_scores(d, ev_texts, print); res_al[arm] = sc
    print(arm, len(rows), "rows; train", round(tt), "s; AUC", wauc(sc), "title-only", wauc(sc, np.array([t.endswith("(no abstract)") for t in ev_texts])), flush=True)
    curve(sc, "decider " + arm)
print("baseline decider (272)    AUC", wauc(d_ev)); curve(d_ev, "decider 272")
