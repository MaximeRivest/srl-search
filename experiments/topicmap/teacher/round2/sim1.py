# the 400-answer (active) decider labels the same 10k papers v3's decider labelled (v3's balanced take())
DA = W / "decider_act"
tl400 = {**train_labels, **{int(w): y for w, y in zip(AL["act"], AL["y_act"]) if y == y}}
dws = [w for w in decided3 if w not in tl400]
t = time.time(); dsc = J.decider_scores(DA, texts_any(dws), print); print("decider 10k:", round(time.time() - t), "s")
decided = dict(zip(dws, dsc.tolist()))
