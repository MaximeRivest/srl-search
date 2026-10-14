lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sg = lambda z: 1 / (1 + np.exp(-z))
taught = {**decided, **tl400}
t = time.time(); lr1 = lr_soft(taught); print("vector student", round(time.time() - t, 1), "s")
# vector student on every candidate with a vector
Xall_w = univ; pv = np.full(len(univ), np.nan)
hv = np.concatenate([np.ones(len(U_w), bool), hH, hC])
pv[:len(U_w)] = lr1.predict_proba(S["U_V"])[:, 1]
for off, ws, hh in ((len(U_w), H, hH), (len(U_w) + len(H), C, hC)):
    X, _ = Fm._vectors(ctx, ws[hh]); tmp = np.full(len(ws), np.nan); tmp[hh] = lr1.predict_proba(X)[:, 1]; pv[off:off + len(ws)] = tmp
BAND2 = (0.1, 0.9)
E_mask = np.isnan(pv) | ((pv > BAND2[0]) & (pv < BAND2[1]))
E_mask |= np.isin(univ, ev_ids)                      # the eval papers always get a text-student score (for measuring)
E = univ[E_mask]; print("text student judges", len(E), "papers")
t = time.time(); tw = list(taught)
pe = J.student_scores(list(zip(texts_any(tw), [taught[w] for w in tw])), texts_any(E), W / "student_soft", print)
print("text student", round(time.time() - t), "s")
pt = np.full(len(univ), np.nan); pt[E_mask] = pe
both_ = ~np.isnan(pv) & ~np.isnan(pt)
stud = np.where(both_, sg((lg(np.nan_to_num(pv, nan=.5)) + lg(np.nan_to_num(pt, nan=.5))) / 2), np.where(np.isnan(pv), pt, pv))
stud_by = dict(zip(univ.tolist(), stud.tolist()))
st_ev = np.array([stud_by[int(w)] for w in ev_ids]); pv_ev = np.array([pv[i] for i in [np.flatnonzero(univ == w)[0] for w in ev_ids]])
print("students AUC", wauc(st_ev), "title-only", wauc(st_ev, np.array([t.endswith("(no abstract)") for t in ev_texts])))
curve(st_ev, "students (soft)")
