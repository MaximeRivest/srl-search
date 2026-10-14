def texts_any(ws):
    det = Fm._details(ctx, [int(w) for w in ws]); out = []
    for w in ws:
        d = det.get(int(w))
        if d: out.append(J.paper_text(d.get("title"), d.get("abstract")))
        else:
            nd = int(ctx.cit.nodes([int(w)])[0]); out.append(J.paper_text(ctx.cit.title(nd) if nd >= 0 else "", None))
    return out
labelled = set(S["labels"]) | set(int(w) for w in ev_ids)
ra = np.random.default_rng(7)
def pick(cands, n):
    cands = np.array([w for w in np.asarray(cands).tolist() if w not in labelled], np.int64)
    out = ra.choice(cands, size=min(n, len(cands)), replace=False); labelled.update(out.tolist()); return out
# active: the first classifier's unsure band (vectors), and cited works with only a title
vec_w = np.concatenate([U_w, H, C[hC]]); vec_p = np.concatenate([p0, pH0, pC0[hC]])
band = vec_w[(vec_p > 0.35) & (vec_p < 0.85)]
print("unsure band:", len(band))
act = np.concatenate([pick(band, 96), pick(C[~hC], 32)])
# spread: the stage-0 design again (pool by distance bucket, cited works), other papers
rowsA_w = S["wa"][S["idxA"]]; n = len(rowsA_w)
edges = S["edges"]; per = 128 * 3 // 4 // (len(edges) - 1) + 1
spr = np.concatenate([pick(rowsA_w[lo:hi], per) for lo, hi in zip(edges, edges[1:]) if hi > lo])[:96]
spr = np.concatenate([spr, pick(S["title_only"], 22), pick(S["with_vec"], 10)])
both = np.concatenate([act, spr]); tx = texts_any(both)
pub = Fm.select_field.__code__  # (publications filter skipped: same for both arms)
tK = realT(1.0); yb = tK.label(tx, "habitat fragmentation", None)
print("labelled", len(both), "usd", round(tK.usd, 3), "failed", tK.failed)
y_act, y_spr = yb[:len(act)], yb[len(act):]
print("on topic: active %.2f  spread %.2f" % (np.nanmean(y_act), np.nanmean(y_spr)))
AL = {"act": act.tolist(), "y_act": y_act.tolist(), "spr": spr.tolist(), "y_spr": y_spr.tolist(), "tx": dict(zip(map(int, both), tx))}
