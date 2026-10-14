t = time.time()
cit = ctx.cit
u_nodes = cit.nodes(univ)                                     # node of each candidate (-1: not in the index)
order_n = np.argsort(u_nodes); sorted_nodes = u_nodes[order_n]
def node_pos(nodes):
    i = np.searchsorted(sorted_nodes, nodes); i = np.minimum(i, len(sorted_nodes) - 1)
    return np.where(sorted_nodes[i] == nodes, order_n[i], -1)
def cite_features(score):
    """in: soft count of candidates citing each candidate (weighted by their score), share of its citations;
       out: score-weighted share of its references that are candidates."""
    src = np.flatnonzero((u_nodes >= 0) & (score >= 0.05))
    offs = np.asarray(cit.fwd_off)
    lens = np.array([offs[n + 1] - offs[n] for n in u_nodes[src]], np.int64)
    allr = np.concatenate([cit.refs(int(n)) for n in u_nodes[src]]).astype(np.int64)
    wts = np.repeat(score[src], lens)
    pos = node_pos(allr); ok = pos >= 0
    fc = np.bincount(pos[ok], weights=wts[ok], minlength=len(univ))
    tot = np.asarray(cit.cites)[np.maximum(u_nodes, 0)].astype(float); tot[u_nodes < 0] = 0
    in_share = fc / np.maximum(tot, 1)
    # outgoing: every candidate's references (all candidates, not only likely ones)
    has = np.flatnonzero(u_nodes >= 0)
    lens2 = np.array([offs[n + 1] - offs[n] for n in u_nodes[has]], np.int64)
    allr2 = np.concatenate([cit.refs(int(n)) for n in u_nodes[has]]).astype(np.int64)
    p2 = node_pos(allr2); sc2 = np.where(p2 >= 0, score[np.maximum(p2, 0)], 0.0)
    owner = np.repeat(np.arange(len(has)), lens2)
    out_sum = np.bincount(owner, weights=sc2, minlength=len(has))
    out_share = np.zeros(len(univ)); nref = np.zeros(len(univ))
    out_share[has] = out_sum / np.maximum(lens2, 1); nref[has] = lens2
    return fc, tot, in_share, out_share, nref
fc, tot, in_share, out_share, nref = cite_features(fin)
print("citation features", round(time.time() - t), "s")
pos_of = {w: i for i, w in enumerate(univ.tolist())}
ev_u = np.array([pos_of[int(w)] for w in ev_ids])
for name, f in (("in_share", in_share), ("log fc", np.log1p(fc)), ("out_share", out_share), ("students", stud)):
    print(f"  {name:10s} AUC on eval {wauc(f[ev_u])}")
mi = np.array([pos_of[w] for w in miss]); neg = ev_u[(ev_y < .5) & ok_e]
print("missed meta refs: median in_share %.3f, fc %.1f, out_share %.3f" % (np.median(in_share[mi]), np.median(fc[mi]), np.median(out_share[mi])))
print("eval negatives:   median in_share %.3f, fc %.1f, out_share %.3f" % (np.median(in_share[neg]), np.median(fc[neg]), np.median(out_share[neg])))
