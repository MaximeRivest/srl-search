import shutil
W = job.dir / "work"; W.mkdir(exist_ok=True)
DEC = W / "decider"
if not (DEC / "merged").exists():
    shutil.copytree(V3 + "/work/decider/merged", DEC / "merged")
t = time.time()
d_ev = J.decider_scores(DEC, ev_texts, print)
print("decider on eval set", round(time.time() - t), "s")
from sklearn.metrics import roc_auc_score
def wauc(score, mask=None):
    m = ok_e & ~np.isnan(score) & (np.ones_like(ok_e) if mask is None else mask)
    return round(roc_auc_score(ev_y[m] >= .5, score[m], sample_weight=w_e[m]), 3)
print("decider AUC vs Kimi on population (weighted):", wauc(d_ev), " title-only:", wauc(d_ev, np.array([t.endswith("(no abstract)") for t in ev_texts])))
# calibration: weighted mean predicted vs observed per probability bin
for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    m = ok_e & (d_ev >= lo) & (d_ev < hi)
    if m.sum(): print(f"  decider p in [{lo},{hi}): n={m.sum():4d} mean p={np.average(d_ev[m], weights=w_e[m]):.2f} Kimi yes={np.average(ev_y[m], weights=w_e[m]):.2f}")
