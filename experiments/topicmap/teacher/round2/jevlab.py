EV = json.load(open("/home/maxime/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher/eval_population.json"))
refs = [json.loads(l) for l in open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/refs_labelled.jsonl")]
meta = json.load(open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/meta.json"))
ref_pos = [int(r["id"][1:]) for r in refs if r["label"] == "on topic"]
ref_ab = np.array([bool(meta["W%d" % w]["abstract"]) for w in ref_pos])
allw = sorted(set(EV["ids"]) | set(CL["ids"]) | set(ref_pos))
deta = Fm._details(ctx, allw)
jv = J.Jev(1.0); t = time.time()
jl = jv.label([pair(w, deta) for w in allw], "habitat fragmentation", None)
JEV = dict(zip(allw, jl.tolist()))
print(len(allw), "Jev", round(time.time() - t, 1), "s $", round(jv.usd, 3), "failed", jv.failed)
ev_ids = np.array(EV["ids"]); ev_y = np.array([np.nan if y is None else y for y in EV["y"]])
m = ~np.isnan(ev_y); jj = np.array([JEV[int(w)] for w in ev_ids])
print("Jev vs Kimi, population sample: agree", round(float(((jj[m] >= .5) == (ev_y[m] >= .5)).mean()), 3))
cy = np.array([np.nan if y is None else y for y in CL["y"]]); cj = np.array([JEV[int(w)] for w in CL["ids"]]); m2 = ~np.isnan(cy)
for s_ in ("2", "3", "4", "5"):
    k_ = m2 & np.array([x.startswith(s_) for x in CL["stratum"]])
    print(f"  classics stratum {s_}: Kimi yes {cy[k_].mean():.2f}, Jev yes {(cj[k_] >= .5).mean():.2f}, agree {((cj[k_] >= .5) == (cy[k_] >= .5)).mean():.2f}")
