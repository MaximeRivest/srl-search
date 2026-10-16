"""Step 9c: a topic the list rules were not developed on (microplastics in freshwater ecosystems, the dev store's
whole-field map): its lists before (previous map code) and after (lens_min_score etc.), Kimi on every list paper."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J
D = Path("/mnt/fast/scholarsreadinglist/search/fields-dev/37a86c411ddffc31f34f0069")
LISTS = ("start_here", "core", "distinctive", "rising", "bridges", "newest", "foundations")
q = "microplastics in freshwater ecosystems"
mains = sorted((D / "maps").glob("*/main.json"), key=lambda f: f.stat().st_mtime)
old, new = json.loads(mains[0].read_text()), json.loads(mains[-1].read_text())
rows = {name: [(n, p) for g in m["groups"] for n in LISTS for p in (g["lenses"].get(n) or [])] for name, m in (("before", old), ("after", new))}
ctx = F.Ctx()
ws = sorted({int(p["id"][1:]) for r in rows.values() for _, p in r})
det = F._details(ctx, ws)
texts = [J.paper_text((det.get(w) or {}).get("title") or ctx.cit.title(int(ctx.cit.nodes([w])[0])), (det.get(w) or {}).get("abstract")) for w in ws]
k = J.Teacher(3.0); y = dict(zip(ws, k.label(texts, q, None).tolist()))
out = {"usd": round(k.usd, 3)}
for name, r in rows.items():
    by = {n: [y[int(p["id"][1:])] for nn, p in r if nn == n and y[int(p["id"][1:])] == y[int(p["id"][1:])]] for n in LISTS}
    lists = [v for n in LISTS[:-1] for v in by[n]]
    out[name] = {"on_topic": round(float(np.mean(lists)), 3), "shown": len(lists),
                 "by_list": {n: round(float(np.mean(v)), 3) if v else None for n, v in by.items()}}
    print(name, json.dumps(out[name]), flush=True)
for g in new["groups"]:
    print(" ", (g.get("label") or {}).get("name") or g["name"], "| core:", " / ".join(p["title"][:45] for p in g["lenses"]["core"][:3]),
          "| foundations:", " / ".join(p["title"][:40] for p in (g["lenses"].get("foundations") or [])[:3]))
(Path(__file__).parent / "listcheck_new_topic.json").write_text(json.dumps(out, indent=1))
print("cost $", out["usd"])
