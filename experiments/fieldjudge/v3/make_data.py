"""Teacher answers -> training rows (v3/training/{train,validation}.jsonl).

pick rows   one per (train unit, paper) with a valid answer; 4% of train units (whole units) are validation
yes/no rows round 2's labels (v2/training/final.jsonl, its fixed validation split kept apart), YESNO_TRAIN of them,
            balanced as in round 2's data. Eval units (held-out schemes and papers) never enter here.
"""
import json, random
from collections import Counter
from pathlib import Path

V3 = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v3")
V2T = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v2/training")
OUT = V3 / "training"; OUT.mkdir(exist_ok=True)
YESNO_TRAIN, YESNO_VAL, VAL_UNITS = 40000, 1500, 0.04
rng = random.Random(23)

rows, val, stats = [], [], Counter()
for f in sorted((V3 / "queue" / "train").glob("*.json")):
    u = json.loads(f.read_text())
    a = V3 / "answers" / u["teacher"] / f.name
    if not a.exists():
        stats["unit_unanswered"] += 1; continue
    ans = {x["work_id"]: x["dist"] for x in json.loads(a.read_text())["answers"]}
    to_val = rng.random() < VAL_UNITS
    for p in u["papers"]:
        d = ans.get(p["work_id"])
        if not d:
            continue
        r = {"kind": "pick", "unit": u["id"], "scheme": u["scheme"], "teacher": u["teacher"], "shape": u["kind"],
             "paper": p["text"], "work_id": p["work_id"], "scheme_text": u["scheme_text"], "question": u["question"],
             "describe": u["describe"], "options": u["options"], "target": d}
        (val if to_val else rows).append(r); stats["pick_val" if to_val else "pick_train"] += 1

pick_ids = {r["work_id"] for r in rows + val}
yn = [r for r in map(json.loads, open(V2T / "final.jsonl")) if r.get("y") in (0, 1, 0.0, 1.0)]
rng.shuffle(yn)
for r in yn[:YESNO_TRAIN]:
    rows.append({"kind": "yesno", "text": r["text"], "topic": r["topic"], "definition": r.get("definition"), "y": float(r["y"])})
    stats["yesno_train"] += 1
yv = [r for r in map(json.loads, open(V2T / "validation.jsonl")) if r.get("y") in (0, 1, 0.0, 1.0)]
rng.shuffle(yv)
for r in yv[:YESNO_VAL]:
    val.append({"kind": "yesno", "text": r["text"], "topic": r["topic"], "definition": r.get("definition"), "y": float(r["y"])})
    stats["yesno_val"] += 1
rng.shuffle(rows)
with open(OUT / "train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(OUT / "validation.jsonl", "w") as f:
    for r in val:
        f.write(json.dumps(r) + "\n")
(OUT / "manifest.json").write_text(json.dumps(dict(stats), indent=1))
print(dict(stats))
