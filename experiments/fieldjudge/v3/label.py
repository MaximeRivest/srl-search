"""One teacher labels its queue units (train: its own; eval: every eval unit). Resumable: one answer file per unit.
Fail-closed allowance checks before each wave (round 2's allowance.py). Invalid replies are saved as errors and
retried once, never guessed.

    ~/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python label.py sol|opus
"""
import json, sys, time
import teacher as T
sys.path.insert(0, str(T.V2))
from allowance import check, Pause                    # noqa: E402

Q = T.ROOT / "queue"
ANS = T.ROOT / "answers"
WAVE = 8


def request(u):
    codes = T.codes_for(len(u["options"]))
    opts = [{"code": c, "name": o["name"],
             "about": "; ".join(x for x in ((o.get("about") if u["describe"] else None),
                                            (f"in {o['parent']}" if o.get("parent") else None)) if x) or None}
            for c, o in zip(codes, u["options"])]
    papers = [{"id": f"p{i + 1:02d}", "text": p["text"]} for i, p in enumerate(u["papers"])]
    return codes, {"scheme": u["scheme_text"], "question": u["question"], "options": opts, "papers": papers}


def main(model):
    out = ANS / model; out.mkdir(parents=True, exist_ok=True)
    fn = T.classify_batch.using(**T.settings(model))
    tried = {}
    while True:
        building = not (Q / "DONE").exists()                       # build_queue.py still adding units
        todo = [f for split in ("eval", "train") for f in sorted((Q / split).glob("*.json"))]
        units = [json.loads(f.read_text()) for f in todo if not (out / f.name).exists() and tried.get(f.name, 0) < 2]
        units = [u for u in units if u["teacher"] in (model, "both")]
        if not units:
            if building:
                time.sleep(60); continue
            print("DONE", model, len(list(out.glob("*.json"))), "units", flush=True); return
        for i in range(0, len(units), WAVE):
            try:
                check(model)
            except Pause as e:
                print(model, "allowance pause:", e, "- sleeping 15 min", flush=True); time.sleep(900); break
            wave = units[i:i + WAVE]
            reqs = [request(u) for u in wave]
            t = time.time()
            rows = fn.map([r for _, r in reqs], threads=4).to_dicts()
            ok = 0
            for u, (codes, req), row in zip(wave, reqs, rows):
                tried[u["id"] + ".json"] = tried.get(u["id"] + ".json", 0) + 1
                try:
                    got = T.validate(req["papers"], set(codes), row["pred_result"])
                except Exception as e:                    # noqa: BLE001
                    (out / "errors").mkdir(exist_ok=True)
                    (out / "errors" / f"{u['id']}.json").write_text(json.dumps({"error": str(e)[:500], "provider": str(row.get("error"))[:500]}))
                    continue
                node = {c: o["node"] for c, o in zip(codes, u["options"])}
                (out / f"{u['id']}.json").write_text(json.dumps({
                    "unit": u["id"], "teacher": model, "seconds": row.get("seconds"),
                    "answers": [{"work_id": p["work_id"], "dist": {node[c]: q for c, q in g["dist"].items()}}
                                for p, g in zip(u["papers"], got)]}))
                ok += 1
            done = len(list(out.glob("*.json")))
            print(f"{model} wave {ok}/{len(wave)} ok in {time.time() - t:.0f} s; {done} units done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
