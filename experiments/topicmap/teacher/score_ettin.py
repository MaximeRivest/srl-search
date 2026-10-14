"""Score the teacher experiment's test papers and the meta-analyses' references with a fine-tuned Ettin; time it.

    cd ~/Projects/tiny-classifiers && uv run python .../score_ettin.py models/frag-ettin-17m data/tasks/frag OUT.json
"""
import json, sys, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
mdir, T, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(mdir)
res = {}
for dev in ("cuda", "cpu"):
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
    on = model.config.label2id["on topic"]
    for name in ("test", "refs_labelled"):
        rows = [json.loads(l) for l in open(f"{T}/{name}.jsonl")]
        if dev == "cpu":
            rows = rows[:400]
        t = time.time(); ps = []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                enc = tok([r["text"] for r in rows[i:i + 128]], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    ps += torch.softmax(model(**enc).logits.float(), 1)[:, on].cpu().tolist()
        if dev == "cuda":
            torch.cuda.synchronize()
            res[name] = {r["id"]: p for r, p in zip(rows, ps)}
        res.setdefault("speed", {})[f"{dev}_{name}"] = round(len(rows) / (time.time() - t), 1)
json.dump(res, open(out, "w"))
print(res["speed"])
