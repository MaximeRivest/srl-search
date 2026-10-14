"""Learning curve and staircase for "is this paper about the topic?" (habitat fragmentation), tiny-classifiers' recipe.

    cd ~/Projects/tiny-classifiers
    uv run python .../curve.py --model jhu-clsp/ettin-encoder-17m --n 1000 --seed 0
    uv run python .../curve.py --model ...-150m --n 4000 --predict pool          # the big student labels the pool
    uv run python .../curve.py --model ...-17m --pseudo PROBS.json --n 4000       # a small student learns from it

Training data: Kimi K3's labels on train.jsonl + more.jsonl (17,580 papers), shuffled by the seed, first N.
With --pseudo: the pool papers labelled by a bigger student (P >= 0.5), plus the same N Kimi labels.
Evaluated on test.jsonl (1,414 papers, Kimi's labels) and refs_labelled.jsonl (the 19 meta-analyses' references,
Kimi's labels): AUC, accuracy, recall and precision at 0.5, recall on papers with a title only. Appends one line
to curve_results.jsonl beside this file; --predict NAME also saves P(on topic) for NAME.jsonl.
Recipe (as tiny-classifiers): all weights, batch 32, AdamW lr 1e-4 wd 0.01, 5% warm-up then cosine, clip 1.0,
max(6, 6 x 9,493 / N) passes unless --epochs.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
T = Path(os.path.expanduser("~/Projects/tiny-classifiers/data/tasks/frag"))
L = Path(os.path.expanduser("~/Projects/tiny-classifiers/labels"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, required=True, help="Kimi-labelled training papers")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pseudo", help="JSON {id: P} from a bigger student over the pool")
ap.add_argument("--pseudo-n", type=int, default=0, help="at most this many pool papers (0: all)")
ap.add_argument("--epochs", type=int)
ap.add_argument("--predict", help="also score NAME.jsonl and save P(on topic)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
DEV = "cuda"
torch.manual_seed(a.seed)
random.seed(a.seed)
LABELS = ["on topic", "not on topic"]
load = lambda n: [json.loads(l) for l in open(T / f"{n}.jsonl")]
meta = json.load(open(T / "meta.json"))

kimi = {}
for f in ("frag-all.jsonl", "frag-more.jsonl"):
    for l in open(L / f):
        r = json.loads(l)
        kimi[r["id"]] = r["pred"] == "on topic"
pool_kimi = [r for r in load("train") + load("more") if r["id"] in kimi]
random.Random(a.seed).shuffle(pool_kimi)
train = [(r["text"], float(kimi[r["id"]])) for r in pool_kimi[:a.n]]
n_pseudo = 0
if a.pseudo:
    probs = json.load(open(a.pseudo))
    pool = [r for r in load("pool") if r["id"] in probs]
    random.Random(a.seed).shuffle(pool)
    if a.pseudo_n:
        pool = pool[:a.pseudo_n]
    train += [(r["text"], float(probs[r["id"]] >= 0.5)) for r in pool]
    n_pseudo = len(pool)
random.Random(a.seed + 1).shuffle(train)

tok = AutoTokenizer.from_pretrained(a.model)
lens = sorted(len(tok(t)["input_ids"]) for t, _ in train[:2000])
MAXLEN = min(512, 64 * math.ceil(lens[int(.95 * len(lens))] / 64))
EPOCHS = a.epochs or max(6, round(6 * 9493 / len(train)))
BATCH, LR, WARMUP = 32, 1e-4, 0.05
enc = lambda ts: [tok(t, truncation=True, max_length=MAXLEN)["input_ids"] for t in ts]
X = enc([t for t, _ in train])
y = torch.tensor([0 if p >= 0.5 else 1 for _, p in train])          # label 0 = "on topic"
model = AutoModelForSequenceClassification.from_pretrained(
    a.model, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    attn_implementation="sdpa", dtype=torch.float32).to(DEV)


def batches(xs, shuffle, bs=BATCH):
    order = list(range(len(xs)))
    if shuffle:
        random.shuffle(order)
    else:                                 # inference: similar lengths together (less padding)
        order.sort(key=lambda i: len(xs[i]))
    for i in range(0, len(xs), bs):
        b = order[i:i + bs]
        yield b, tok.pad({"input_ids": [xs[j] for j in b]}, return_tensors="pt").to(DEV)


def logits_of(e):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**e).logits.float()


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(X) / BATCH)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                          0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
model.train()
t0 = time.time()
for ep in range(EPOCHS):
    for b, e in batches(X, True):
        loss = F.cross_entropy(logits_of(e), y[b].to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
torch.cuda.synchronize()
train_s = time.time() - t0
model.eval()


def score(rows):
    xs = enc([r["text"] for r in rows])
    p = np.zeros(len(rows))
    t = time.time()
    with torch.no_grad():
        for b, e in batches(xs, False, 256):
            p[b] = torch.softmax(logits_of(e), 1)[:, 0].cpu().numpy()
    torch.cuda.synchronize()
    return p, len(rows) / (time.time() - t)


res = {"model": a.model.split("/")[-1], "n": a.n, "seed": a.seed, "pseudo": n_pseudo,
       "pseudo_from": Path(a.pseudo).name if a.pseudo else None, "tag": a.tag, "epochs": EPOCHS, "maxlen": MAXLEN,
       "train_rows": len(X), "train_s": round(train_s, 1)}
for name, file in (("test", "test"), ("refs", "refs_labelled")):
    rows = load(file)
    p, speed = score(rows)
    yy = np.array([r["label"] == "on topic" for r in rows])
    abs_ = np.array([meta[r["id"]]["abstract"] for r in rows])
    k = p >= 0.5
    res[name] = {"auc": round(float(roc_auc_score(yy, p)), 4), "acc": round(float((k == yy).mean()), 4),
                 "recall": round(float(k[yy].mean()), 4), "precision": round(float(yy[k].mean()), 4) if k.any() else 0,
                 "recall_title_only": round(float(k[yy & ~abs_].mean()), 4) if (yy & ~abs_).any() else None}
    res["papers_per_s"] = round(speed)
if a.predict:
    rows = load(a.predict)
    p, speed = score(rows)
    out = HERE / "preds" / f"{a.predict}__{res['model']}__n{a.n}__s{a.seed}{('__' + a.tag) if a.tag else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({r["id"]: round(float(x), 4) for r, x in zip(rows, p)}))
    res["predicted"] = {"file": out.name, "rows": len(rows), "papers_per_s": round(speed)}
with open(HERE / "curve_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
