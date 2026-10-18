"""Step 6: the general judge's yes/no line, per topic, from the 400 reference-model answers every whole-field run already has
(the real runs' teacher answers; disjoint from the test yardsticks). P(y | score) does not depend on how papers were
sampled when the sampling depends only on the paper (distance buckets, citations), so a logistic fit on those 400
(Platt) gives probabilities for the whole population; then (a) the line at 0.5, or (b) the line that maximises the
expected F1 over the population. Scored on GPU 1 with transformers (CPU is far too slow for Qwen3.5's linear attention).

    ../../.venv/bin/python calibrate.py MODEL_DIR NAME
"""
import json, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J
from transformers import AutoModelForCausalLM, AutoTokenizer

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
mdir, name = Path(sys.argv[1]) / "merged", sys.argv[2]
RUNS = {"frag": ("habitat fragmentation", None, "fields-test5/a699318c9a209d289e4c3702"),
        "llm": ("LLM post-training", "training large language models after pretraining: supervised fine-tuning and instruction "
                "tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement "
                "learning for reasoning, and alignment training", "fields-test6/1a0c3446c53e7f1da081def4")}
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'


from srl_search.citations import Citations
REL = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
CIT = Citations(Path(REL["paths"]["citations"]))


def text_of(w):
    """As the pipeline reads it: title and abstract, or the citation index's title alone (works without a record)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8810/works/W{w}?abstracts=full", timeout=30) as r:
            d = json.load(r)
        d = d.get("work", d)
        return J.paper_text(d.get("title"), d.get("abstract"))
    except urllib.error.HTTPError:
        nd = int(CIT.nodes([int(w)])[0])
        return J.paper_text(CIT.title(nd) if nd >= 0 else "", None)


cfg = json.loads((mdir / "decider.json").read_text())
torch.set_num_threads(32)
tok = AutoTokenizer.from_pretrained(mdir)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).to("cuda").eval()
W = model.get_output_embeddings().weight[[cfg["yes"], cfg["no"]]].float()


def score(system, texts):
    out = []
    for s in range(0, len(texts), 16):
        ps = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in texts[s:s + 16]]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=768, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            h = model.model(**enc).last_hidden_state[:, -1, :].float()
        lg = h @ W.T
        out += torch.sigmoid(lg[:, 0] - lg[:, 1]).cpu().tolist()
    return np.array(out)


tests = [json.loads(l) for l in open(D / "tests.jsonl")]
p_test = np.load(D / f"scores_{name}.npy")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
res = {}
for t, (topic, defin, run) in RUNS.items():
    teach = json.loads(Path(f"/mnt/fast/scholarsreadinglist/search/{run}/field.json").read_text())["labels"]["teacher"]
    ws, ys = list(teach), np.array([v >= 0.5 for v in teach.values()])
    with ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(text_of, ws))
    pt = score(J.instruction(topic, defin) + ANSWER, texts)
    platt = LogisticRegression(C=100).fit(lg(pt).reshape(-1, 1), ys)
    cal = lambda p: platt.predict_proba(lg(p).reshape(-1, 1))[:, 1]
    idx = [i for i, r in enumerate(tests) if r["test"] == t and r["set"] == "population" and r["reference"] is not None]
    y = np.array([tests[i]["reference"] >= 0.5 for i in idx]); w = np.array([tests[i]["weight"] for i in idx])
    raw, c = p_test[idx], cal(p_test[idx])
    o = np.argsort(-c); cw = np.cumsum((w * c)[o]); nw = np.cumsum(w[o])
    f1 = 2 * cw / (nw + (w * c).sum()); cut = float(c[o][int(np.argmax(f1))])
    def pr(yes):
        tp = (w * yes * y).sum(); P, R = tp / (w * yes).sum(), tp / (w * y).sum(); return round(P, 3), round(R, 3), round(2 * P * R / (P + R), 3)
    res[t] = {"raw@0.5": pr(raw >= .5), "calibrated@0.5": pr(c >= .5), f"calibrated@expected-F1 ({cut:.2f})": pr(c >= cut),
              "teacher_auc_of_raw": None, "expected_on_topic_share": round(float((w * c).sum() / w.sum()), 4),
              "reference_on_topic_share": round(float((w * y).sum() / w.sum()), 4)}
    for i_set in ("meta_refs", "landmarks", "cited50"):
        j = [i for i, r in enumerate(tests) if r["test"] == t and r["set"] == i_set]
        if j:
            res[t][f"{i_set} found (calibrated@0.5)"] = round(float((cal(p_test[j]) >= .5).mean()), 3)
    print(t, json.dumps(res[t]), flush=True)
(Path(__file__).parent / f"calibration_{name}.json").write_text(json.dumps(res, indent=1))
