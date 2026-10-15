"""Step 3: Kimi K3 judges every sampled paper for its topic, with the whole-field pipeline's own question
(srl_search/judges.py: instruction, labels, provider rotation). Resumable: labels.jsonl gets one line per answer.

    ../../.venv/bin/python label.py [BUDGET_USD]      (Kimi keys in the environment; ~$45 for ~39k papers)
"""
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
topics = {t["id"]: t for t in json.loads((D / "topics.json").read_text())["topics"]}
samples = [json.loads(l) for l in open(D / "samples.jsonl")]
done = set()
if (D / "labels.jsonl").exists():
    done = {(r["topic_id"], r["id"]) for r in map(json.loads, open(D / "labels.jsonl"))}
todo = [s for s in samples if (s["topic_id"], s["id"]) not in done]
systems = {i: J.instruction(t["topic"], t["definition"]) + " Answer with exactly one label from this list and nothing else:\n"
           + "\n".join(f"- {l}" for l in J.LABELS) for i, t in topics.items()}
teacher = J.Teacher(budget)
out = open(D / "labels.jsonl", "a")
lock = threading.Lock()
n = [0]
t0 = time.time()


def one(k_s):
    k, s = k_s
    if teacher.usd >= budget:
        return
    y = teacher._one(k, systems[s["topic_id"]], s["text"])
    if y == y:
        with lock:
            out.write(json.dumps({"topic_id": s["topic_id"], "id": s["id"], "y": y}) + "\n")
            n[0] += 1
            if n[0] % 2000 == 0:
                out.flush()
                print(f"{n[0]:6d} answers  ${teacher.usd:.2f}  {time.time() - t0:.0f}s  failed {teacher.failed}", flush=True)


print(len(todo), "to label", flush=True)
with ThreadPoolExecutor(384) as ex:
    list(ex.map(one, enumerate(todo)))
out.close()
print(f"LABELS DONE: {n[0]} answers, ${teacher.usd:.2f}, failed {teacher.failed}, slowest {teacher.slowest}", flush=True)
