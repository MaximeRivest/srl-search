"""Step 1: research topics across all of science, as researchers would type them, for training a general field
judge (README.md). Kimi K3 proposes 18 per OpenAlex field; half get a one-sentence definition, a few are
deliberately ambiguous words (with the definition that settles them). Near-duplicates and topics close to
the test topics are taken out in sample.py, by meaning.

    ../../.venv/bin/python topics.py            (Kimi keys in the environment; writes topics_raw.json, ~$0.30)
"""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J

FIELDS = ["Agricultural and Biological Sciences", "Arts and Humanities", "Biochemistry, Genetics and Molecular Biology",
          "Business, Management and Accounting", "Chemical Engineering", "Chemistry", "Computer Science", "Decision Sciences",
          "Dentistry", "Earth and Planetary Sciences", "Economics, Econometrics and Finance", "Energy", "Engineering",
          "Environmental Science", "Health Professions", "Immunology and Microbiology", "Materials Science", "Mathematics",
          "Medicine", "Neuroscience", "Nursing", "Pharmacology, Toxicology and Pharmaceutics", "Physics and Astronomy",
          "Psychology", "Social Sciences", "Veterinary"]
TESTS = [("habitat fragmentation", None), ("LLM post-training", "training large language models after pretraining: "
         "supervised fine-tuning and instruction tuning, reinforcement learning from human or AI feedback, preference "
         "optimization such as DPO, reinforcement learning for reasoning, and alignment training")]
ASK = """List 18 research topics in {field} that a researcher might want a map of the whole literature for.
Vary them: some broad subfields, most mid-sized topics (a few thousand to tens of thousands of papers), a few narrow
ones (a method, a phenomenon, a disease, a material, an application). Write each the way a researcher would type it
into a search box (short noun phrases, abbreviations where people use them). For about half, add a one-sentence
definition of what the topic covers. Include 2 whose words are ambiguous across fields (like "LLM", also a law
degree, or "transformers" in electrical engineering vs machine learning), with the definition that settles it.
Answer with JSON only: [{{"topic": "...", "definition": "..." or null}}, ...]"""


def kimi(prompt: str) -> str:
    for name, url, model, extra, price, key in [(n, *v) for n, v in J.KIMI.items() if os.environ.get(v[4])]:
        try:
            body = json.dumps({"model": model, "temperature": 0.7, "max_tokens": 5000, **extra,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {os.environ[key]}",
                                         "Content-Type": "application/json", "User-Agent": "scholarsreadinglist-fields/1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(name, "failed:", e, file=sys.stderr)
    raise RuntimeError("no provider answered")


def parse(text: str) -> list[dict]:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)
    return json.loads(text[text.index("["):text.rindex("]") + 1])


def one(f):
    for attempt in range(4):
        try:
            return f, parse(kimi(ASK.format(field=f)))
        except ValueError as e:
            print(f, "unreadable answer, again:", e, file=sys.stderr)
    return f, []


with ThreadPoolExecutor(26) as ex:
    lists = list(ex.map(one, FIELDS))
rows, seen = [], set()
for f, items in lists:
    for it in items:
        t = " ".join(str(it.get("topic", "")).split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            rows.append({"topic": t, "definition": (it.get("definition") or None), "field": f})
Path("/mnt/fast/scholarsreadinglist/fieldjudge/topics_raw.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print(len(rows), "topics;", sum(1 for r in rows if r["definition"]), "with a definition")
