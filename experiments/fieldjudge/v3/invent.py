"""Invented classification schemes: the made-up ones the student must also follow. Teachers A and B each invent half.
Each request: an area of research (sampled across OpenAlex's subfields and topics, plus broad ones), a way of dividing
it (by subject, method, organism, ...) and target sizes. Output: schemes/inv-<n>.json in build_schemes.py's format.
A small share are held out (split "heldout") to test schemes the student never saw.

    ~/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python invent.py [N]
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functai import ai, _ai
from pydantic import BaseModel, Field
import teacher as T

A, B = T.ROLES
SCHEMES = T.ROOT / "schemes"
STYLES = [
    "by subject matter (what is studied)", "by research method or technique", "by organism, material or system studied",
    "by application area or sector served", "by type of data or evidence used", "by study design (experiment, observation, "
    "simulation, review, ...)", "by scale (molecular to global, or individual to society)", "by the main question asked "
    "(cause, mechanism, prediction, design, evaluation, ...)", "by theoretical school or framework", "by stage of a "
    "pipeline or life cycle", "by population or setting studied", "by outcome or property of interest",
    "the way a funding agency would organise its review panels", "the way a journal would organise its sections",
    "the way a university library would shelve it", "the way a review article would organise this literature",
]


class Node(BaseModel):
    name: str = Field(description="Short option name, as it would appear in the scheme (2-8 words).")
    about: str = Field(default="", description="One sentence saying what belongs here, or empty.")
    parent: str = Field(default="", description="For a second-level option: its parent's name exactly; empty for top-level.")


class Scheme(BaseModel):
    name: str = Field(description="Name of the classification scheme.")
    purpose: str = Field(description="One sentence: who would use it, for what.")
    top_level: str = Field(description="What a top-level option is called (e.g. 'section', 'theme', 'method family').")
    second_level: str = Field(default="", description="What a second-level option is called, or empty if one level.")
    nodes: list[Node]


@ai
def invent(area: str, style: str, top_options: int, children_each: str, descriptions: bool) -> Scheme:
    """Invent a realistic classification scheme for scientific papers in the given area, divided in the given way.
    Use about `top_options` top-level options; give each `children_each` second-level options (none if '0').
    Options at one level should be mutually exclusive enough that an expert could pick one best option for most
    papers, while covering the area; real schemes often have an 'Other ...' option, add one when natural.
    Write a one-sentence `about` for every option when `descriptions` is true, otherwise leave it empty.
    Use the vocabulary experts in the area use. List each top-level option before its children."""
    return _ai


def areas(rng):
    st = json.loads((SCHEMES / "openalex.json").read_text())["nodes"]
    subs = [n for n in st if n["level"] == 2]
    tops = [n for n in st if n["level"] == 3]
    fields = [n for n in st if n["level"] == 1]
    out = [f"all of science ({rng.choice(['every discipline', 'natural and social sciences and humanities'])})"] * 6
    out += [f"{n['name']} (a field)" for n in rng.sample(fields, 20)]
    out += [n["name"] for n in rng.sample(subs, 160)]
    out += [n["name"] + (f": {n['about']}" if n.get("about") else "") for n in rng.sample(tops, 120)]
    rng.shuffle(out)
    return out


def main(N):
    rng = random.Random(3)
    jobs = []
    for i, area in enumerate(areas(rng)[:N]):
        two = rng.random() < 0.75
        jobs.append({"i": i, "area": area, "style": rng.choice(STYLES), "top_options": rng.choice([3, 4, 5, 6, 8, 10, 12, 15, 20, 25]),
                     "children_each": rng.choice(["2-4", "3-6", "4-8", "5-12"]) if two else "0",
                     "descriptions": rng.random() < 0.5, "teacher": A if i % 2 else B,
                     "split": "heldout" if i % 15 == 7 else "train"})

    def run(model):
        mine = [j for j in jobs if j["teacher"] == model and not (SCHEMES / f"inv-{j['i']:03d}.json").exists()]
        fn = invent.using(**T.settings(model))
        rows = fn.map([{k: j[k] for k in ("area", "style", "top_options", "children_each", "descriptions")} for j in mine],
                      threads=4).to_dicts()
        ok = 0
        for j, r in zip(mine, rows):
            s = {k[5:]: v for k, v in r.items() if k.startswith("pred_")}
            if r.get("error") not in (None, "None") or not s.get("nodes"):
                print(model, j["i"], "error", str(r.get("error"))[:150], flush=True); continue
            s = Scheme.model_validate(s)
            tops = {n.name for n in s.nodes if not n.parent}
            nodes, ids = [], {}
            for n in s.nodes:
                if n.name in ids or (n.parent and n.parent not in tops):
                    continue
                ids[n.name] = f"n{len(ids)}"
                nodes.append({"id": ids[n.name], "name": n.name, "about": n.about or None,
                              "parent": ids.get(n.parent) if n.parent else None, "level": 1 if n.parent else 0})
            nodes = [n for n in nodes if n["level"] == 0 or n["parent"]]
            if sum(n["level"] == 0 for n in nodes) < 2:
                continue
            levels = [s.top_level] + ([s.second_level or "subgroup"] if any(n["level"] for n in nodes) else [])
            (SCHEMES / f"inv-{j['i']:03d}.json").write_text(json.dumps(
                {"id": f"inv-{j['i']:03d}", "name": s.name, "purpose": s.purpose, "source": f"invented by {model}",
                 "area": j["area"], "style": j["style"], "split": j["split"], "levels": levels, "nodes": nodes}, indent=0))
            ok += 1
        print(model, "schemes written:", ok, "of", len(mine), flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(run, T.ROLES))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
