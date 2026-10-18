"""Freeze evaluation papers, then collect family-grouped coverage/contrast/mining pools.

Run with the search-dev Python, on CPU, with LimitNOFILE=262144. Pin SRL_RELEASE.
No teacher or GPU calls. Evaluation weights refer to the eligible retrieval pool,
not every paper in science. Test -> validation -> train ownership prevents the
same work or normalized title crossing a split, including alternate work IDs.
Per-family files commit atomically, so resuming cannot skip half a topic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import faiss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import field as F, judges as J
from srl_search.engine import passes
from srl_search.scope import NON_PUBLICATION_TYPES

ROOT = Path(os.environ.get("FIELDJUDGE_V2", "/mnt/fast/scholarsreadinglist/fieldjudge/v2"))
RECIPE = json.loads((HERE / "recipe.json").read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def recipe_sha256():
    """The recipe's digest, without its record of earlier digests (recipe.json: earlier_sha256)."""
    return digest({k: v for k, v in RECIPE.items() if not k.startswith("earlier_sha256")})


def same_recipe(saved):
    """`saved` with an earlier digest of the same recipe (before a rewording) read as the current one."""
    if saved.get("recipe_sha256") in RECIPE.get("earlier_sha256", []):
        return {**saved, "recipe_sha256": recipe_sha256()}
    return saved


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def title_key(text):
    return hashlib.sha256(re.sub(r"[^\w]", "", text.splitlines()[0].casefold()).encode()).hexdigest()

class Collector:
    def __init__(self, family_dir=None):
        self.family_dir = Path(family_dir) if family_dir else ROOT/'families'
        self.ctx = F.Ctx()
        self.cit = self.ctx.cit
        self.eng = self.ctx.engine
        faiss.omp_set_num_threads(16)
        self.bad = [i for i, name in enumerate(self.cit.types) if name in NON_PUBLICATION_TYPES]
        self.owners = {}
        self.wowners = {}
        manifest = ROOT / "corpus-manifest.json"
        current = {"release": self.ctx.release, "recipe_sha256": recipe_sha256()}
        if manifest.exists() and same_recipe(json.loads(manifest.read_text())) != current:
            raise ValueError("release/recipe changed; do not mix data snapshots")
        if not manifest.exists():
            write(manifest, current)

    def own(self, row):
        split = row["split"]
        for owners, key in ((self.owners, row["text_group"]), (self.wowners, row["work_id"])):
            old = owners.get(key)
            if old is not None and old != split:
                return False
        self.owners[row["text_group"]] = split
        self.wowners[row["work_id"]] = split
        return True

    def reserve(self):
        # The old two-topic test rows stay excluded even though they are development diagnostics now.
        for row in map(json.loads, open(ROOT.parent / "tests.jsonl")):
            self.owners[title_key(row["text"])] = "test"
            self.wowners[int(row["id"])] = "test"
        for path in sorted((ROOT / "benchmark").glob("*.json")) + sorted((ROOT / "benchmark-titles").glob("*.json")):
            for row in json.loads(path.read_text())["pairs"]:
                if not self.own(row):
                    raise ValueError("benchmark ownership conflict")
        for split in ("validation", "train"):
            for path in sorted(self.family_dir.glob("*.json")):
                data = json.loads(path.read_text())
                if data["split"] == split:
                    for row in data["coverage"] + data["cross_scope"] + data["mining_pool"]:
                        if not self.own(row):
                            raise ValueError("saved cross-split duplicate")

    def pool(self, scope, size=300000):
        query = scope["topic"] + (": " + scope["definition"] if scope.get("definition") else "")
        q = self.eng.encoder.encode(query)[0]
        _, rows = self.eng.index.search(q.reshape(1, -1), size, params=faiss.SearchParametersIVF(nprobe=512))
        rows = rows[0][rows[0] >= 0]
        return np.asarray(self.eng.work_ids[rows], np.int64), rows

    def papers(self, ws):
        det = F._details(self.ctx, ws)
        out = []
        for w in ws:
            d = det.get(int(w))
            if d:
                if not passes(d, {}):
                    continue
                title, abstract = d.get("title"), d.get("abstract")
            else:
                nd = int(self.cit.nodes([int(w)])[0])
                if nd < 0 or int(self.cit.kind[nd]) in self.bad:
                    continue
                title, abstract = self.cit.title(nd), None
            if len(title or "") < 10:
                continue
            text = J.paper_text(title, abstract)
            out.append({"work_id": int(w), "text": text, "text_group": title_key(text), "title_only": not bool(abstract)})
        return out

    def pair(self, paper, scope, family, split, source, **extra):
        return {**paper, "id": digest([scope["id"], paper["work_id"], paper["text"]])[:24],
                "topic": scope["topic"], "definition": scope.get("definition"), "scope_id": scope["id"],
                "family_id": family, "split": split, "source": source, **extra}

    def benchmark(self):
        for scope0 in RECIPE["new_holdouts"]:
            scope = {**scope0, "id": digest(scope0)[:20]}
            path = ROOT / "benchmark" / f"{scope['id']}.json"
            if path.exists():
                continue
            ws, rows = self.pool(scope)
            vectors = F.unit(self.eng.vectors.get(rows))
            good = self.ctx.quality.junk(ws, vectors) < .5
            del vectors
            ws = ws[good]
            # Exact eligible population sizes, not pre-filter sizes.
            papers = self.papers(ws)
            bounds = [0, 100, 1000, 10000, 50000, len(papers)]
            bounds = sorted(set(min(len(papers), b) for b in bounds))
            rng = random.Random(int(scope["id"][:8], 16))
            pairs = []
            for lo, hi in zip(bounds, bounds[1:]):
                bucket = papers[lo:hi]
                sample = rng.sample(bucket, min(100, len(bucket)))
                for p in sample:
                    row = self.pair(p, scope, scope["id"], "test", "retrieval_rank", stratum=f"{lo}:{hi}",
                                    population_n=len(bucket), sample_n=len(sample), weight=len(bucket)/len(sample))
                    if not self.own(row):
                        raise ValueError("test ownership conflict")
                    pairs.append(row)
            write(path, {"scope": scope, "eligible_population": len(papers), "pairs": pairs,
                         "coverage": "Rank-stratified nearest-300k retrieval pool; not a full-map recall benchmark."})
            print("BENCHMARK", scope["topic"], len(pairs), flush=True)

    def benchmark_titles(self):
        for scope0 in RECIPE['new_holdouts']:
            scope = {**scope0, 'id': digest(scope0)[:20]}
            path = ROOT/'benchmark-titles'/f"{scope['id']}.json"
            if path.exists():
                continue
            ws, _ = self.pool(scope)
            nodes = self.cit.nodes(ws[:5000]); nodes = nodes[nodes >= 0]
            refs = np.concatenate([self.cit.refs(int(n)) for n in nodes]) if len(nodes) else np.zeros(0,np.uint32)
            u,cnt = np.unique(refs,return_counts=True)
            cited = [self.cit.work(int(n)) for n in u[cnt >= 3]]
            candidates = [p for p in self.papers(cited) if p['title_only']]
            rng = random.Random(int(scope['id'][:8],16))
            chosen = rng.sample(candidates,min(100,len(candidates)))
            pairs = []
            for p in chosen:
                row = self.pair(p,scope,scope['id'],'test','cited_title_only',stratum='cited_titles',
                                population_n=len(candidates),sample_n=len(chosen),weight=len(candidates)/len(chosen))
                if not self.own(row):
                    raise ValueError('title benchmark ownership conflict')
                pairs.append(row)
            write(path,{'scope':scope,'pairs':pairs,'eligible_population':len(candidates),
                        'coverage':'Cited title-only papers, a separate diagnostic from retrieval-pool precision/recall.'})
            print('TITLE BENCHMARK',scope['topic'],len(pairs),flush=True)

    def family(self, family):
        path = self.family_dir / f"{family['id']}.json"
        if path.exists():
            return
        split = family["split"]
        rng = random.Random(int(family["id"][:8], 16))
        coverage, mining = [], []
        used_pairs = set()

        def take(candidates, scope, count, source):
            out = []
            for paper in candidates:
                row = self.pair(paper, scope, family["id"], split, source)
                if row["id"] in used_pairs or not self.own(row):
                    continue
                used_pairs.add(row["id"])
                out.append(row)
                if len(out) == count:
                    break
            return out

        for scope in family["scopes"]:
            ws, rows = self.pool(scope)
            idx = []
            bounds = [0, 100, 1000, 10000, 50000, len(ws)]
            for lo, hi in zip(bounds, bounds[1:]):
                idx += rng.sample(range(lo, hi), min(30, hi-lo))
            random_idx = rng.sample(range(len(ws)), 100)
            random_work = {int(ws[i]) for i in random_idx}
            idx += random_idx
            idx = list(dict.fromkeys(idx))
            vectors = F.unit(self.eng.vectors.get(rows[idx]))
            good = self.ctx.quality.junk(ws[idx], vectors) < .5
            candidates = self.papers(ws[np.asarray(idx)[good]])
            # Preserve rank-bucket membership through eligibility filtering.
            ranks = {int(ws[i]): i for i in idx}
            buckets = [[p for p in candidates if lo <= ranks[p['work_id']] < hi]
                       for lo, hi in zip(bounds, bounds[1:])]
            spread = [p for j in range(max(map(len, buckets), default=0)) for b in buckets if j < len(b) for p in [b[j]]]
            part = take(spread, scope, 12, "retrieval_spread")
            broad = [p for p in candidates if p['work_id'] in random_work]
            rng.shuffle(broad)
            part += take(broad, scope, 2, "random_candidate")
            nodes = self.cit.nodes(ws[:5000]); nodes = nodes[nodes >= 0]
            refs = np.concatenate([self.cit.refs(int(n)) for n in nodes]) if len(nodes) else np.zeros(0, np.uint32)
            u, cnt = np.unique(refs, return_counts=True)
            u = u[cnt >= 3]
            cited_w = np.array([self.cit.work(int(n)) for n in u], np.int64)
            rng.shuffle(cited_w)
            cited = self.papers(cited_w[:5000])
            title_only = [p for p in cited if p["title_only"]]
            with_abs = [p for p in cited if not p["title_only"]]
            part += take(title_only, scope, 4, "cited_title_only")
            part += take(with_abs, scope, 2, "cited_with_abstract")
            if len(part) < 20:
                part += take(candidates, scope, 20-len(part), "coverage_fallback")
            if len(part) != 20:
                raise ValueError(f"not enough coverage for {scope['topic']}")
            coverage += part
            rng.shuffle(candidates)
            mining += take(candidates, scope, 20, "mining_pool")
        # Cross-scope judgments share papers but never assume that a sibling is negative.
        cross = []
        choices = [(p, s) for p in coverage for s in family["scopes"] if p["scope_id"] != s["id"]]
        rng.shuffle(choices)
        for p, scope in choices:
            row = self.pair({k:p[k] for k in ('work_id','text','text_group','title_only')}, scope,
                            family["id"], split, "cross_scope", origin_scope=p["scope_id"])
            if row["id"] not in used_pairs:
                used_pairs.add(row["id"])
                cross.append(row)
            if len(cross) == 50:
                break
        if len(coverage) != 100 or len(cross) != 50 or len(mining) != 100:
            raise ValueError("short family; refuse an incomplete training shard")
        write(path, {"family_id": family["id"], "split": split, "coverage": coverage, "cross_scope": cross, "mining_pool": mining})
        print("FAMILY", family["id"], split, "100 coverage + 50 cross-scope + 100 mining candidates", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["benchmark", "benchmark-titles", "families"])
    args = ap.parse_args()
    c = Collector(); c.reserve()
    if args.stage == "benchmark":
        c.benchmark()
    elif args.stage == 'benchmark-titles':
        c.benchmark_titles()
    else:
        if any(len(list((ROOT / d).glob('*.json'))) != len(RECIPE['new_holdouts']) for d in ('benchmark','benchmark-titles')):
            raise ValueError("freeze all benchmark papers first")
        families = json.loads((ROOT / "topics.json").read_text())["families"]
        split = json.loads((ROOT / 'split.json').read_text())
        if split['topics_sha256'] != hashlib.sha256((ROOT/'topics.json').read_bytes()).hexdigest():
            raise ValueError('split belongs to a different topic manifest')
        families = [{**f, 'split': split['family_split'][f['id']]} for f in families]
        for family in sorted(families, key=lambda f: (f["split"] != "validation", f["id"])):
            c.family(family)
