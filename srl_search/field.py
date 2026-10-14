"""Whole-field topic maps: every paper of a research field, not the 1,000 closest to a question.

Measured on habitat fragmentation (experiments/topicmap/README.md): the 1,000 closest papers are a corner of a
field of ~25-60k; no distance cut-off separates the field from its neighbours; exact phrases miss half of it; and
30% of the studies that meta-analyses cite have no English abstract, so no text search can see them. What worked:

  1 archetypes   the 300 papers closest in meaning to the topic, labelled by Jev (TypeSafe's yes/no judgment
                 model); the field's centre = the mean of the on-topic ones
  2 pool         the 300,000 papers closest to that centre (publications, not junk)
  3 labels       Jev labels a sample spread evenly from nearest to farthest
  4 classifier   logistic regression on the papers' meaning vectors, trained on Jev's labels (on far papers it
                 ranks far better than distance: cross-validated AUC 0.97 vs 0.67); a second pool of 300,000
                 along the classifier's own direction; two rounds of labelling where it is unsure
  5 phrases      papers with the topic's exact phrase: kept if the classifier agrees (P >= 0.3), or if Jev does
                 when they have no meaning vector
  6 citations    works cited by >= 3 papers of the field so far (reaches studies without abstracts), each judged
                 by Jev (on its title and abstract, or its title alone)
  7 map          the topic map of all of it (srl_search/topics.py, strongest 30 reference links per paper)
  8 subgroups    every group of 300+ papers split into its own groups

On 19 meta-analyses of habitat fragmentation, steps 1-6 found 94% of the studies they cite (text alone: 42-68%).
A field takes ~20-40 minutes and ~$1-3 of Jev. It runs as a job (`python -m srl_search.field worker`, started by
the srl-fields path unit when a job file appears), never in the search API, which only reads results and queues
jobs (FieldStore). Results are kept across nightly releases: the selected field (field.json) is the expensive
part; maps are rebuilt from it when the map code changes (~10 min).

Store: ROOT/<key>/job.json (status), field.json (the papers and how each was found), maps/<code version>/
main.json and group-<n>.json; ROOT/queue/<key>.job (jobs to run); ROOT/spend.json (Jev dollars per day).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException

ROOT = Path(os.environ.get("SRL_FIELDS", "/mnt/fast/scholarsreadinglist/search/fields"))
SELECTION_VERSION = "1"
JEV_MODEL = "jev-1.13.0"                 # pinned: the labels' meaning must not move under a running job
JEV_USD_PER_TOKEN = 0.042 / 1e6          # input tokens; output is free (docs.typesafe.ai/models, 2026-10)
BUDGET_USD = float(os.environ.get("SRL_FIELD_BUDGET_USD", "10"))    # Jev spend a day, all jobs together
EST_USD = 3.0                            # what a new field is assumed to cost when deciding to accept it
MAX_QUEUED = 3
POOL = 300_000
SEED_K = 300                             # archetype candidates
MIN_ARCHETYPES = 15
SAMPLE_PER_BUCKET = 40
ACTIVE_ROUNDS, ACTIVE_N = 2, 300
CITED_K = 3
EXPANSION_MAX = 300_000
TOO_BROAD_TAIL = 0.30                    # share on topic among the farthest of the 300k: the field goes on beyond
TOO_BROAD_PAPERS = 150_000
SPLIT_MIN = 300                          # groups at least this big get their own subgroups
STAGES = [("archetypes", 60), ("pool", 150), ("labels", 45), ("classifier", 180), ("phrases", 90),
          ("citations", 420), ("map", 600), ("subgroups", 420)]


def field_key(q: str, definition: str | None = None, keywords: str | None = None) -> str:
    norm = " ".join(q.lower().split())
    return hashlib.sha256(json.dumps([norm, (definition or "").strip(), (keywords or "").strip()]).encode()).hexdigest()[:24]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    os.replace(tmp, path)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Pending(Exception):
    """The whole-field map is being made (or waits its turn): the job's status, for the caller to show."""

    def __init__(self, status: dict):
        super().__init__(status.get("status"))
        self.status = status


# ------------------------------------------------------------------------------------------------ the store
class FieldStore:
    """What the search API does with whole-field maps: read finished ones, report progress, queue new jobs."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        (self.root / "queue").mkdir(parents=True, exist_ok=True)

    def spent_today(self) -> float:
        return float((_read(self.root / "spend.json") or {}).get(date.today().isoformat(), 0.0))

    def queued(self) -> list[str]:
        return [p.stem for p in sorted((self.root / "queue").glob("*.job"), key=lambda p: p.stat().st_mtime)]

    def status(self, key: str) -> dict | None:
        st = _read(self.root / key / "job.json")
        if st and st.get("status") == "queued":
            q = self.queued()
            st["position"] = q.index(key) + 1 if key in q else None
            ahead = q[:q.index(key)] if key in q else []
            st["eta_s"] = sum(job_seconds((_read(self.root / k / "job.json") or {}).get("kind")) for k in ahead) \
                + job_seconds(st.get("kind")) + (60 if not ahead else 0)    # + the worker starting
        return st

    def get(self, q: str, definition: str | None, keywords: str | None, version: str, group: int | None,
            start: bool = True) -> dict:
        key = field_key(q, definition, keywords)
        d = self.root / key
        main = d / "maps" / version / "main.json"
        if main.exists():
            if group is None:
                return _read(main)
            sub = d / "maps" / version / f"group-{group}.json"
            if sub.exists():
                return _read(sub)
            m = _read(main)
            n = len(m.get("groups", []))
            if not 1 <= group <= n:
                raise HTTPException(404, f"no group {group}; this map has groups 1-{n}")
            raise HTTPException(404, f"group {group} has fewer than {SPLIT_MIN} papers and was not split further: "
                                     f"read it in full instead (read={group})")
        st = self.status(key)
        if st and st.get("status") in ("queued", "running"):
            raise Pending(st)
        if st and st.get("status") == "failed" and st.get("final"):
            raise HTTPException(422, st.get("error") or "this topic cannot be mapped as a whole field")
        if not start:
            raise HTTPException(409, "No whole-field map of this topic yet: ask for it without read= to start one "
                                     "(it takes about half an hour).")
        kind = "map" if (d / "field.json").exists() else "full"
        if kind == "full":
            queued_full = sum(1 for k in self.queued() if (_read(self.root / k / "job.json") or {}).get("kind") == "full")
            if len(self.queued()) >= MAX_QUEUED:
                raise HTTPException(503, f"{MAX_QUEUED} whole-field maps are already waiting. Try again later; the "
                                         "1,000-paper map of the topic works right away.", headers={"Retry-After": "1800"})
            if self.spent_today() + EST_USD * (queued_full + 1) > BUDGET_USD:
                raise HTTPException(503, "Today's budget for whole-field maps is used up (each costs a few dollars of "
                                         "AI judgments). Try again tomorrow; the 1,000-paper map works right away.",
                                    headers={"Retry-After": "21600"})
        st = {"status": "queued", "kind": kind, "key": key, "created": _now(), "version": version,
              "request": {"q": q, "definition": definition, "keywords": keywords}}
        _write(d / "job.json", st)
        (self.root / "queue" / f"{key}.job").write_text(key)
        raise Pending(self.status(key) or st)


STAGES_TOTAL = sum(s for _, s in STAGES)


def job_seconds(kind: str | None) -> int:
    """Typical length: a new field runs every stage; a map job (the field kept, the map code changed) the last two."""
    return sum(s for _, s in STAGES[6:]) if kind == "map" else STAGES_TOTAL


def pending_text(st: dict, q: str, base: str) -> str:
    """What an AI or a person reads while a whole-field map is being made."""
    eta = int(st.get("eta_s") or job_seconds(st.get("kind")))
    where = (f"{st['position']} in line" if st.get("status") == "queued" and st.get("position")
             else f"step {st.get('stage_index', 0) + 1} of {len(STAGES)} ({st.get('stage', 'starting')})")
    return (f"# Whole-field map of “{q}”: being made\n\n"
            f"Status: {st.get('status')}, {where}. About {max(1, round(eta / 60))} more minutes. "
            + (f"{st['detail']}. " if st.get("detail") else "") +
            "It finds every paper of the field (not just the 1,000 closest): papers close in meaning, judged one by "
            "one by an AI classifier, the topic's exact phrases, and the studies the field cites (which reach papers "
            "without abstracts), then maps them into research groups and subgroups. Once made, it is kept and "
            "opens instantly for everyone.\n\n"
            f"Ask again later (the same request). Meanwhile, the map of the 1,000 closest papers works right away: "
            f"{base}/topics/{urllib.parse.quote(q)}\n")


# ------------------------------------------------------------------------------------------------ Jev
def jev_question(topic: str, definition: str | None) -> dict:
    ins = {"topic": topic}
    if definition:
        ins["definition"] = definition
    ins["question"] = ("Is `topic` a central subject of the paper described in `paper`?"
                       + (" `definition` says what the topic covers." if definition else ""))
    return {"type": "noul", "instructions": ins, "criteria": {
        "true": "The paper studies the topic itself as a main subject: the phenomenon, its causes, consequences, "
                "mechanisms, measurement, methods, or how it is managed, in any system or organism.",
        "false": "The topic is absent or only mentioned in passing; or the words mean something else in this paper; "
                 "or the paper studies a neighbouring subject without studying the topic itself."}}


class Jev:
    def __init__(self, store_root: Path, threads: int = 64):
        self.key = os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.root, self.threads = store_root, threads
        self.tokens = self.calls = self.failed = 0

    def _one(self, state: dict, question: dict) -> float:
        body = json.dumps({"state": state, "model": JEV_MODEL, "questions": {"q": question}}).encode()
        for attempt in range(7):
            try:
                req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=body, headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.load(r)
                self.tokens += int(out.get("usage", {}).get("input_tokens", 0))
                return float(out["answers"]["q"]["noul"])
            except urllib.error.HTTPError as e:
                wait = float(e.headers.get("retry-after") or 0) or 2 ** attempt
                if e.code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(min(wait, 60))
            except (OSError, ValueError, KeyError):
                time.sleep(2 ** attempt)
        self.failed += 1
        return float("nan")

    def label(self, papers: list[dict], question: dict) -> np.ndarray:
        """P(yes) for each {"title", "abstract"}; NaN when the service did not answer."""
        t0 = self.tokens
        with ThreadPoolExecutor(self.threads) as ex:
            out = np.array(list(ex.map(lambda p: self._one({"paper": p}, question), papers)), dtype=np.float64)
        self.calls += len(papers)
        self._spend((self.tokens - t0) * JEV_USD_PER_TOKEN)
        return out

    def _spend(self, usd: float) -> None:
        path = self.root / "spend.json"
        with open(self.root / "spend.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            led = _read(path) or {}
            today = date.today().isoformat()
            led[today] = round(led.get(today, 0.0) + usd, 4)
            _write(path, led)

    @property
    def usd(self) -> float:
        return self.tokens * JEV_USD_PER_TOKEN


# ------------------------------------------------------------------------------------------------ the worker
class Ctx:
    """Everything a job reads, loaded once per worker run (~1 minute, ~15 GB)."""

    def __init__(self):
        import faiss  # noqa: F401
        from srl_search.citations import Citations
        from srl_search.engine import Engine
        from srl_search.keyword import KeywordIndex
        from srl_search.quality import Quality
        rel_dir = Path(os.environ.get("SRL_RELEASE", "/mnt/fast/scholarsreadinglist/release/current"))
        self.release = json.loads((rel_dir / "manifest.json").read_text())
        P = {k: Path(v) for k, v in self.release["paths"].items()}
        upd = P.get("updates") if P.get("updates") and (P["updates"] / "DONE.json").exists() else None
        quality_dir = P["corpus"].parent / "quality-2026-10-05"
        self.quality = Quality(quality_dir, more=[upd / "quality"] if upd else [])
        self.engine = Engine(P["index"], P["corpus"] / "works.sqlite", threads=16, quality=self.quality,
                             more_meta=[upd / "works.sqlite"] if upd else [])
        self.keywords = KeywordIndex([P["corpus"] / "keywords"] + ([upd / "keywords"] if upd else []))
        self.cit = Citations(P["citations"])
        self.xy = np.load(P["tiles"] / "xy.npy", mmap_mode="r") if (P["tiles"] / "xy.npy").exists() else None
        self.readable_rows = np.load(P["rows"] / "smr.npy", mmap_mode="r") if (P["rows"] / "smr.npy").exists() else None
        self.RELEASE, self.ATLAS = self.release, None

    # the few things topic_service.finish() asks of "api"
    def place_work(self, w: int):
        if self.xy is None:
            return None
        row = self.engine.position_of(int(w))
        if row is None or np.isnan(self.xy[row, 0]):
            return None
        return float(self.xy[row, 0]), float(self.xy[row, 1])

    def readable(self, w: int):
        if self.readable_rows is None:
            return None
        row = self.engine.position_of(int(w))
        return bool(self.readable_rows[row]) if row is not None else None


def unit(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


class Job:
    def __init__(self, store: Path, key: str):
        self.dir = store / key
        self.key = key
        self.st = _read(self.dir / "job.json") or {}
        self.t0 = time.time()

    def stage(self, i: int, detail: str = "") -> None:
        name, _ = STAGES[i]
        self.st.update({"status": "running", "stage": name, "stage_index": i, "detail": detail,
                        "eta_s": sum(s for _, s in STAGES[i:]), "updated": _now()})
        self.st.setdefault("started", _now())
        self.st.setdefault("stage_seconds", {})
        _write(self.dir / "job.json", self.st)
        print(f"[{self.key}] {time.time() - self.t0:6.0f}s {name}: {detail}", flush=True)

    def done_stage(self, i: int, started: float) -> None:
        self.st["stage_seconds"][STAGES[i][0]] = round(time.time() - started)


def _details(ctx: Ctx, ws) -> dict:
    out = {}
    ws = [int(w) for w in ws]
    for s in range(0, len(ws), 5000):
        out.update(ctx.engine.details(ws[s:s + 5000]))
    return out


def _vectors(ctx: Ctx, ws) -> tuple[np.ndarray, np.ndarray]:
    pos = [ctx.engine.position_of(int(w)) for w in ws]
    has = np.array([p is not None for p in pos], bool)
    V = np.zeros((len(ws), 768), np.float32)
    if has.any():
        V[has] = unit(ctx.engine.vectors.get(np.array([p for p in pos if p is not None], np.int64)))
    return V, has


def _paper(d: dict | None, cit=None, w: int | None = None) -> dict:
    """What Jev reads about a paper: title and abstract, or the citation index's title alone."""
    if d:
        return {"title": re.sub(r"<[^>]+>", "", d.get("title") or ""), "abstract": re.sub(r"<[^>]+>", " ", d.get("abstract") or "")[:3000]}
    nd = int(cit.nodes([w])[0]) if cit is not None and w is not None else -1
    return {"title": cit.title(nd) if nd >= 0 else "", "abstract": ""}


def select_field(ctx: Ctx, job: Job, q: str, definition: str | None, keywords: str | None, jev: Jev) -> dict:
    import faiss
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from srl_search.engine import passes
    from srl_search.scope import NON_PUBLICATION_TYPES
    eng, cit = ctx.engine, ctx.cit
    question = jev_question(q, definition)
    faiss.omp_set_num_threads(16)
    report: dict = {"topic": q, "definition": definition}
    labels: dict[int, float] = {}               # work number -> Jev P(on topic)

    def ok_publications(ws: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
        det = _details(ctx, ws)
        junk = ctx.quality.junk(ws.astype(np.int64), V) if len(ws) else np.zeros(0)
        ok = np.array([int(w) in det and passes(det[int(w)], {}) for w in ws]) & (junk < 0.5)
        return ok, det

    # 1 archetypes: the closest papers, judged; the field's centre is the mean of the on-topic ones
    t = time.time()
    job.stage(0, "judging the papers closest to the topic")
    qv = eng.encoder.encode(q)[0]
    _, cand = eng.index.search(qv.reshape(1, -1), 2000, params=faiss.SearchParametersIVF(nprobe=512))
    cand = cand[0][cand[0] >= 0]
    V = unit(eng.vectors.get(cand))
    ws = np.asarray(eng.work_ids[cand], dtype=np.int64)
    order = np.argsort(-(V @ qv))
    ws, V = ws[order], V[order]
    ok, det = ok_publications(ws, V)
    ws, V = ws[ok][:SEED_K], V[ok][:SEED_K]
    lab = jev.label([_paper(det.get(int(w))) for w in ws], question)
    labels.update({int(w): float(p) for w, p in zip(ws, lab) if not math.isnan(p)})
    on = np.nan_to_num(lab) >= 0.5
    report["archetypes"] = {"judged": int(len(ws)), "on_topic": int(on.sum())}
    if on.sum() < MIN_ARCHETYPES:
        raise FinalError(f"Only {int(on.sum())} of the {len(ws)} papers closest to “{q}” are about it: too few to find "
                         "the field. Phrase the topic as papers about it would (e.g. “effects of habitat fragmentation "
                         "on biodiversity”), or give a definition.")
    centre = unit((V[on] * lab[on, None]).sum(0, keepdims=True))[0]
    job.done_stage(0, t)

    # 2 pool: the closest to the centre
    t = time.time()
    job.stage(1, f"{int(on.sum())} archetypes; gathering the {POOL:,} papers closest to them")
    _, ca = eng.index.search(centre.reshape(1, -1).astype(np.float32), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_a = ca[0][ca[0] >= 0]
    VA = unit(eng.vectors.get(rows_a))
    wa = np.asarray(eng.work_ids[rows_a], dtype=np.int64)
    sa = VA @ centre
    o = np.argsort(-sa, kind="stable")
    rows_a, VA, wa, sa = rows_a[o], VA[o], wa[o], sa[o]
    okA, detA = ok_publications(wa, VA)
    report["pool"] = {"nearest": int(len(wa)), "publications": int(okA.sum()),
                      "similarity_range": [round(float(sa[0]), 4), round(float(sa[-1]), 4)]}
    job.done_stage(1, t)

    # 3 labels spread from nearest to farthest
    t = time.time()
    job.stage(2, "judging a sample from the nearest to the farthest")
    idxA = np.flatnonzero(okA)
    n = len(idxA)
    edges = sorted({0, *[int(n * f) for f in (.002, .004, .007, .01, .017, .025, .035, .05, .07, .1, .14, .2, .27, .4,
                                                .55, .75)], n})
    rng = np.random.default_rng(0)
    sample, bucket_of = [], []
    for bi, (a, b) in enumerate(zip(edges, edges[1:])):
        if b > a:
            pick = rng.choice(np.arange(a, b), size=min(SAMPLE_PER_BUCKET, b - a), replace=False)
            sample += idxA[pick].tolist()
            bucket_of += [bi] * len(pick)
    lab = jev.label([_paper(detA.get(int(wa[i]))) for i in sample], question)
    labels.update({int(wa[i]): float(p) for i, p in zip(sample, lab) if not math.isnan(p)})
    bucket_of = np.array(bucket_of)
    last = bucket_of == bucket_of.max()
    tail = float(np.nanmean(lab[last] >= 0.5))
    report["distance_curve"] = [{"ranks": [int(a), int(b)], "on_topic": round(float(np.nanmean(lab[bucket_of == bi] >= 0.5)), 3)}
                                for bi, (a, b) in enumerate(zip(edges, edges[1:])) if (bucket_of == bi).any()]
    report["farthest_on_topic"] = round(tail, 3)
    if tail > TOO_BROAD_TAIL:
        raise FinalError(f"“{q}” is broader than a whole-field map can hold: {round(100 * tail)}% of the papers at the "
                         f"edge of the {POOL:,} closest are still about it. Map a narrower topic, or use the "
                         "1,000-paper map.")
    job.done_stage(2, t)

    # 4 classifier on meaning, a second pool along its direction, two rounds where it is unsure
    t = time.time()
    job.stage(3, "training a classifier on the judgments; searching along it")

    def train():
        ws_l = np.array(list(labels), np.int64)
        Xl, hl = _vectors(ctx, ws_l)
        yl = np.array([labels[int(w)] >= 0.5 for w in ws_l])
        Xl, yl = Xl[hl], yl[hl]
        clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xl, yl)
        return clf, Xl, yl

    clf, Xl, yl = train()
    wdir = unit(clf.coef_.astype(np.float32))[0]
    _, cb = eng.index.search(wdir.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_b = np.setdiff1d(cb[0][cb[0] >= 0], rows_a)
    VB = unit(eng.vectors.get(rows_b)) if len(rows_b) else np.zeros((0, 768), np.float32)
    wb = np.asarray(eng.work_ids[rows_b], dtype=np.int64)
    okB, detB = ok_publications(wb, VB)
    U_w = np.concatenate([wa[okA], wb[okB]])
    U_V = np.concatenate([VA[okA], VB[okB]])
    del VA, VB
    det_U = {**detA, **detB}
    for r in range(ACTIVE_ROUNDS):
        p = clf.predict_proba(U_V)[:, 1]
        unsure = np.flatnonzero((p > 0.1) & (p < 0.9) & ~np.isin(U_w, list(labels)))
        if not len(unsure):
            break
        pick = unsure[np.argsort(p[unsure])][np.linspace(0, len(unsure) - 1, min(ACTIVE_N, len(unsure))).astype(int)]
        lab = jev.label([_paper(det_U.get(int(U_w[i]))) for i in pick], question)
        labels.update({int(U_w[i]): float(x) for i, x in zip(pick, lab) if not math.isnan(x)})
        job.stage(3, f"round {r + 1}: judged {len(pick)} papers where the classifier was unsure")
        clf, Xl, yl = train()
    p_U = clf.predict_proba(U_V)[:, 1]
    cv = cross_val_predict(LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"), Xl, yl,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(cv, yl)
    report["classifier"] = {"labels": int(len(yl)), "on_topic_labels": int(yl.sum()),
                            "cross_validated_auc": round(float(roc_auc_score(yl, cv)), 3),
                            "pool": int(len(U_w)), "kept": int((p_U >= 0.5).sum()),
                            "expected_on_topic_in_pool": round(float(iso.predict(p_U).sum()))}
    field: dict[int, dict] = {int(w): {"via": "meaning", "p": round(float(x), 3)} for w, x in zip(U_w, p_U) if x >= 0.5}
    if len(field) > TOO_BROAD_PAPERS:
        raise FinalError(f"“{q}” has about {len(field):,} papers: too broad for a whole-field map. Map a narrower topic.")
    for w in list(labels):            # what Jev said overrules the classifier for the papers it judged
        if w in field and labels[w] < 0.5:
            del field[w]
        elif labels[w] >= 0.5 and w not in field:
            field[w] = {"via": "judged", "p": None}
    job.done_stage(3, t)

    # 5 the topic's exact phrase
    t = time.time()
    phrase = keywords or (f'"{q}"' if len(q.split()) <= 6 and not re.search(r'["()*:]|\b(AND|OR|NOT)\b', q) else None)
    report["phrases"] = {"query": phrase}
    if phrase:
        job.stage(4, f"papers with the exact words {phrase}")
        hits, matches, _ = ctx.keywords.search(phrase, 200_000, {}, "all")
        hw = np.array([h[0] for h in hits if h[0] not in field], np.int64)
        Vh, hh = _vectors(ctx, hw)
        ph = clf.predict_proba(Vh[hh])[:, 1] if hh.any() else np.zeros(0)
        keep_v = [int(w) for w, x in zip(hw[hh], ph) if x >= 0.3]
        rest = hw[~hh]
        det_r = _details(ctx, rest)
        rest = np.array([w for w in rest if int(w) in det_r and passes(det_r[int(w)], {})], np.int64)
        lab = jev.label([_paper(det_r.get(int(w))) for w in rest], question) if len(rest) else np.zeros(0)
        keep_j = [int(w) for w, x in zip(rest, lab) if x >= 0.5]
        for w in keep_v:
            field[w] = {"via": "phrase", "p": None}
        for w in keep_j:
            field[w] = {"via": "phrase+judged", "p": None}
        report["phrases"].update({"matches": int(matches), "new": int(len(hw)), "kept_by_classifier": len(keep_v),
                                  "judged_without_vector": int(len(rest)), "kept_by_judgment": len(keep_j)})
    else:
        job.stage(4, "no exact phrase for this topic")
    job.done_stage(4, t)

    # 6 what the field cites
    t = time.time()
    job.stage(5, f"{len(field):,} papers so far; following their references")
    nodes = cit.nodes(np.array(sorted(field), np.int64))
    nodes = nodes[nodes >= 0]
    allrefs = np.concatenate([cit.refs(int(x)) for x in nodes]) if len(nodes) else np.zeros(0, np.uint32)
    u, cnt = np.unique(allrefs, return_counts=True)
    k = CITED_K
    while (cnt >= k).sum() > EXPANSION_MAX:
        k += 1
    in_field = set(field)
    bad_types = {i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES}
    cand_nodes = u[cnt >= k]
    cand_nodes = cand_nodes[~np.isin(np.asarray(cit.kind[cand_nodes]), list(bad_types))]
    cand = np.array([cit.work(int(x)) for x in cand_nodes], np.int64)
    cand = cand[~np.isin(cand, list(in_field))]
    det_c = _details(ctx, cand)
    cand = np.array([w for w in cand if int(w) not in det_c or passes(det_c[int(w)], {})], np.int64)
    Vc, hc = _vectors(ctx, cand)
    pc = np.full(len(cand), np.nan)
    if hc.any():
        pc[hc] = clf.predict_proba(Vc[hc])[:, 1]
    sure = hc & (pc >= 0.9)
    ask = np.flatnonzero(~sure)
    job.stage(5, f"judging {len(ask):,} works cited by ≥ {k} papers of the field")
    lab = jev.label([_paper(det_c.get(int(cand[i])), cit, int(cand[i])) for i in ask], question) if len(ask) else np.zeros(0)
    added = 0
    for i in np.flatnonzero(sure):
        field[int(cand[i])] = {"via": "cited", "p": round(float(pc[i]), 3)}
        added += 1
    for i, x in zip(ask, lab):
        if x >= 0.5:
            field[int(cand[i])] = {"via": "cited+judged", "p": None if math.isnan(pc[i]) else round(float(pc[i]), 3)}
            added += 1
    report["citations"] = {"cited_at_least": k, "candidates": int(len(cand)), "kept_by_classifier": int(sure.sum()),
                           "judged": int(len(ask)), "added": added,
                           "added_without_abstract_vector": int(sum(1 for i, x in zip(ask, lab) if x >= 0.5 and not hc[i]))}
    job.done_stage(5, t)
    report["papers"] = len(field)
    report["by_route"] = {v: sum(1 for f in field.values() if f["via"] == v) for v in sorted({f["via"] for f in field.values()})}
    report["jev"] = {"calls": jev.calls, "failed": jev.failed, "tokens": jev.tokens, "usd": round(jev.usd, 3)}
    return {"version": SELECTION_VERSION, "made": _now(), "release": ctx.release["id"], "topic": q,
            "definition": definition, "keywords": keywords, "report": report,
            "papers": {str(w): f for w, f in field.items()}, "labels": {str(w): round(p, 3) for w, p in labels.items()}}


class FinalError(Exception):
    """The topic cannot be mapped as a whole field (too broad, too unclear): asking again will not help."""


def build_maps(ctx: Ctx, job: Job, sel: dict, version: str) -> dict:
    from srl_search import topics as T
    from srl_search.topic_service import finish
    cit = ctx.cit
    t = time.time()
    job.stage(6, f"grouping {len(sel['papers']):,} papers")
    ws = np.array(sorted(int(w) for w in sel["papers"]), np.int64)
    det = _details(ctx, ws)
    V, has = _vectors(ctx, ws)
    au = cit.authors
    details = []
    for w in ws:
        d = det.get(int(w))
        if d is None:                                  # known only from the citation index (no English abstract)
            nd = int(cit.nodes([int(w)])[0])
            names = []
            if au is not None and nd >= 0:
                rows, pos, _ = au.of(nd)
                names = [au.name(int(r)) for r in rows[np.argsort(pos)][:12]]
            d = {"title": cit.title(nd) if nd >= 0 else None, "year": int(cit.year[nd]) if nd >= 0 else None,
                 "type": cit.type(nd) if nd >= 0 else None, "citations": int(cit.cites[nd]) if nd >= 0 else None,
                 "authors": " | ".join(names), "authors_count": len(names)}
        details.append(d)
    score = np.array([sel["papers"][str(int(w))].get("p") or 0.5 for w in ws], np.float32)
    seed = T.Seed(ws, V, has, details, score)
    seed = seed.subset(T.collapse_versions(seed, cit))
    q = sel["topic"]
    raw = T.build(seed, cit, q, T.Options(refs_top=30))
    out = finish(ctx, cit, raw, seed, ctx.release["id"])
    via = sel["papers"]
    out["papers"]["via"] = [via.get(pid[1:], {}).get("via", "") for pid in out["papers"]["id"]]
    out.update({"question": q, "mode": "field", "filters": {}, "asked_papers": len(seed.wids),
                "field": {**sel["report"], "made": sel["made"], "release": sel["release"]}})
    dirm = job.dir / "maps" / version
    _write(dirm / "main.json", out)
    job.done_stage(6, t)

    t = time.time()
    big = [g for g in out["groups"] if g["size"] >= SPLIT_MIN]
    by_w = {int(w): i for i, w in enumerate(seed.wids)}
    for j, g in enumerate(big):
        job.stage(7, f"splitting group {g['id']} ({g['size']:,} papers), {j + 1} of {len(big)}")
        members = [by_w[int(pid[1:])] for pid, gg in zip(out["papers"]["id"], out["papers"]["group"]) if gg == g["id"]]
        sub_seed = seed.subset(np.array(members))
        sub = T.build(sub_seed, cit, q, T.Options(refs_top=30, min_size=max(8, len(members) // 25), min_groups=2, max_groups=8))
        sub = finish(ctx, cit, sub, sub_seed, ctx.release["id"])
        sub["papers"]["via"] = [via.get(pid[1:], {}).get("via", "") for pid in sub["papers"]["id"]]
        sub.update({"question": q, "mode": "field", "filters": {}, "asked_papers": len(members),
                    "parent": {"group": g["id"], "name": g["name"], "size": g["size"]}})
        _write(dirm / f"group-{g['id']}.json", sub)
    job.done_stage(7, t)
    return out


def run_job(ctx: Ctx, store: Path, key: str) -> None:
    from srl_search.topic_service import VERSION
    job = Job(store, key)
    req = job.st.get("request") or {}
    sel_path = job.dir / "field.json"
    sel = _read(sel_path)
    try:
        if sel is None or job.st.get("kind") == "full":
            jev = Jev(store)
            sel = select_field(ctx, job, req["q"], req.get("definition"), req.get("keywords"), jev)
            _write(sel_path, sel)
        out = build_maps(ctx, job, sel, VERSION)
        job.st.update({"status": "done", "finished": _now(), "eta_s": 0, "detail": "",
                       "summary": {"papers": sel["report"]["papers"], "groups": len(out["groups"]),
                                   "usd": sel["report"]["jev"]["usd"]}})
    except FinalError as e:
        job.st.update({"status": "failed", "final": True, "error": str(e), "finished": _now()})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.st.update({"status": "failed", "final": False, "error": f"the job failed ({type(e).__name__}); it can be asked again",
                       "finished": _now()})
    _write(job.dir / "job.json", job.st)


def worker(store: Path = ROOT) -> None:
    """Run queued jobs one at a time until the queue is empty (started by the srl-fields path unit)."""
    store.mkdir(parents=True, exist_ok=True)
    lock = open(store / "worker.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another worker is running", flush=True)
        return
    ctx = None
    while True:
        jobs = sorted((store / "queue").glob("*.job"), key=lambda p: p.stat().st_mtime)
        if not jobs:
            return
        key = jobs[0].stem
        ctx = ctx or Ctx()
        print(f"job {key}: {(_read(store / key / 'job.json') or {}).get('request')}", flush=True)
        try:
            run_job(ctx, store, key)
        finally:
            jobs[0].unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    if argv[:1] == ["worker"]:
        return worker()
    if argv[:1] == ["run"] and len(argv) >= 2:      # one topic, now, in this process (for tests and experiments)
        from srl_search.topic_service import VERSION
        store = Path(os.environ.get("SRL_FIELDS", str(ROOT)))
        key = field_key(argv[1])
        _write(store / key / "job.json", {"status": "queued", "kind": "full", "key": key, "created": _now(),
                                         "request": {"q": argv[1], "definition": None, "keywords": None}})
        run_job(Ctx(), store, key)
        print(json.dumps(_read(store / key / "job.json"), indent=1))
        return
    print(__doc__)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main(sys.argv[1:])
