"""Whole-field topic maps: every paper of a research field, not the 1,000 closest to a question.

Why (experiments/topicmap/README.md): the 1,000 closest papers are a corner of a field of ~10-60k; no distance
cut-off separates a field from its neighbours; exact phrases miss half of it; and ~30% of the studies meta-analyses
cite have no English abstract, so only the field's own citations reach them. How, for <= $0.50 of AI labels and
without Jev (TypeSafe's terms forbid training on its answers; srl_search/judges.py):

  1 teacher      the 300k papers closest in meaning to the topic; Kimi K3 judges 375 of them, spread from nearest
                 to farthest (~$0.49)
  2 candidates   a classifier on meaning vectors trained on those answers searches a second 300k along its own
                 direction; the topic's exact phrase; works cited >= 3 times by the likely field
  3 decider      Qwen3.5-0.8B learns the teacher's judgments (yes/no head, LoRA, ~2.5 min on GPU 1)
  4 decider labels  it judges ~10,000 candidates where it matters (vLLM, ~2 min)
  5 students     the vector classifier again, and Ettin-17M on text, both taught by the teacher and the decider;
                 Ettin judges papers with a title only, and those the vector classifier is unsure about
  6 judging      every candidate: the teacher's or decider's own answer, else the students'
  7 map          the topic map of all of it (srl_search/topics.py; Leiden runs on 24 cores)
  8 subgroups    every group of 300+ papers split into its own groups

The search API only reads results and queues jobs (FieldStore); the worker (`python -m srl_search.field worker`,
started by the srl-fields path unit) runs them one at a time. GPU steps wait for free memory on GPU 1 and never stop
the household services there. Results are kept across nightly releases: the selected field (field.json) is the
expensive part; maps are rebuilt from it when the map code changes (~5 min).

Store: ROOT/<key>/job.json (status), field.json (the papers, how each was found, the labels), work/ (the job's
models and files), maps/<code version>/main.json and group-<n>.json; ROOT/queue/<key>.job (jobs to run);
ROOT/spend.json (teacher dollars per day); ROOT/PAUSED (new fields refused, with the reason).
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
SELECTION_VERSION = "2"                  # 2: Kimi K3 teacher + 0.8B decider + students (no Jev); 1: Jev
TEACHER_USD = float(os.environ.get("SRL_FIELD_TEACHER_USD", "0.50"))   # Kimi K3 labels per field
TEACHER_N = 340                          # ~$0.48 at $1.42 per 1,000 papers (measured in the job: longer prompt)
DECIDER_N = 10_000                       # papers the decider labels for the students
BUDGET_USD = float(os.environ.get("SRL_FIELD_BUDGET_USD", "10"))    # teacher spend a day, all jobs together
EST_USD = TEACHER_USD                    # what a new field is assumed to cost when deciding to accept it
MAX_QUEUED = 3
POOL = 300_000
MIN_ON_TOPIC = 15                        # teacher's on-topic answers needed to go on
CITED_K = 3
EXPANSION_MAX = 300_000
BAND = (0.2, 0.8)                        # the vector classifier is unsure here: the text student judges too
TOO_BROAD_TAIL = 0.30                    # share on topic among the farthest of the 300k: the field goes on beyond
TOO_BROAD_PAPERS = 150_000
SPLIT_MIN = 300                          # groups at least this big get their own subgroups
LEIDEN_WORKERS = int(os.environ.get("SRL_FIELDS_WORKERS", "24"))
STAGES = [("teacher", 40), ("candidates", 150), ("decider", 200), ("decider labels", 150), ("students", 90),
          ("judging", 120), ("map", 200), ("subgroups", 120)]


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


def paused(root: Path) -> str | None:
    """New fields are paused when ROOT/PAUSED exists (its text says why). Maps of fields already selected can
    still be rebuilt: that uses no AI judgments."""
    try:
        return (root / "PAUSED").read_text().strip() or "New whole-field maps are paused."
    except OSError:
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
        if kind == "full" and paused(self.root):
            raise HTTPException(503, paused(self.root) + " Whole-field maps already made still open; the 1,000-paper "
                                     "map works for any topic.", headers={"Retry-After": "86400"})
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


def _spend(root: Path, usd: float) -> None:
    """Add to today's teacher spend (all jobs, for the daily budget)."""
    path = root / "spend.json"
    with open(root / "spend.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        led = _read(path) or {}
        today = date.today().isoformat()
        led[today] = round(led.get(today, 0.0) + usd, 4)
        _write(path, led)


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


def select_field(ctx: Ctx, job: Job, q: str, definition: str | None, keywords: str | None) -> dict:
    import faiss
    from sklearn.linear_model import LogisticRegression
    from srl_search import judges as J
    from srl_search.engine import passes
    from srl_search.scope import NON_PUBLICATION_TYPES
    eng, cit = ctx.engine, ctx.cit
    faiss.omp_set_num_threads(16)
    report: dict = {"topic": q, "definition": definition, "teacher": "Kimi K3", "decider": J.DECIDER_BASE,
                    "student": J.STUDENT_BASE}
    work = job.dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    say = lambda detail: job.stage(job.st.get("stage_index", 0), detail)

    def texts_of(ws) -> list[str]:
        det = _details(ctx, ws)
        out = []
        for w in ws:
            d = det.get(int(w))
            if d:
                out.append(J.paper_text(d.get("title"), d.get("abstract")))
            else:
                nd = int(cit.nodes([int(w)])[0])
                out.append(J.paper_text(cit.title(nd) if nd >= 0 else "", None))
        return out

    def lr(ws_y: dict[int, float]):
        ws = np.array(list(ws_y), np.int64)
        X, h = _vectors(ctx, ws)
        y = np.array([ws_y[int(w)] >= 0.5 for w in ws])
        return LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(X[h], y[h])

    # 1 the teacher labels papers spread from the nearest to the farthest of the 300k closest to the topic
    t = time.time()
    job.stage(0, f"gathering the {POOL:,} papers closest to the topic")
    qv = eng.encoder.encode(q)[0]
    _, ca = eng.index.search(qv.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_a = ca[0][ca[0] >= 0]
    VA = unit(eng.vectors.get(rows_a))
    wa = np.asarray(eng.work_ids[rows_a], dtype=np.int64)
    o = np.argsort(-(VA @ qv), kind="stable")
    rows_a, VA, wa = rows_a[o], VA[o], wa[o]
    okA = ctx.quality.junk(wa, VA) < 0.5
    idxA = np.flatnonzero(okA)
    n = len(idxA)
    edges = sorted({0, *[int(n * f) for f in (.002, .004, .007, .01, .017, .025, .035, .05, .07, .1, .14, .2, .27, .4,
                                                .55, .75)], n})
    rng = np.random.default_rng(0)
    per = math.ceil(TEACHER_N * 1.25 / (len(edges) - 1))          # 25% spare: some are not publications
    sample, bucket = [], []
    for bi, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if hi > lo:
            pick = rng.choice(np.arange(lo, hi), size=min(per, hi - lo), replace=False)
            sample += idxA[pick].tolist()
            bucket += [bi] * len(pick)
    sw = wa[sample]
    det = _details(ctx, sw)
    keep = [k for k, w in enumerate(sw) if int(w) in det and passes(det[int(w)], {})]
    keep = sorted(rng.choice(keep, size=min(TEACHER_N, len(keep)), replace=False).tolist())
    sw, bucket = sw[keep], np.array(bucket)[keep]
    job.stage(0, f"the teacher (Kimi K3) judges {len(sw)} papers")
    teacher = J.Teacher(TEACHER_USD)
    lab = teacher.label([J.paper_text(det[int(w)].get("title"), det[int(w)].get("abstract")) for w in sw], q, definition)
    _spend(job.dir.parent, teacher.usd)
    labels = {int(w): float(x) for w, x in zip(sw, lab) if not math.isnan(x)}
    on = np.nan_to_num(lab) >= 0.5
    last = bucket == bucket.max()
    tail = float(on[last].mean()) if last.any() else 0.0
    report["teacher_labels"] = {"papers": len(labels), "on_topic": int(on.sum()), "failed": teacher.failed,
                                "usd": round(teacher.usd, 3), "farthest_on_topic": round(tail, 3)}
    if on.sum() < MIN_ON_TOPIC:
        raise FinalError(f"Only {int(on.sum())} of {len(sw)} papers judged near “{q}” are about it: too few to find the "
                         "field. Phrase the topic as papers about it would, or give a definition.")
    if tail > TOO_BROAD_TAIL:
        raise FinalError(f"“{q}” is broader than a whole-field map can hold today: {round(100 * tail)}% of the papers at "
                         f"the edge of the {POOL:,} closest are still about it. Map a narrower topic, or use the "
                         "1,000-paper map.")
    job.done_stage(0, t)

    # 2 candidates: a classifier on meaning (teacher labels) searches a second pool along its direction; the exact
    #   phrase; what the likely field cites. Nothing is judged for good yet.
    t = time.time()
    job.stage(1, "searching along a first classifier; following citations")
    lr0 = lr(labels)
    wdir = unit(lr0.coef_.astype(np.float32))[0]
    _, cb = eng.index.search(wdir.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_b = np.setdiff1d(cb[0][cb[0] >= 0], rows_a)
    VB = unit(eng.vectors.get(rows_b)) if len(rows_b) else np.zeros((0, 768), np.float32)
    wb = np.asarray(eng.work_ids[rows_b], dtype=np.int64)
    okB = ctx.quality.junk(wb, VB) < 0.5 if len(wb) else np.zeros(0, bool)
    U_w = np.concatenate([wa[okA], wb[okB]])
    U_V = np.concatenate([VA[okA], VB[okB]])
    del VA, VB
    p0 = lr0.predict_proba(U_V)[:, 1]
    phrase = keywords or (f'"{q}"' if len(q.split()) <= 6 and not re.search(r'["()*:]|\b(AND|OR|NOT)\b', q) else None)
    H = np.zeros(0, np.int64)
    if phrase:
        hits, matches, _ = ctx.keywords.search(phrase, 200_000, {}, "all")
        H = np.setdiff1d(np.array([h[0] for h in hits], np.int64), U_w)
        report["phrases"] = {"query": phrase, "matches": int(matches), "outside_pool": int(len(H))}
    likely = np.concatenate([U_w[p0 >= 0.5], H])
    nodes = cit.nodes(np.unique(likely))
    nodes = nodes[nodes >= 0]
    allrefs = np.concatenate([cit.refs(int(x)) for x in nodes]) if len(nodes) else np.zeros(0, np.uint32)
    u, cnt = np.unique(allrefs, return_counts=True)
    k = CITED_K
    while (cnt >= k).sum() > EXPANSION_MAX:
        k += 1
    bad_types = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]
    cn = u[cnt >= k]
    cn = cn[~np.isin(np.asarray(cit.kind[cn]), bad_types)]
    C = np.setdiff1d(np.array([cit.work(int(x)) for x in cn], np.int64), np.concatenate([U_w, H]))
    report["candidates"] = {"pool": int(len(U_w)), "likely_by_first_classifier": int((p0 >= 0.5).sum()),
                            "phrase_outside_pool": int(len(H)), "cited_at_least": k, "cited_candidates": int(len(C))}
    job.done_stage(1, t)

    # 3 the decider learns from the teacher
    t = time.time()
    job.stage(2, "teaching the 0.8B decider")
    lw = list(labels)
    J.train_decider(list(zip(texts_of(lw), [labels[w] for w in lw])), q, definition, work / "decider", say)
    job.done_stage(2, t)

    # 4 the decider labels papers where judgments matter: half where the first classifier is unsure, the rest spread
    #   over its scores, and works the field cites (many have only a title)
    t = time.time()
    job.stage(3, f"the decider judges {DECIDER_N:,} papers")
    band = np.flatnonzero((p0 > 0.1) & (p0 < 0.9))
    spread = np.argsort(p0)[np.linspace(0, len(p0) - 1, len(p0)).astype(int)]
    n_band, n_cited = DECIDER_N // 2, min(len(C), DECIDER_N // 5)
    pick_u = set(rng.choice(band, size=min(n_band, len(band)), replace=False).tolist()) if len(band) else set()
    rest = [i for i in rng.choice(spread, size=min(len(spread), DECIDER_N), replace=False) if i not in pick_u]
    pick_u |= set(rest[:DECIDER_N - n_cited - len(pick_u)])
    dws = [int(U_w[i]) for i in sorted(pick_u)] + [int(w) for w in rng.choice(C, size=n_cited, replace=False)]
    dws = [w for w in dws if w not in labels]
    dtexts = texts_of(dws)
    dp = J.decider_scores(work / "decider", dtexts, say)
    decided = dict(zip(dws, dp.tolist()))
    report["decider_labels"] = {"papers": len(decided), "on_topic": int((dp >= 0.5).sum())}
    job.done_stage(3, t)

    # 5 students: the vector classifier and Ettin-17M learn from the teacher's and the decider's labels; Ettin judges
    #   what has no vector (title only) and what the vector classifier is unsure about
    t = time.time()
    job.stage(4, "training the students")
    taught = {**decided, **labels}                               # the teacher's answer wins where both exist
    lr1 = lr(taught)
    p1 = lr1.predict_proba(U_V)[:, 1]
    _, hH = _vectors(ctx, H)
    _, hC = _vectors(ctx, C)
    XH, _ = _vectors(ctx, H[hH]) if hH.any() else (np.zeros((0, 768), np.float32), None)
    XC, _ = _vectors(ctx, C[hC]) if hC.any() else (np.zeros((0, 768), np.float32), None)
    pH = np.full(len(H), np.nan)
    pC = np.full(len(C), np.nan)
    if len(XH):
        pH[hH] = lr1.predict_proba(XH)[:, 1]
    if len(XC):
        pC[hC] = lr1.predict_proba(XC)[:, 1]
    unsure = lambda p: np.isnan(p) | ((p > BAND[0]) & (p < BAND[1]))
    E = np.concatenate([U_w[unsure(p1)], H[unsure(pH)], C[unsure(pC)]])
    E = np.array([w for w in E if int(w) not in taught], np.int64)
    say(f"the text student judges {len(E):,} papers (title only, or unsure)")
    tw = list(taught)
    pe = J.student_scores(list(zip(texts_of(tw), [taught[w] for w in tw])), texts_of(E), work / "student", say) \
        if len(E) else np.zeros(0)
    by_e = dict(zip(E.tolist(), pe.tolist()))
    job.done_stage(4, t)

    # 6 judging: the teacher's and the decider's own answers where they exist; else the vector classifier, averaged
    #   with the text student where it was unsure; the text student alone without a vector
    t = time.time()
    job.stage(5, "judging every candidate")

    def final(ws, p_vec):
        out = {}
        for w, pv in zip(ws.tolist(), p_vec.tolist()):
            if w in taught:
                p = taught[w]
            elif w in by_e:
                p = by_e[w] if math.isnan(pv) else (pv + by_e[w]) / 2
            else:
                p = pv
            if p == p and p >= 0.5:
                out[w] = round(float(p), 3)
        return out
    route = {}
    for name, ws, pv in (("meaning", U_w, p1), ("phrase", H, pH), ("cited", C, pC)):
        for w, p in final(ws, pv).items():
            route.setdefault(w, (name, p))
    ids = np.array(list(route), np.int64)
    det = _details(ctx, ids)
    field = {}
    for w in ids.tolist():                                       # publications only, not retracted
        d = det.get(w)
        if d is not None and not passes(d, {}):
            continue
        if d is None:
            nd = int(cit.nodes([w])[0])
            if nd < 0 or cit.type(nd) in NON_PUBLICATION_TYPES:
                continue
        field[w] = {"via": route[w][0], "p": route[w][1]}
    if len(field) > TOO_BROAD_PAPERS:
        raise FinalError(f"“{q}” has about {len(field):,} papers: too broad for a whole-field map today. Map a narrower topic.")
    report["papers"] = len(field)
    report["by_route"] = {v: sum(1 for f in field.values() if f["via"] == v) for v in ("meaning", "phrase", "cited")}
    report["judged_by_text_student"] = int(len(E))
    report["teacher_usd"] = round(teacher.usd, 3)
    job.done_stage(5, t)
    return {"version": SELECTION_VERSION, "made": _now(), "release": ctx.release["id"], "topic": q,
            "definition": definition, "keywords": keywords, "report": report,
            "papers": {str(w): f for w, f in field.items()},
            "labels": {"teacher": {str(w): v for w, v in labels.items()},
                       "decider": {str(w): round(v, 3) for w, v in decided.items()}}}


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
    T.PARALLEL_MIN = 1500                              # the worker owns its cores: Leiden runs side by side
    raw = T.build(seed, cit, q, T.Options(refs_top=30, workers=LEIDEN_WORKERS))
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
        sub = T.build(sub_seed, cit, q, T.Options(refs_top=30, min_size=max(8, len(members) // 25), min_groups=2,
                                                  max_groups=8, workers=LEIDEN_WORKERS))
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
            if paused(store):                          # queued before the pause: not run
                raise FinalError(paused(store))
            sel = select_field(ctx, job, req["q"], req.get("definition"), req.get("keywords"))
            _write(sel_path, sel)
        out = build_maps(ctx, job, sel, VERSION)
        job.st.update({"status": "done", "finished": _now(), "eta_s": 0, "detail": "",
                       "summary": {"papers": sel["report"]["papers"], "groups": len(out["groups"]),
                                   "usd": sel["report"].get("teacher_usd", sel["report"].get("jev", {}).get("usd"))}})
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
