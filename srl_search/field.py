"""Whole-field topic maps: every paper of a research field, not the 1,000 closest to a question.

Why (experiments/topicmap/README.md): the 1,000 closest papers are a corner of a field of ~10-60k; no distance
cut-off separates a field from its neighbours; exact phrases miss half of it; and ~30% of the studies meta-analyses
cite have no English abstract, so only the field's own citations reach them. How, for ~$1 of AI answers: Kimi K3
teaches small models (its providers allow training on its answers); Jev only checks papers and teaches nothing
(srl_search/judges.py):

  1 teacher      the 300k papers closest in meaning to the topic; Kimi K3 judges 272 papers spread from nearest
                 to farthest, a quarter of them works the nearest papers cite (many with a title only)
  2 candidates   a classifier on meaning vectors trained on those answers searches a second 300k along its own
                 direction; the topic's exact phrase; works cited >= 3 times by the likely field. Kimi K3 then
                 judges 128 more where that classifier is unsure (active learning): 400 answers, ~$0.49
  3 decider      Qwen3.5-0.8B learns all 400 answers (yes/no head, LoRA, ~1.5 min on GPU 1)
  4 decider labels  it judges 10,000 candidates (vLLM, ~2.5 min)
  5 students     a classifier on meaning vectors and Ettin-17M on text learn the decider's probabilities; Ettin
                 judges papers with a title only, those the vector student is unsure about, and the teacher's
  6 judging      a combiner (student score + citation signals, fitted on the teacher's answers) scores every
                 candidate; the teacher's and decider's own answers come first; the cut-off maximises the F1
                 expected from those probabilities; then Jev checks the well-cited papers near the cut-off
                 (scores 0.1-0.7, cited 20+ times; ~20k papers, ~$0.55): its answers decide them, nothing learns
                 from them (the classics in older words, and the neighbours that would top reading lists)
  7 map          the topic map of all of it (srl_search/topics.py; Leiden runs on 24 cores)
  8 subgroups    every group of 300+ papers split into its own groups

The search API only reads results and queues jobs (FieldStore); the worker (`python -m srl_search.field worker`,
started by the srl-fields path unit) runs them one at a time. GPU steps wait for free memory on GPU 1 and never stop
the household services there. Results are kept across nightly releases: the selected field (field.json) is the
expensive part; maps are rebuilt from it when the map code changes (~5 min).

Store: ROOT/<key>/job.json (status), field.json (the papers, how each was found, the labels), work/ (the job's
models and files), maps/<code version>/main.json and group-<n>.json; ROOT/queue/<key>.job (jobs to run);
ROOT/spend.json (Kimi and Jev dollars per day); ROOT/PAUSED (new fields refused, with the reason).
"""
from __future__ import annotations

import contextvars
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException

ROOT = Path(os.environ.get("SRL_FIELDS", "/mnt/fast/scholarsreadinglist/search/fields"))
SELECTION_VERSION = "5"                  # 5: + Jev checks well-cited papers near the cut-off; 4: 400 teacher answers in two rounds (active learning), students learn
                                         #    probabilities, citation combiner, expected-F1 cut-off; 3: held-back
                                         #    cut-offs; 2: first no-Jev recipe; 1: Jev
TEACHER_USD = float(os.environ.get("SRL_FIELD_TEACHER_USD", "0.50"))   # Kimi K3 labels per field
TEACHER_R1 = 272                         # first round, spread; ~$1.22 per 1,000 papers (measured)
TEACHER_R2 = 128                         # second round, where the first classifier is unsure: 400 answers ~$0.49
ACTIVE_BAND = (0.35, 0.85)               # the first classifier (class-balanced) is unsure here
CHECK_BAND = (0.1, 0.7)                  # Jev decides the papers scoring here that are cited CHECK_CITES+ times:
CHECK_CITES = 20                         #   the classics just below the cut-off, the wrong ones just above it
CHECK_USD = float(os.environ.get("SRL_FIELD_CHECK_USD", "0.60"))   # Jev per field (~$0.027 per 1,000 papers);
                                         #   most cited first when a field has more
DECIDER_N = 10_000                       # papers the decider labels for the students
BUDGET_USD = float(os.environ.get("SRL_FIELD_BUDGET_USD", "60"))    # AI spend a day, all jobs: a safety cap
FREE_USD = float(os.environ.get("SRL_FIELD_FREE_USD", "10"))        # AI spend a day for the free line
EST_USD = 1.0                            # a new field's AI cost (v5: Kimi ~$0.50 + Jev $0.04-0.60), for planning
FREE_MAX_QUEUED = 60                     # maps waiting in the free line
PER_ASKER = 2                            # free maps one reader (the door's per-address code) may have waiting
POOL = 300_000
MIN_ON_TOPIC = 15                        # teacher's on-topic answers needed to go on
CITED_K = 3
EXPANSION_MAX = 300_000
BAND = (0.1, 0.9)                        # the vector student is unsure here: the text student judges too
TOO_BROAD_TAIL = 0.30                    # share on topic among the farthest of the 300k: the field goes on beyond
TOO_BROAD_PAPERS = 150_000
SPLIT_MIN = 300                          # groups at least this big get their own subgroups
LEIDEN_WORKERS = int(os.environ.get("SRL_FIELDS_WORKERS", "24"))
STAGES = [("teacher", 40), ("candidates", 60), ("decider", 110), ("decider labels", 160), ("students", 170),
          ("judging", 40), ("map", 250), ("subgroups", 220)]


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
# Who asked (the door's anonymous per-address code, X-SRL-Asker): set per request by the API, read by FieldStore.get
ASKER: contextvars.ContextVar[str | None] = contextvars.ContextVar("srl_field_asker", default=None)
RESERVED = Path("/run/gpu1-reserved")          # Figurer training has the GPU (srl_search/judges.py waits for it)


def _ts(iso: str | None) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else 0.0
    except ValueError:
        return 0.0


def _next_midnight(t: float) -> float:
    """The next local midnight after t (the free line's budget is per local day)."""
    return datetime.combine(datetime.fromtimestamp(t).date() + timedelta(days=1), datetime.min.time()).timestamp()


class FieldStore:
    """What the search API does with whole-field maps: read finished ones, report progress, queue new jobs; and the
    plan the worker follows. Two lines: maps paid for (ROOT/paid/<key>.json, written by the site's door once Stripe
    says the payment went through) go first, in the order they were paid; the free line follows in the order asked,
    as long as today's free budget (FREE_USD of AI) lasts, then tomorrow's. Map rebuilds (no AI) go with the free
    line but need no budget."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        (self.root / "queue").mkdir(parents=True, exist_ok=True)

    def spent_today(self, lane: str | None = None) -> float:
        today = date.today().isoformat()
        if lane is None:
            return float((_read(self.root / "spend.json") or {}).get(today, 0.0))
        return float(((_read(self.root / "spend_lanes.json") or {}).get(today) or {}).get(lane, 0.0))

    def queued(self) -> list[str]:
        return [p.stem for p in sorted((self.root / "queue").glob("*.job"), key=lambda p: p.stat().st_mtime)]

    def paid(self, key: str) -> dict | None:
        """The payment that moved this map to the paid line (accepted: the map was waiting when it was paid)."""
        m = _read(self.root / "paid" / f"{key}.json")
        return m if m and m.get("accepted") else None

    def credit(self, key: str) -> str | None:
        m = self.paid(key)
        return (m.get("name") or None) if m and m.get("name_ok") else None

    def lane(self, key: str, st: dict) -> str:
        if self.paid(key):
            return "paid"
        return "rebuild" if st.get("kind") == "map" else "free"

    def plan(self, now: float | None = None) -> dict:
        """When each waiting map should start and be ready, by simulating the worker: the running map finishes;
        then paid maps, then rebuilds and the free line while today's free budget lasts (an estimate: maps paid for
        later go ahead of the free line)."""
        now = now or time.time()
        running, waiting = None, []
        for k in self.queued():
            st = _read(self.root / k / "job.json") or {}
            if st.get("status") == "running":
                left = max(60.0, float(st.get("eta_s") or 0) - (now - _ts(st.get("updated"))))
                running = {"key": k, "st": st, "left": left, "lane": st.get("lane") or self.lane(k, st)}
            elif st.get("status") == "queued":
                waiting.append({"key": k, "st": st, "lane": self.lane(k, st)})
        prio = {"paid": 0, "rebuild": 1, "free": 1}
        waiting.sort(key=lambda j: (prio[j["lane"]], _ts((self.paid(j["key"]) or {}).get("paid_at"))
                                    if j["lane"] == "paid" else _ts(j["st"].get("created"))))
        t = now + (running["left"] if running else 60)       # a worker starting loads ~15 GB first (~1 min)
        free_left = FREE_USD - self.spent_today("free") - (0.5 * EST_USD if running and running["lane"] == "free" else 0)
        todo, out = list(waiting), []
        while todo:
            pick = next((j for j in todo if j["lane"] != "free" or free_left >= 0.5 * EST_USD), None)
            if pick is None:                                 # only free maps left and today's budget is spent
                t, free_left = _next_midnight(t), FREE_USD
                continue
            todo.remove(pick)
            if pick["lane"] == "free":
                free_left -= EST_USD
            dur = job_seconds(pick["st"].get("kind"))
            out.append({**pick, "start": t, "ready": t + dur})
            t += dur
        paid_ahead = sum(job_seconds(j["st"].get("kind")) for j in out if j["lane"] == "paid")
        # the GPU is taken (Figurer's training reservation, or the running map waits for memory other work holds):
        # no one can say when it frees, so no promise of a time, and no offer to pay for one
        gpu_busy = RESERVED.exists() or bool(running and str(running["st"].get("detail", "")).startswith("waiting for"))
        return {"now": now, "running": running, "waiting": out,
                "paid_ready_s": (running["left"] if running else 60) + paid_ahead + job_seconds("full"),
                "gpu_reserved": gpu_busy}

    def status(self, key: str) -> dict | None:
        st = _read(self.root / key / "job.json")
        if not st:
            return None
        st = {k: v for k, v in st.items() if k != "asker"}
        if st.get("status") in ("queued", "running"):
            pl = self.plan()
            mine = next((j for j in pl["waiting"] if j["key"] == key), None)
            st["lane"] = st.get("lane") or self.lane(key, st)
            if mine:
                same = [j for j in pl["waiting"] if j["lane"] == mine["lane"] or {j["lane"], mine["lane"]} == {"free", "rebuild"}]
                st["position"] = [j["key"] for j in same].index(key) + 1
                st["start_s"] = round(mine["start"] - pl["now"])
                st["eta_s"] = round(mine["ready"] - pl["now"])
                if mine["lane"] != "paid":
                    st["paid_eta_s"] = round(pl["paid_ready_s"])
            st["gpu_reserved"] = pl["gpu_reserved"]
            st["free_per_day"] = round(FREE_USD / EST_USD)
        if self.credit(key):
            st["credit"] = self.credit(key)
        return st

    def get(self, q: str, definition: str | None, keywords: str | None, version: str, group: int | None,
            start: bool = True) -> dict:
        key = field_key(q, definition, keywords)
        d = self.root / key
        if (d / "maps" / version / "main.json").exists():
            return self._serve(key, d / "maps" / version, group)
        st = self.status(key)
        older = self._older(d, version)
        if older is not None:
            # made by older map code: it opens as it is while the current code remakes it (no AI, minutes)
            failed_here = bool(st and st.get("status") == "failed" and st.get("version") == version)
            if not (st and st.get("status") in ("queued", "running")) and not failed_here and (d / "field.json").exists():
                prev = _read(d / "job.json") or {}
                _write(d / "job.json", {"status": "queued", "kind": "map", "key": key, "created": _now(),
                                        "version": version, "request": {"q": q, "definition": definition,
                                                                        "keywords": keywords},
                                        **{k: prev[k] for k in ("summary", "finished") if k in prev}})
                (self.root / "queue" / f"{key}.job").write_text(key)
            out = self._serve(key, older, group)
            out["field"] = {**out.get("field", {}), "rebuilding": True}
            return out
        if st and st.get("status") in ("queued", "running"):
            raise Pending(st)
        if st and st.get("status") == "failed" and st.get("final"):
            raise HTTPException(422, st.get("error") or "this topic cannot be mapped as a whole field")
        if not start:
            raise HTTPException(409, "No whole-field map of this topic yet: ask for it without read= to start one.")
        kind = "map" if (d / "field.json").exists() else "full"
        asker = ASKER.get()
        if kind == "full" and paused(self.root):
            raise HTTPException(503, paused(self.root) + " Whole-field maps already made still open; the 1,000-paper "
                                     "map works for any topic.", headers={"Retry-After": "86400"})
        if kind == "full":
            waiting = [(_read(self.root / k / "job.json") or {}) for k in self.queued()]
            free = [w for w in waiting if w.get("status") == "queued" and w.get("kind") == "full"
                    and not self.paid(w.get("key", ""))]
            if len(free) >= FREE_MAX_QUEUED:
                raise HTTPException(503, f"The free line is full ({FREE_MAX_QUEUED} whole-field maps waiting). Try again "
                                         "in a few hours; the 1,000-paper map of the topic works right away.",
                                    headers={"Retry-After": "7200"})
            if asker and sum(1 for w in free if w.get("asker") == asker) >= PER_ASKER:
                raise HTTPException(429, f"You already have {PER_ASKER} whole-field maps waiting in the free line: ask "
                                         "for another once one of them is made.", headers={"Retry-After": "3600"})
        st = {"status": "queued", "kind": kind, "key": key, "created": _now(), "version": version,
              "request": {"q": q, "definition": definition, "keywords": keywords}}
        if asker:
            st["asker"] = asker
        _write(d / "job.json", st)
        (self.root / "queue" / f"{key}.job").write_text(key)
        raise Pending(self.status(key) or st)

    @staticmethod
    def _older(d: Path, version: str) -> Path | None:
        """The newest map of this topic made by other (older) map code, if any."""
        dirs = [m.parent for m in (d / "maps").glob("*/main.json") if m.parent.name != version]
        return max(dirs, key=lambda x: (x / "main.json").stat().st_mtime) if dirs else None

    def _serve(self, key: str, dirm: Path, group: int | None) -> dict:
        main = dirm / "main.json"
        if group is None:
            out = _read(main)
            if self.credit(key):
                out["field"] = {**out.get("field", {}), "credit": self.credit(key)}
            return out
        sub = dirm / f"group-{group}.json"
        if sub.exists():
            return _read(sub)
        m = _read(main)
        n = len(m.get("groups", []))
        if not 1 <= group <= n:
            raise HTTPException(404, f"no group {group}; this map has groups 1-{n}")
        raise HTTPException(404, f"group {group} has fewer than {SPLIT_MIN} papers and was not split further: "
                                 f"read it in full instead (read={group})")

    def queue_view(self) -> dict:
        """The public line: what is being made, what waits (topics are public, as the maps will be), today's count."""
        pl = self.plan()

        def row(j: dict, pos: int | None = None) -> dict:
            st, req = j["st"], j["st"].get("request") or {}
            r = {"key": j["key"], "topic": req.get("q"), "definition": req.get("definition"), "line": j["lane"],
                 "kind": st.get("kind"), "asked": st.get("created")}
            if pos is not None:
                r.update({"position": pos, "start_s": round(j["start"] - pl["now"]), "ready_s": round(j["ready"] - pl["now"])})
            if self.credit(j["key"]):
                r["credit"] = self.credit(j["key"])
            return r
        run = pl["running"]
        today = date.today().isoformat()
        made_today = sum(1 for p in self.root.glob("*/job.json")
                         if ((_read(p) or {}).get("status") == "done"
                             and str((_read(p) or {}).get("finished", "")).startswith(today)))
        return {"running": {**row(run), "stage": run["st"].get("stage"), "stage_index": run["st"].get("stage_index"),
                            "stages": len(STAGES), "ready_s": round(run["left"])} if run else None,
                "waiting": [row(j, i + 1) for i, j in enumerate(pl["waiting"])],
                "paid_ready_s": round(pl["paid_ready_s"]), "gpu_reserved": pl["gpu_reserved"],
                "free_per_day": round(FREE_USD / EST_USD), "made_today": made_today}

    def made(self) -> list[dict]:
        """Every whole-field map made, newest first (the public library)."""
        out = []
        for p in self.root.glob("*/job.json"):
            st = _read(p) or {}
            if not (p.parent / "field.json").exists():
                continue
            if st.get("status") != "done" and st.get("kind") != "map":   # a map being rebuilt (new map code) stays listed
                continue
            req, summ = st.get("request") or {}, st.get("summary") or {}
            out.append({"key": p.parent.name, "topic": req.get("q"), "definition": req.get("definition"),
                        "papers": summ.get("papers"), "groups": summ.get("groups"), "made": st.get("finished"),
                        "credit": self.credit(p.parent.name)})
        return sorted(out, key=lambda r: r.get("made") or "", reverse=True)

    def remove(self, key: str, reason: str) -> None:
        """Moderation: take a waiting topic out of the line (its text is public)."""
        (self.root / "queue" / f"{key}.job").unlink(missing_ok=True)
        st = _read(self.root / key / "job.json") or {}
        st.update({"status": "failed", "final": True, "error": reason, "finished": _now()})
        _write(self.root / key / "job.json", st)


STAGES_TOTAL = sum(s for _, s in STAGES)


def job_seconds(kind: str | None) -> int:
    """Typical length: a new field runs every stage; a map job (the field kept, the map code changed) the last two."""
    return sum(s for _, s in STAGES[6:]) if kind == "map" else STAGES_TOTAL


def pending_text(st: dict, q: str, base: str) -> str:
    """What an AI or a person reads while a whole-field map is being made."""
    eta = int(st.get("eta_s") or job_seconds(st.get("kind")))
    line = "the paid line" if st.get("lane") == "paid" else "the free line"
    where = (f"{st['position']} in {line}" if st.get("status") == "queued" and st.get("position")
             else f"step {st.get('stage_index', 0) + 1} of {len(STAGES)} ({st.get('stage', 'starting')})")
    when = f"about {max(1, round(eta / 60))} more minutes" if eta < 5400 else f"about {round(eta / 3600)} hours"
    return (f"# Whole-field map of “{q}”: being made\n\n"
            f"Status: {st.get('status')}, {where}; ready in {when}. "
            + (f"{st['detail']}. " if st.get("detail") else "") +
            "It finds every paper of the field (not just the 1,000 closest): papers close in meaning, judged one by "
            "one by an AI classifier, the topic's exact phrases, and the studies the field cites (which reach papers "
            "without abstracts), then maps them into research groups and subgroups. Once made, it is kept and "
            "opens instantly for everyone.\n\n"
            f"Ask again later (the same request). Meanwhile, the map of the 1,000 closest papers works right away: "
            f"{base}/topics/{urllib.parse.quote(q)}\n")


def _spend(root: Path, usd: float, lane: str = "free") -> None:
    """Add to today's AI spend: spend.json (all jobs, for the safety cap) and spend_lanes.json (by line: the free
    line has its own daily budget; paid maps pay for themselves)."""
    with open(root / "spend.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        today = date.today().isoformat()
        led = _read(root / "spend.json") or {}
        led[today] = round(led.get(today, 0.0) + usd, 4)
        _write(root / "spend.json", led)
        lanes = _read(root / "spend_lanes.json") or {}
        day = lanes.setdefault(today, {})
        day[lane] = round(day.get(lane, 0.0) + usd, 4)
        _write(root / "spend_lanes.json", lanes)


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


def _plugin_cut(p: np.ndarray) -> tuple[float, float]:
    """The cut-off that maximises the F1 expected from probabilities `p` of every candidate (no labels needed; Lipton et
    al. 2014: with calibrated probabilities the best cut is about half the best F1): (cut, expected F1)."""
    o = np.sort(p)[::-1]
    f1 = 2 * np.cumsum(o) / (np.arange(1, len(o) + 1) + o.sum())
    k = int(np.argmax(f1))
    return round(float(o[k]), 3), round(float(f1[k]), 3)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def select_field(ctx: Ctx, job: Job, q: str, definition: str | None, keywords: str | None) -> dict:
    import faiss
    from scipy.optimize import brentq
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
    rng = np.random.default_rng(0)
    bad_types = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]

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

    def pairs_of(ws) -> list[tuple[str, str | None]]:
        det = _details(ctx, ws)
        out = []
        for w in ws:
            d = det.get(int(w))
            if d:
                out.append((d.get("title"), d.get("abstract")))
            else:
                nd = int(cit.nodes([int(w)])[0])
                out.append((cit.title(nd) if nd >= 0 else "", None))
        return out

    def publications(ws) -> np.ndarray:
        det = _details(ctx, ws)
        ok = []
        for w in ws:
            d = det.get(int(w))
            if d is not None:
                ok.append(passes(d, {}))
            else:
                nd = int(cit.nodes([int(w)])[0])
                ok.append(nd >= 0 and int(cit.kind[nd]) not in bad_types and len(cit.title(nd)) >= 10)
        return np.array(ok, bool)

    def cited_by(ws, k=CITED_K):
        nodes = cit.nodes(np.unique(np.asarray(ws, np.int64)))
        nodes = nodes[nodes >= 0]
        allrefs = np.concatenate([cit.refs(int(x)) for x in nodes]) if len(nodes) else np.zeros(0, np.uint32)
        u, cnt = np.unique(allrefs, return_counts=True)
        while (cnt >= k).sum() > EXPANSION_MAX:
            k += 1
        cn = u[cnt >= k]
        cn = cn[~np.isin(np.asarray(cit.kind[cn]), bad_types)]
        return np.array([cit.work(int(x)) for x in cn], np.int64), k

    def lr_hard(ws_y: dict[int, float]):
        """First classifier (teacher's answers only): a search direction and strata, not a judge."""
        ws = np.array(list(ws_y), np.int64)
        X, h = _vectors(ctx, ws)
        y = np.array([ws_y[int(w)] >= 0.5 for w in ws])
        return LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(X[h], y[h])

    def lr_soft(ws_p: dict[int, float]):
        """Vector student: learns probabilities (each paper counts as yes with weight p and no with weight 1 - p)."""
        ws = np.array(list(ws_p), np.int64)
        X, h = _vectors(ctx, ws)
        X, p = X[h], np.array([ws_p[int(w)] for w in ws])[h]
        return LogisticRegression(C=2.0, max_iter=4000).fit(np.concatenate([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))],
                                                             sample_weight=np.r_[p, 1 - p])

    def proba(model, ws) -> tuple[np.ndarray, np.ndarray]:
        out = np.full(len(ws), np.nan)
        X, has = _vectors(ctx, ws)
        if has.any():
            out[has] = model.predict_proba(X[has])[:, 1]
        return out, has

    # 1 the teacher, first round: papers spread from the nearest to the farthest of the 300k closest to the topic, and a
    #   quarter from the works the nearest papers cite (many have only a title: the decider must learn those too)
    t = time.time()
    job.stage(0, f"gathering the {POOL:,} papers closest to the topic")
    qv = eng.encoder.encode(f"{q}: {definition}" if definition else q)[0]     # "LLM" alone is also a law degree
    _, ca = eng.index.search(qv.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_a = ca[0][ca[0] >= 0]
    VA = unit(eng.vectors.get(rows_a))
    wa = np.asarray(eng.work_ids[rows_a], dtype=np.int64)
    o = np.argsort(-(VA @ qv), kind="stable")
    rows_a, VA, wa = rows_a[o], VA[o], wa[o]
    okA = ctx.quality.junk(wa, VA) < 0.5
    idxA = np.flatnonzero(okA)
    n = len(idxA)
    n_cited = TEACHER_R1 // 4
    edges = sorted({0, *[int(n * f) for f in (.002, .004, .007, .01, .017, .025, .035, .05, .07, .1, .14, .2, .27, .4,
                                                .55, .75)], n})
    per = math.ceil((TEACHER_R1 - n_cited) * 1.25 / (len(edges) - 1))
    sample, bucket = [], []
    for bi, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if hi > lo:
            pick = rng.choice(np.arange(lo, hi), size=min(per, hi - lo), replace=False)
            sample += idxA[pick].tolist()
            bucket += [bi] * len(pick)
    sw = wa[sample]
    ok = publications(sw)
    keep = rng.choice(np.flatnonzero(ok), size=min(TEACHER_R1 - n_cited, int(ok.sum())), replace=False)
    sw_pool, bucket = sw[np.sort(keep)], np.array(bucket)[np.sort(keep)]
    pre, _ = cited_by(wa[idxA[:5000]])
    pre = np.setdiff1d(pre, wa)
    _, hv = _vectors(ctx, pre)
    title_only, with_vec = pre[~hv], pre[hv]
    pick = list(rng.choice(title_only, size=min(len(title_only), n_cited * 2), replace=False)) + \
        list(rng.choice(with_vec, size=min(len(with_vec), n_cited), replace=False))
    pick = np.array(pick, np.int64)
    pick = pick[publications(pick)]
    n_t = min(len(pick), n_cited)
    title_set = set(title_only.tolist())
    t_only = [w for w in pick if w in title_set][:round(n_t * 0.7)]
    t_only_set = set(t_only)
    sw_cit = np.array(t_only + [w for w in pick if w not in t_only_set][:n_t - len(t_only)], np.int64)
    sw = np.concatenate([sw_pool, sw_cit])
    job.stage(0, f"the teacher (Kimi K3) judges {len(sw)} papers")
    teacher = J.Teacher(TEACHER_USD)
    texts_sw = texts_of(sw)
    t_label = time.time()
    lab = teacher.label(texts_sw, q, definition)
    secs = [round(time.time() - t_label, 1)]
    spent = teacher.usd
    _spend(job.dir.parent, spent, job.st.get("lane", "free"))
    labels = {int(w): float(x) for w, x in zip(sw, lab) if not math.isnan(x)}
    on_pool = np.nan_to_num(lab[:len(sw_pool)]) >= 0.5
    last = bucket == bucket.max()
    tail = float(on_pool[last].mean()) if last.any() else 0.0
    if int(on_pool.sum()) < MIN_ON_TOPIC:
        raise FinalError(f"Only {int(on_pool.sum())} of {len(sw_pool)} papers judged near “{q}” are about it: too few to "
                         "find the field. Phrase the topic as papers about it would, or give a definition.")
    if tail > TOO_BROAD_TAIL:
        raise FinalError(f"“{q}” is broader than a whole-field map can hold today: {round(100 * tail)}% of the papers at "
                         f"the edge of the {POOL:,} closest are still about it. Map a narrower topic, or use the "
                         "1,000-paper map.")
    job.done_stage(0, t)

    # 2 candidates: a classifier on meaning searches a second pool along its direction; the exact phrase; what the
    #   likely field cites. Then the teacher's second round (active learning): papers this classifier is unsure about,
    #   where the field's neighbours are (measured: +0.07 F1 for the decider over spreading the same answers)
    t = time.time()
    job.stage(1, "searching along a first classifier; following citations")
    lr0 = lr_hard(labels)
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
    C, k = cited_by(np.concatenate([U_w[p0 >= 0.5], H]))
    C = np.setdiff1d(C, np.concatenate([U_w, H]))
    pH0, hH = proba(lr0, H)
    pC0, hC = proba(lr0, C)
    report["candidates"] = {"pool": int(len(U_w)), "likely_by_first_classifier": int((p0 >= 0.5).sum()),
                            "phrase_outside_pool": int(len(H)), "cited_at_least": k, "cited_candidates": int(len(C)),
                            "cited_title_only": int((~hC).sum())}
    vec_w = np.concatenate([U_w, H[hH], C[hC]])
    vec_p = np.concatenate([p0, pH0[hH], pC0[hC]])
    band = vec_w[(vec_p > ACTIVE_BAND[0]) & (vec_p < ACTIVE_BAND[1])]
    band = band[~np.isin(band, sw)]
    cit_t = C[~hC]
    cit_t = cit_t[~np.isin(cit_t, sw)]
    n_band = round(TEACHER_R2 * 0.75)
    r2 = np.concatenate([rng.choice(band, size=min(len(band), n_band * 2), replace=False),
                         rng.choice(cit_t, size=min(len(cit_t), (TEACHER_R2 - n_band) * 2), replace=False)])
    okr = publications(r2)
    nb = min(len(band), n_band * 2)
    r2 = np.concatenate([r2[:nb][okr[:nb]][:n_band], r2[nb:][okr[nb:]][:TEACHER_R2 - n_band]])
    job.stage(1, f"the teacher judges {len(r2)} papers the first classifier is unsure about")
    texts_r2 = texts_of(r2)
    t_label = time.time()
    lab2 = teacher.label(texts_r2, q, definition)
    secs.append(round(time.time() - t_label, 1))
    _spend(job.dir.parent, teacher.usd - spent, job.st.get("lane", "free"))
    labels.update({int(w): float(x) for w, x in zip(r2, lab2) if not math.isnan(x)})
    report["teacher_labels"] = {"papers": len(labels), "on_topic": int(sum(v >= 0.5 for v in labels.values())),
                                "first_round": int(len(sw)), "cited": int(len(sw_cit)),
                                "title_only": int(sum(1 for w in sw_cit if w in title_set)) + int(len(r2) - min(len(r2), n_band)),
                                "active": int(len(r2)), "active_on_topic": int(np.nansum(lab2 >= 0.5)),
                                "failed": teacher.failed, "usd": round(teacher.usd, 3),
                                "farthest_on_topic": round(tail, 3), "seconds": secs,
                                "slowest_answer_s": teacher.slowest}
    job.done_stage(1, t)

    # 3 the decider learns from all the teacher's answers
    t = time.time()
    job.stage(2, "teaching the 0.8B decider")
    tw = list(labels)
    J.train_decider(list(zip(texts_of(tw), [labels[w] for w in tw])), q, definition, work / "decider", say)
    job.done_stage(2, t)

    # 4 the decider judges a balanced set for the students: likely papers, unsure ones, a spread, and cited works
    t = time.time()
    job.stage(3, f"the decider judges {DECIDER_N:,} papers")
    taken: set[int] = set(labels)

    def take(cands, m):
        cands = [int(w) for w in cands if int(w) not in taken]
        out = [cands[i] for i in rng.choice(len(cands), size=min(m, len(cands)), replace=False)] if cands else []
        taken.update(out)
        return out
    dws = take(U_w[p0 >= 0.5], int(DECIDER_N * .30)) + take(U_w[(p0 > 0.1) & (p0 < 0.9)], int(DECIDER_N * .25)) \
        + take(U_w, int(DECIDER_N * .15)) + take(C[~hC], int(DECIDER_N * .20)) + take(C[hC], int(DECIDER_N * .10))
    dp = J.decider_scores(work / "decider", texts_of(dws), say)
    decided = dict(zip(dws, dp.tolist()))
    report["decider_labels"] = {"papers": len(decided), "expected_on_topic": round(float(dp.sum()))}
    job.done_stage(3, t)

    # 5 students learn the decider's probabilities (not the teacher's answers: those stay unseen, to fit the
    #   combiner). The vector student judges every paper with a vector; Ettin judges papers with a title only, those
    #   the vector student is unsure about, and the teacher's papers
    t = time.time()
    job.stage(4, "training the students")
    univ = np.concatenate([U_w, H, C])
    lr1 = lr_soft(decided)
    pv = np.full(len(univ), np.nan)
    pv[:len(U_w)] = lr1.predict_proba(U_V)[:, 1]
    pv[len(U_w):len(U_w) + len(H)] = proba(lr1, H)[0]
    pv[len(U_w) + len(H):] = proba(lr1, C)[0]
    del U_V
    is_teacher = np.isin(univ, np.array(list(labels), np.int64))
    e_mask = np.isnan(pv) | ((pv > BAND[0]) & (pv < BAND[1])) | is_teacher
    E = univ[e_mask]
    say(f"the text student judges {len(E):,} papers (title only, unsure, or judged by the teacher)")
    dw = list(decided)
    pt = np.full(len(univ), np.nan)
    pt[e_mask] = J.student_scores(list(zip(texts_of(dw), [decided[w] for w in dw])), texts_of(E), work / "student", say)
    both = ~np.isnan(pv) & ~np.isnan(pt)
    stud = np.where(both, _sigmoid((_logit(np.nan_to_num(pv, nan=.5)) + _logit(np.nan_to_num(pt, nan=.5))) / 2),
                    np.where(np.isnan(pv), pt, pv))
    stud = np.nan_to_num(stud, nan=0.0)
    report["students"] = {"judged_by_text_student": int(len(E)), "expected_on_topic": round(float(stud.sum()))}
    job.done_stage(4, t)

    # 6 judging. Citations: works the likely field cites a lot, and papers whose references are mostly the field's,
    #   are on topic more often than their text alone says (the field's classics in older words: "woodlots as
    #   islands", "forest remnants"). A small combiner (student score + citation signals) is fitted on the teacher's
    #   answers, its baseline shifted so it expects as many papers on topic as the students; the teacher's and decider's
    #   own answers come first; the cut-off maximises the expected F1 over all candidates
    t = time.time()
    job.stage(5, "judging every candidate")
    pos = {w: i for i, w in enumerate(univ.tolist())}
    fixed = np.zeros(len(univ), bool)
    s1 = stud.copy()
    for w, v in decided.items():
        s1[pos[w]] = v
        fixed[pos[w]] = True
    feats = _citation_features(cit, univ, s1, _logit(stud))
    tix = np.array([pos[w] for w in labels if w in pos])
    ty = np.array([labels[int(univ[i])] >= 0.5 for i in tix])
    comb = LogisticRegression(C=1.0, max_iter=2000).fit(feats[tix], ty)
    z = feats @ comb.coef_[0] + comb.intercept_[0]
    free = ~fixed & ~is_teacher
    target = float(stud[free].sum())
    shift = brentq(lambda s: _sigmoid(z[free] + s).sum() - target, -20, 20)
    final = np.where(fixed, s1, _sigmoid(z + shift))
    for w, v in labels.items():
        if w in pos:
            final[pos[w]] = v
    cut, ef1 = _plugin_cut(final)
    before_check = final.copy()
    # Jev checks what readers will see: well-cited papers near the cut-off (measured on habitat fragmentation:
    # precision 0.75 -> 0.82, recall 0.73 -> 0.77; among papers cited 50+ times 0.70 / 0.78 -> 0.86 / 0.86). Its
    # answers decide those papers; nothing learns from them
    cites = np.zeros(len(univ), np.int64)
    nd_u = cit.nodes(univ)
    cites[nd_u >= 0] = np.asarray(cit.cites)[nd_u[nd_u >= 0]]
    chk = np.flatnonzero((final >= CHECK_BAND[0]) & (final < CHECK_BAND[1]) & (cites >= CHECK_CITES) & ~is_teacher)
    chk = chk[np.argsort(-cites[chk], kind="stable")]
    job.stage(5, f"Jev checks {len(chk):,} well-cited papers near the cut-off")
    jev = J.Jev(min(CHECK_USD, max(0.0, BUDGET_USD - FieldStore(job.dir.parent).spent_today())))
    t_jev = time.time()
    jl = jev.label(pairs_of(univ[chk]), q, definition) if len(chk) else np.zeros(0)
    _spend(job.dir.parent, jev.usd, job.st.get("lane", "free"))
    checked = {}
    for i, v in zip(chk, jl):
        if v == v:
            checked[int(univ[i])] = float(v)
            final[i] = float(v >= 0.5)
    report["check"] = {"by": f"Jev ({J.JEV_MODEL})", "band": list(CHECK_BAND), "cited_at_least": CHECK_CITES,
                       "papers": len(chk), "answered": len(checked), "failed": jev.failed,
                       "in": int(sum(v >= 0.5 for v in checked.values())), "usd": round(jev.usd, 3),
                       "seconds": round(time.time() - t_jev, 1)}
    np.savez_compressed(work / "scores.npz", ids=univ, final=final, before_check=before_check, cut=cut, student=stud, route=np.array(
        [0] * len(U_w) + [1] * len(H) + [2] * len(C), np.int8), features=feats.astype(np.float32))
    names = ("student", "cited by the field", "share of its citations", "share of its references", "no references")
    report["judging"] = {"cut": cut, "expected_f1": ef1, "expected_on_topic": round(float(final.sum())),
                         "combiner": {n_: round(float(c_), 3) for n_, c_ in zip(names, comb.coef_[0])},
                         "baseline_shift": round(float(shift), 3)}
    srcs = (["meaning"] * len(U_w)) + (["phrase"] * len(H)) + (["cited"] * len(C))
    yes = np.flatnonzero(final >= cut)
    ok = publications(univ[yes])
    field = {int(univ[i]): {"via": srcs[i], "p": round(float(final[i]), 3)} for i, o_ in zip(yes, ok) if o_}
    if len(field) > TOO_BROAD_PAPERS:
        raise FinalError(f"“{q}” has about {len(field):,} papers: too broad for a whole-field map today. Map a narrower topic.")
    report["papers"] = len(field)
    report["by_route"] = {v: sum(1 for f in field.values() if f["via"] == v) for v in ("meaning", "phrase", "cited")}
    report["teacher_usd"] = round(teacher.usd, 3)
    report["usd"] = round(teacher.usd + jev.usd, 3)
    job.done_stage(5, t)
    return {"version": SELECTION_VERSION, "made": _now(), "release": ctx.release["id"], "topic": q,
            "definition": definition, "keywords": keywords, "report": report,
            "papers": {str(w): f for w, f in field.items()},
            "labels": {"teacher": {str(w): v for w, v in labels.items()},
                       "decider": {str(w): round(v, 3) for w, v in decided.items()},
                       "jev": {str(w): round(v, 3) for w, v in checked.items()}}}


def _citation_features(cit, univ: np.ndarray, score: np.ndarray, student_logit: np.ndarray) -> np.ndarray:
    """Per candidate: its student logit; log(1 + candidates citing it, each counted by its score); the share of all its
    citations that come from them; the score-weighted share of its references that are candidates; no references."""
    nodes = cit.nodes(univ)
    order = np.argsort(nodes)
    sn = nodes[order]

    def at(ns):
        i = np.minimum(np.searchsorted(sn, ns), len(sn) - 1)
        return np.where(sn[i] == ns, order[i], -1)
    offs = np.asarray(cit.fwd_off)
    has = np.flatnonzero(nodes >= 0)
    lens = (offs[nodes[has] + 1] - offs[nodes[has]]).astype(np.int64)
    refs = np.concatenate([cit.refs(int(x)) for x in nodes[has]]).astype(np.int64) if len(has) else np.zeros(0, np.int64)
    owner = np.repeat(has, lens)
    tgt = at(refs)
    hit = tgt >= 0
    fc = np.bincount(tgt[hit], weights=score[owner[hit]], minlength=len(univ))       # in: likely field citing it
    tot = np.zeros(len(univ))
    tot[has] = np.asarray(cit.cites)[nodes[has]]
    out_sum = np.bincount(owner[hit], weights=score[tgt[hit]], minlength=len(univ))  # out: its references in the field
    nref = np.zeros(len(univ))
    nref[has] = lens
    return np.column_stack([student_logit, np.log1p(fc), fc / np.maximum(tot, 1), out_sum / np.maximum(nref, 1),
                            (nref == 0).astype(float)])


class FinalError(Exception):
    """The topic cannot be mapped as a whole field (too broad, too unclear): asking again will not help."""


def build_maps(ctx: Ctx, job: Job, sel: dict, version: str) -> dict:
    from srl_search import topics as T
    from srl_search.topic_service import embed_splits, finish
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
    background = ctx.keywords.phrase_counts if ctx.keywords is not None else None
    raw = T.build(seed, cit, q, T.Options(refs_top=30, workers=LEIDEN_WORKERS), background=background)
    out = finish(ctx, cit, raw, seed, ctx.release["id"])
    via = sel["papers"]
    out["papers"]["via"] = [via.get(pid[1:], {}).get("via", "") for pid in out["papers"]["id"]]
    out.update({"question": q, "mode": "field", "filters": {}, "asked_papers": len(seed.wids),
                "field": {**sel["report"], "made": sel["made"], "release": sel["release"]}})
    dirm = job.dir / "maps" / version
    job.done_stage(6, t)

    t = time.time()
    big = [g for g in out["groups"] if g["size"] >= SPLIT_MIN]
    splits = {}
    by_w = {int(w): i for i, w in enumerate(seed.wids)}
    for j, g in enumerate(big):
        job.stage(7, f"splitting group {g['id']} ({g['size']:,} papers), {j + 1} of {len(big)}")
        members = [by_w[int(pid[1:])] for pid, gg in zip(out["papers"]["id"], out["papers"]["group"]) if gg == g["id"]]
        sub_seed = seed.subset(np.array(members))
        sub = T.build(sub_seed, cit, q, T.Options(refs_top=30, min_size=max(8, len(members) // 25), min_groups=2,
                                                  max_groups=8, workers=LEIDEN_WORKERS), background=background)
        sub = finish(ctx, cit, sub, sub_seed, ctx.release["id"])
        sub["papers"]["via"] = [via.get(pid[1:], {}).get("via", "") for pid in sub["papers"]["id"]]
        sub.update({"question": q, "mode": "field", "filters": {}, "asked_papers": len(members),
                    "parent": {"group": g["id"], "name": g["name"], "size": g["size"]}})
        _write(dirm / f"group-{g['id']}.json", sub)
        splits[g["id"]] = sub
    embed_splits(out, splits)
    _write(dirm / "main.json", out)                    # last: a version with its main map is complete
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
                                   "usd": sel["report"].get("usd", sel["report"].get("teacher_usd", sel["report"].get("jev", {}).get("usd")))}})
    except FinalError as e:
        job.st.update({"status": "failed", "final": True, "error": str(e), "finished": _now()})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.st.update({"status": "failed", "final": False, "error": f"the job failed ({type(e).__name__}); it can be asked again",
                       "finished": _now()})
    _write(job.dir / "job.json", job.st)


def worker(store: Path = ROOT) -> None:
    """Run waiting maps one at a time in the plan's order (FieldStore.plan: paid first, then the free line while
    today's free budget lasts) until none is waiting. When only free maps wait and today's budget is spent, it
    frees its memory and waits for the budget (it stays running, so the path unit does not restart it in a loop).
    A paid map that fails for a reason other than the topic itself is tried once more."""
    import gc
    store.mkdir(parents=True, exist_ok=True)
    lock = open(store / "worker.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another worker is running", flush=True)
        return
    fs = FieldStore(store)
    ctx = None
    while True:
        for k in fs.queued():                          # a queue file whose job is no longer waiting: stale
            if (_read(store / k / "job.json") or {}).get("status") not in ("queued", "running"):
                (store / "queue" / f"{k}.job").unlink(missing_ok=True)
        pl = fs.plan()
        if not pl["waiting"]:
            return
        nxt = pl["waiting"][0]
        if nxt["start"] - pl["now"] > 120:             # only free maps wait, for tomorrow's budget
            if ctx is not None:
                ctx = None
                gc.collect()
            print(f"free budget spent for today; next map in {round((nxt['start'] - pl['now']) / 60)} min", flush=True)
            time.sleep(30)                             # a map paid for meanwhile starts at once
            continue
        key, lane = nxt["key"], nxt["lane"]
        st = _read(store / key / "job.json") or {}
        st["lane"] = lane
        _write(store / key / "job.json", st)
        ctx = ctx or Ctx()
        print(f"job {key} ({lane}): {st.get('request')}", flush=True)
        try:
            run_job(ctx, store, key)
        finally:
            st = _read(store / key / "job.json") or {}
            if lane == "paid" and st.get("status") == "failed" and not st.get("final") and not st.get("retried"):
                print(f"job {key}: paid, failed ({st.get('error')}); once more", flush=True)
                st.update({"status": "queued", "retried": True, "error": None})
                _write(store / key / "job.json", st)
            else:
                (store / "queue" / f"{key}.job").unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    if argv[:1] == ["worker"]:
        return worker()
    if argv[:1] == ["run"] and len(argv) >= 2:      # run TOPIC [DEFINITION]: one topic, now, in this process
        store = Path(os.environ.get("SRL_FIELDS", str(ROOT)))
        definition = argv[2] if len(argv) >= 3 else None
        key = field_key(argv[1], definition)
        _write(store / key / "job.json", {"status": "queued", "kind": "full", "key": key, "created": _now(),
                                         "request": {"q": argv[1], "definition": definition, "keywords": None}})
        run_job(Ctx(), store, key)
        print(json.dumps(_read(store / key / "job.json"), indent=1))
        return
    print(__doc__)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main(sys.argv[1:])
