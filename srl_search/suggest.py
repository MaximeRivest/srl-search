"""AI help for people asking for a whole-field map: five well-phrased versions of the topic they typed, and names
and short descriptions for the research groups of a whole-field map.

People describe topics loosely ("ai papers", "LLM", "bees dying"); a map is only as good as its topic, and a
definition settles ambiguous words ("LLM" is also a law degree). Kimi K3 (the providers in srl_search/judges.py)
rewrites the request; the person chooses one, or keeps their own words. Group names: the map's own phrases are
precise but terse ("cooling · green spaces · parks"); a sentence about what the community studies helps a reader.

Cost: ~$0.003 a rewrite, ~$0.003 a group (~$0.15-0.30 a whole-field map with its subgroups); capped a day
(SRL_SUGGEST_USD, SRL_LABELS_USD) in ROOT/spend_lanes.json beside the maps' spend. Rewrites are cached.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from srl_search import judges as J

ROOT = Path(os.environ.get("SRL_FIELDS", "/mnt/fast/scholarsreadinglist/search/fields"))
SUGGEST_USD = float(os.environ.get("SRL_SUGGEST_USD", "3"))      # topic rewrites a day (~1,000)
LABELS_USD = float(os.environ.get("SRL_LABELS_USD", "5"))        # group names a day (~20 whole-field maps)
TOPIC_LABELS_USD = float(os.environ.get("SRL_TOPIC_LABELS_USD", "5"))   # ... of 1,000-paper maps (~50 a day)
_LOCK = threading.Lock()


class Unavailable(Exception):
    """No rewrite now (no key, today's cap reached, providers down): the person submits their own wording."""


def _spent(lane: str) -> float:
    try:
        return float((json.loads((ROOT / "spend_lanes.json").read_text()).get(date.today().isoformat()) or {}).get(lane, 0.0))
    except (OSError, ValueError):
        return 0.0


def _spend(lane: str, usd: float) -> None:
    with open(ROOT / "spend.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        path = ROOT / "spend_lanes.json"
        try:
            led = json.loads(path.read_text())
        except (OSError, ValueError):
            led = {}
        day = led.setdefault(date.today().isoformat(), {})
        day[lane] = round(day.get(lane, 0.0) + usd, 5)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(led))
        os.replace(tmp, path)


def _kimi(system: str, user: str, max_tokens: int = 900) -> tuple[str, float]:
    """One Kimi K3 answer through the first provider that answers; (text, dollars)."""
    providers = [(n, *v) for n, v in J.KIMI.items() if os.environ.get(v[4])]
    if not providers:
        raise Unavailable("no Kimi K3 key on this server")
    last = None
    for name, url, model, extra, price, key in providers:
        body = json.dumps({"model": model, "temperature": 0.3, "max_tokens": max_tokens, **extra, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {os.environ[key]}",
                                                              "Content-Type": "application/json",
                                                              "User-Agent": "scholarsreadinglist-fields/1"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                out = json.load(r)
            u = out.get("usage") or {}
            usd = (u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[1]) / 1e6
            return out["choices"][0]["message"].get("content") or "", usd
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
            last = e
    raise Unavailable(f"Kimi K3 did not answer ({type(last).__name__})")


def _json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0] or [0])
    return json.loads(text[start:])


# ------------------------------------------------------------------------------------------------ topic rewrites
REWRITE = """You help people ask for a map of every research paper on a topic. People describe topics loosely or
ambiguously. Write 5 ways to name the topic they mean, each as researchers would type it into a search box (a short
noun phrase, 2-8 words, abbreviations where researchers use them), with a one-sentence definition that sets its scope
and settles any ambiguous word (e.g. "LLM" is also a law degree; "transformers" are also electrical devices).
1: the closest to what they typed, cleaned up. Then, as fits: a narrower and a broader reading, and other meanings
if the words are ambiguous, or the established name of the field if they used everyday words. Each must be a real
research topic with a literature. Never answer the question itself or change the subject.
Answer with JSON only: [{"topic": "...", "definition": "...", "note": "2-6 words: how it differs, e.g. 'closest to
what you typed', 'narrower: only ...', 'if you meant ...'"}]"""


def rewrites(q: str, definition: str | None = None) -> dict:
    """Five versions of the topic someone typed: {"suggestions": [{topic, definition, note}], "cached": bool}."""
    q = " ".join(q.split())[:300]
    definition = " ".join((definition or "").split())[:600] or None
    key = hashlib.sha256(json.dumps([q.lower(), (definition or "").lower()]).encode()).hexdigest()[:24]
    cache = ROOT / "suggest-cache" / f"{key}.json"
    if cache.exists():
        return {**json.loads(cache.read_text()), "cached": True}
    with _LOCK:
        if _spent("suggest") >= SUGGEST_USD:
            raise Unavailable("today's topic suggestions are used up; your own wording works")
    user = f"They typed: {q}" + (f"\nThey described it as: {definition}" if definition else "")
    text, usd = _kimi(REWRITE, user)
    _spend("suggest", usd)
    out = []
    for it in _json(text) if text else []:
        t = " ".join(str(it.get("topic") or "").split())
        d = " ".join(str(it.get("definition") or "").split())
        if 3 <= len(t) <= 120 and t.lower() not in {o["topic"].lower() for o in out}:
            out.append({"topic": t, "definition": d[:400] or None, "note": " ".join(str(it.get("note") or "").split())[:60]})
    if not out:
        raise Unavailable("no usable suggestions this time; your own wording works")
    result = {"query": q, "definition": definition, "suggestions": out[:5], "by": "Kimi K3"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return {**result, "cached": False}


# ------------------------------------------------------------------------------------------------ group names
NAME = """You name the research communities of a map of a research field. For one group of papers you get the field,
the phrases most over-represented in the group, and some of its papers' titles (its most central papers first).
Answer with JSON only: {"name": "3-7 words, what this community studies, plain and specific", "about": "1-2
sentences (at most 45 words) for a curious reader: what these researchers study and how, in plain words"}.
Describe only what the titles and phrases show; do not invent findings, authors or dates."""


def _group_prompt(field: str, g: dict, parent: str | None) -> str:
    titles = []
    for lens in ("start_here", "core", "distinctive", "rising"):
        for p in (g.get("lenses", {}).get(lens) or [])[:5]:
            t = (p.get("title") or "").strip()
            if t and t not in titles:
                titles.append(t)
    typ = (g.get("typical") or {}).get("title")
    if typ and typ not in titles:
        titles.insert(0, typ)
    return (f"Field: {field}\n" + (f"This group is part of: {parent}\n" if parent else "") +
            f"Size: {g.get('size')} papers\nOver-represented phrases: {g.get('name')}\nTitles:\n- " +
            "\n- ".join(titles[:14]))


def label_groups(field: str, groups: list[dict], parent: str | None = None, lane: str = "labels",
                 cap: float | None = None) -> float:
    """Give each group g["label"] = {"name", "about", "by"} in place (missing on a failure: the page keeps the
    phrases). Returns the dollars spent. Stops when today's cap for this lane is reached. Labels are cached by
    their prompt (ROOT/label-cache): the same group in tomorrow's rebuilt map, or a split map's groups already
    named inside its main map, cost nothing."""
    cap = LABELS_USD if cap is None else cap
    spent = [0.0]
    lock = threading.Lock()
    cache = ROOT / "label-cache"

    def one(g):
        if g.get("label"):
            return
        prompt = _group_prompt(field, g, parent)
        f = cache / (hashlib.sha256((NAME + prompt).encode()).hexdigest()[:32] + ".json")
        if f.exists():
            try:
                g["label"] = json.loads(f.read_text())
                return
            except ValueError:
                pass
        with lock:
            if _spent(lane) + spent[0] >= cap:
                return
        try:
            text, usd = _kimi(NAME, prompt, max_tokens=300)
            with lock:
                spent[0] += usd
            v = _json(text)
            name = " ".join(str(v.get("name") or "").split())[:80]
            about = " ".join(str(v.get("about") or "").split())[:400]
            if name:
                g["label"] = {"name": name, "about": about or None, "by": "Kimi K3"}
                cache.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
                tmp.write_text(json.dumps(g["label"]))
                os.replace(tmp, f)
        except (Unavailable, ValueError, AttributeError, TypeError, OSError):
            pass
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, groups))
    if spent[0]:
        _spend(lane, spent[0])
    return spent[0]


def label_map(question: str, m: dict, parent: str | None = None, lane: str = "labels", cap: float | None = None) -> float:
    """A map's groups, then the subgroups it carries (each named within its group)."""
    usd = label_groups(question, m.get("groups", []), parent, lane, cap)
    subs = [(g, s) for g in m.get("groups", []) for s in (g.get("subgroups") or [])]
    by_parent: dict[str, list] = {}
    for g, s in subs:
        by_parent.setdefault((g.get("label") or {}).get("name") or g.get("name") or "", []).append(s)
    for pname, ss in by_parent.items():
        usd += label_groups(question, ss, pname or None, lane, cap)
    return usd
