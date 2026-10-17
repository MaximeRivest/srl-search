"""One whole-field map per topic: a request for a topic already mapped, or already waiting in line, under other
words opens that map instead of making another (srl_search/field.py FieldStore.get asks before queueing).

A map's identity is its exact wording and definition (field.field_key), so "human-computer interaction" with no
definition, "HCI" or "LLM post-training" would each queue a second map of a topic made that morning. Measured on
194 labelled pairs (2026-10-17: the topic rewrites' "closest to what you typed" as the same topic; their narrower,
broader and other readings, and hand pairs from the maps made, as different), neither the topics' embedding
similarity nor the share of their 100 closest papers tells "the same" from "narrower, broader, a neighbour": same
pairs went down to cosine 0.35 and 0 shared papers ("exoplanets" / "extrasolar planets": 6), different ones up to
0.84 and 38. So:

1. the same words (case, punctuation, plurals aside) and no definition of its own: the map of those words;
2. otherwise the 6 maps whose topics are closest in meaning go to an AI (Kimi K3, the providers of suggest.py) that
   names each one's relation to the request: same, narrower, broader, overlapping, another meaning, different. Only
   "same" opens that map. On the pairs: 33 of 35 same found, 2 of 157 different called same (both borderline);
   ~$0.002 a request, capped a day (SRL_SAME_USD).

A decision is kept (ROOT/same/<key>.json), so the AI is asked once per wording; a kept match whose map has gone
(failed, removed) is asked again. When the AI can't answer, a request close in meaning to a map (cosine >= 0.5,
above all but 2 of the 35 same pairs) waits rather than risk a duplicate; one far from every map is new for sure.
Someone who does mean a different topic says so in a definition: a request with its own definition is judged
again, as its own wording.
"""
from __future__ import annotations
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np

SHORTLIST = 6
UNSURE_COS = 0.5          # with no AI answer, a request this close to a map's topic waits (see above)
_PLURAL = re.compile(r"(?<=[a-z]{3})(?<!s)(?<!u)(?<!i)s$")   # "reefs" -> "reef"; not "class", "virus", "analysis"


def norm(t: str) -> str:
    """The words of a topic, as compared: no case, accents, punctuation or plural "s" ("Human–Computer
    Interactions" = "human computer interaction", "C. elegans" = "c elegans")."""
    t = re.sub(r"[\W_]+", " ", t.replace("&", " and "))            # dashes, dots, quotes: spaces (any script)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return " ".join(_PLURAL.sub("", w) for w in re.sub(r"[^a-z0-9]+", " ", t).split())


def text(topic: str, definition: str | None) -> str:
    """A topic as the map maker reads it (field.select_field encodes the same string)."""
    return f"{topic}: {definition}" if definition else topic


PROMPT = """Someone asks for a map of every research paper on a topic. Maps of other topics already exist or are being made.
For each map below, say how the requested topic relates to the map's topic, as bodies of research:
- "same": the same body of research in other words: a synonym, an abbreviation, another spelling or the plural,
  the established name for what was said in everyday words, a rewording, or the same topic with no definition, a
  shorter one or another one that sets the same scope. A map of either would hold the same papers.
- "narrower": the request is a part of the map's topic (a subfield, one theory, one method, one population...).
- "broader": the request contains the map's topic and more.
- "overlapping": they share much research, but each has a sizeable part the other lacks.
- "other meaning": the same or similar words meaning something else.
- "different": a separate topic.
A definition settles which meaning and how wide; a topic without one takes its usual research meaning. "Same"
only when nothing sizeable is in one and not the other; if unsure, choose another answer.

Answer with JSON only: {"maps": [{"n": <number>, "relation": "<one of the above>"}]}"""


class Unsure(Exception):
    """No AI answer and the request is close to a map: it waits rather than risk making that map again."""


def ask_ai(request: str, maps: list[str]) -> list[str]:
    """The AI's relation of the request to each map, in order ("same", "narrower", ...; "" when it gave none)."""
    import os
    from srl_search import suggest
    cap = float(os.environ.get("SRL_SAME_USD", "2"))
    if suggest._spent("same") >= cap:
        raise suggest.Unavailable("today's checks are used up")
    user = "Asked for: " + request + "\n\nMaps:\n" + "\n".join(f"{i + 1}. {m}" for i, m in enumerate(maps))
    out, usd = suggest._kimi(PROMPT, user, max_tokens=60 + 25 * len(maps))
    suggest._spend("same", usd)
    rel = {int(m.get("n", 0)): str(m.get("relation") or "") for m in (suggest._json(out).get("maps") or [])}
    return [rel.get(i + 1, "") for i in range(len(maps))]


class Matcher:
    """Finds the map (made or waiting) a new request is the same topic as. encode: text -> unit vector (the search
    engine's query encoder); ask: the AI (ask_ai; tests pass their own)."""

    def __init__(self, root: Path, encode: Callable[[str], np.ndarray], ask: Callable = ask_ai):
        self.dir = Path(root) / "same"
        self.encode, self.ask = encode, ask
        self.lock = threading.Lock()                   # two wordings of one topic at once: the second sees the first
        self._vec: dict[str, np.ndarray] = {}

    def _v(self, t: str) -> np.ndarray:
        v = self._vec.get(t)
        if v is None:
            v = self._vec[t] = np.asarray(self.encode(t), np.float32).reshape(-1)
        return v

    def find(self, key: str, q: str, definition: str | None, maps: list[dict]) -> dict | None:
        """maps: [{key, topic, definition}] made or waiting (not this request's own). The one that is the same
        topic, or None. Raises Unsure when it can't tell and one is close."""
        if not maps:
            return None
        by_key = {m["key"]: m for m in maps}
        kept = self._read(key)
        if kept is not None and (kept.get("same") is None or kept["same"] in by_key) \
                and set(kept.get("asked", [])) >= {m["key"] for m in self._shortlist(q, definition, maps)}:
            return by_key.get(kept.get("same"))
        if not definition:                             # the same words: the map of those words
            words = [m for m in maps if norm(m["topic"]) == norm(q)]
            if len(words) == 1:
                return self._keep(key, words[0], [words[0]["key"]], "same words")
        short = self._shortlist(q, definition, maps)
        try:
            rel = self.ask(text(q, definition), [text(m["topic"], m.get("definition")) for m in short])
        except Exception as e:  # noqa: BLE001 (no key, cap reached, providers down, a reply that isn't JSON)
            best = max(float(self._v(text(q, definition)) @ self._v(text(m["topic"], m.get("definition"))))
                       for m in short)
            if best >= UNSURE_COS:
                raise Unsure(str(e)) from e
            return None                                # far from every map: new for sure (not kept: ask next time)
        same = next((m for m, r in zip(short, rel) if r.strip().lower() == "same"), None)
        return self._keep(key, same, [m["key"] for m in short], "AI", rel)

    def _shortlist(self, q: str, definition: str | None, maps: list[dict]) -> list[dict]:
        v = self._v(text(q, definition))
        sims = [float(v @ self._v(text(m["topic"], m.get("definition")))) for m in maps]
        return [maps[i] for i in np.argsort(sims)[::-1][:SHORTLIST]]

    def _keep(self, key: str, m: dict | None, asked: list[str], by: str, rel: list[str] | None = None) -> dict | None:
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {"same": m["key"] if m else None, "asked": asked, "by": by, **({"relations": rel} if rel else {})}
        (self.dir / f"{key}.json").write_text(json.dumps(rec))
        return m

    def _read(self, key: str) -> dict | None:
        try:
            return json.loads((self.dir / f"{key}.json").read_text())
        except (OSError, ValueError):
            return None
