"""One whole-field map per topic (srl_search/same.py, FieldStore.get): another wording of a topic made or waiting
opens that map; a different topic is queued. The AI is replaced by a stub: its accuracy was measured on 194
labelled pairs (the module's docstring)."""
import json

import numpy as np
import pytest
from fastapi import HTTPException

from srl_search import field as F
from srl_search import same as S

HCI_DEF = "The study of how people interact with computers and the design of interactive computing systems."


def enc(t: str) -> np.ndarray:
    """A stand-in encoder: topics sharing a first word are close."""
    v = np.zeros(8, np.float32)
    v[hash(S.norm(t).split()[0]) % 8] = 1
    v[hash(t) % 8] += 0.3
    return v / np.linalg.norm(v)


class AI:
    def __init__(self, same: set[tuple[str, str]] | None = None, down: bool = False):
        self.same, self.down, self.calls = same or set(), down, 0

    def __call__(self, request: str, maps: list[str]) -> list[str]:
        self.calls += 1
        if self.down:
            raise RuntimeError("providers down")
        return ["same" if (request.split(":")[0], m.split(":")[0]) in self.same else "narrower" for m in maps]


def made(root, topic, definition=None):
    """A finished map of this topic."""
    key = F.field_key(topic, definition)
    d = root / key
    (d / "maps" / "v1").mkdir(parents=True)
    (d / "maps" / "v1" / "main.json").write_text(json.dumps({"question": topic, "groups": [], "field": {"papers": 9}}))
    (d / "field.json").write_text("{}")
    (d / "job.json").write_text(json.dumps({"status": "done", "kind": "full", "key": key,
                                            "request": {"q": topic, "definition": definition, "keywords": None}}))
    return key


def store(tmp_path, ai):
    return F.FieldStore(tmp_path, same=S.Matcher(tmp_path, enc, ai))


def test_norm_sets_aside_case_punctuation_and_plurals():
    assert S.norm("Human–Computer Interactions") == S.norm("human computer interaction")
    assert S.norm("Coral reefs") == S.norm("coral reef") and S.norm("C. elegans") == "c elegan"
    assert S.norm("virus analysis") == "virus analysis"


def test_the_same_words_without_a_definition_open_the_map_made_with_one(tmp_path):
    made(tmp_path, "human-computer interaction", HCI_DEF)
    ai = AI()
    out = store(tmp_path, ai).get("human-computer interaction", None, None, "v1", None)
    assert out["field"]["same_as"] == {"topic": "human-computer interaction", "definition": HCI_DEF,
                                       "asked": "human-computer interaction", "asked_definition": None}
    assert ai.calls == 0 and not list((tmp_path / "queue").glob("*.job"))   # no AI needed, nothing queued


def test_another_wording_opens_the_map_and_is_asked_once(tmp_path):
    made(tmp_path, "Large Language Model Post Training")
    ai = AI({("LLM post-training", "Large Language Model Post Training")})
    st = store(tmp_path, ai)
    for _ in range(2):
        out = st.get("LLM post-training", None, None, "v1", None)
        assert out["field"]["same_as"]["topic"] == "Large Language Model Post Training"
    assert ai.calls == 1 and not st.queued()


def test_a_topic_already_in_line_shows_that_job(tmp_path):
    ai = AI({("HCI", "human-computer interaction")})
    st = store(tmp_path, ai)
    with pytest.raises(F.Pending):
        st.get("human-computer interaction", HCI_DEF, None, "v1", None)
    with pytest.raises(F.Pending) as e:
        st.get("HCI", None, None, "v1", None)
    assert e.value.status["key"] == F.field_key("human-computer interaction", HCI_DEF)
    assert e.value.status["same_as"]["topic"] == "human-computer interaction"
    assert len(st.queued()) == 1


def test_a_different_topic_is_queued(tmp_path):
    made(tmp_path, "human-computer interaction", HCI_DEF)
    st = store(tmp_path, AI())                         # the AI says "narrower"
    with pytest.raises(F.Pending) as e:
        st.get("human-chatbot interaction", None, None, "v1", None)
    assert e.value.status["key"] == F.field_key("human-chatbot interaction") and len(st.queued()) == 1


def test_the_same_words_with_a_definition_of_their_own_are_judged(tmp_path):
    made(tmp_path, "LLM", "Large language models.")
    ai = AI()                                          # "LLM: the law degree" is not the language models
    st = store(tmp_path, ai)
    with pytest.raises(F.Pending):
        st.get("LLM", "The Master of Laws degree.", None, "v1", None)
    assert ai.calls == 1 and len(st.queued()) == 1


def test_without_the_ai_a_request_close_to_a_map_waits_and_a_far_one_is_queued(tmp_path):
    made(tmp_path, "coral reefs")
    st = store(tmp_path, AI(down=True))
    with pytest.raises(HTTPException) as e:
        st.get("coral bleaching", None, None, "v1", None)        # close (same first word): might be the same
    assert e.value.status_code == 503 and not st.queued()
    far = next(t for t in ("zebrafish", "quasars", "tax policy", "glaciers") if S.norm(t)[0] != "c"
               and float(enc(t) @ enc("coral reefs")) < S.UNSURE_COS)
    with pytest.raises(F.Pending):
        st.get(far, None, None, "v1", None)
    assert len(st.queued()) == 1


def test_a_kept_match_whose_map_is_gone_is_asked_again(tmp_path):
    key = made(tmp_path, "sleep research")
    ai = AI({("sleep science", "sleep research")})
    st = store(tmp_path, ai)
    assert st.get("sleep science", None, None, "v1", None)["field"]["same_as"]["topic"] == "sleep research"
    import shutil
    shutil.rmtree(tmp_path / key)                      # removed
    with pytest.raises(F.Pending):
        st.get("sleep science", None, None, "v1", None)
    assert ai.calls == 1 and len(st.queued()) == 1     # nothing left to compare with: queued without asking


def test_without_a_matcher_nothing_changes(tmp_path):
    made(tmp_path, "human-computer interaction", HCI_DEF)
    with pytest.raises(F.Pending):
        F.FieldStore(tmp_path).get("human-computer interaction", None, None, "v1", None)
