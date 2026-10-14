"""Whole-field maps (srl_search/field.py): the store the API uses (queue, budget, progress, finished maps), and the
scale-safe pieces of topics.py they rely on. The pipeline itself needs the indexes and Jev: it is measured in
experiments/topicmap (validate_field.py), not here."""
import json
from datetime import date

import numpy as np
import pytest
from fastapi import HTTPException
from scipy import sparse

from srl_search import field as F
from srl_search import topics as T


def test_queues_a_job_and_reports_progress(tmp_path):
    st = F.FieldStore(tmp_path)
    with pytest.raises(F.Pending) as e:
        st.get("Habitat  fragmentation", None, None, "v1", None)
    assert e.value.status["status"] == "queued" and e.value.status["position"] == 1
    key = F.field_key("habitat fragmentation")
    assert (tmp_path / "queue" / f"{key}.job").exists()
    with pytest.raises(F.Pending):                      # asking again does not queue twice
        st.get("habitat fragmentation", None, None, "v1", None)
    assert len(st.queued()) == 1
    text = F.pending_text(e.value.status, "habitat fragmentation", "https://x")
    assert "being made" in text and "https://x/topics/habitat%20fragmentation" in text


def test_reading_never_starts_a_job(tmp_path):
    with pytest.raises(HTTPException) as e:
        F.FieldStore(tmp_path).get("x y", None, None, "v1", 2, start=False)
    assert e.value.status_code == 409 and not list((tmp_path / "queue").glob("*"))


def test_serves_finished_maps_and_says_why_a_group_has_no_split(tmp_path):
    key = F.field_key("x y")
    m = tmp_path / key / "maps" / "v1"
    m.mkdir(parents=True)
    (m / "main.json").write_text(json.dumps({"groups": [{"id": 1}, {"id": 2}]}))
    (m / "group-1.json").write_text(json.dumps({"parent": {"group": 1}}))
    st = F.FieldStore(tmp_path)
    assert st.get("x y", None, None, "v1", None) == {"groups": [{"id": 1}, {"id": 2}]}
    assert st.get("x y", None, None, "v1", 1)["parent"]["group"] == 1
    with pytest.raises(HTTPException) as e:
        st.get("x y", None, None, "v1", 2)
    assert "read it in full" in e.value.detail
    with pytest.raises(HTTPException):
        st.get("x y", None, None, "v1", 9)
    # the map code changed: the selected field is kept, only the maps are made again
    (tmp_path / key / "field.json").write_text("{}")
    with pytest.raises(F.Pending) as e:
        st.get("x y", None, None, "v2", None)
    assert e.value.status["kind"] == "map"


def test_budget_and_queue_limits(tmp_path, monkeypatch):
    st = F.FieldStore(tmp_path)
    (tmp_path / "spend.json").write_text(json.dumps({date.today().isoformat(): F.BUDGET_USD - 1}))
    with pytest.raises(HTTPException) as e:
        st.get("a b", None, None, "v1", None)
    assert e.value.status_code == 503 and "budget" in e.value.detail
    (tmp_path / "spend.json").write_text("{}")
    monkeypatch.setattr(F, "MAX_QUEUED", 1)
    with pytest.raises(F.Pending):
        st.get("a b", None, None, "v1", None)
    with pytest.raises(HTTPException) as e:
        st.get("c d", None, None, "v1", None)
    assert e.value.status_code == 503 and "waiting" in e.value.detail


def test_a_topic_that_cannot_be_mapped_says_why_once(tmp_path):
    key = F.field_key("everything")
    (tmp_path / key).mkdir(parents=True)
    (tmp_path / key / "job.json").write_text(json.dumps({"status": "failed", "final": True, "error": "too broad"}))
    with pytest.raises(HTTPException) as e:
        F.FieldStore(tmp_path).get("everything", None, None, "v1", None)
    assert e.value.status_code == 422 and e.value.detail == "too broad"


def test_definition_is_part_of_the_identity():
    assert F.field_key("bats") != F.field_key("bats", "the flying mammals")
    q = F.jev_question("bats", "the flying mammals")
    assert q["instructions"]["definition"] == "the flying mammals" and q["type"] == "noul"


def test_blockwise_coupling_keeps_the_same_strongest_links(monkeypatch):
    rng = np.random.default_rng(0)
    A = sparse.csr_matrix((rng.random((300, 80)) < 0.08).astype(np.float32))
    monkeypatch.setattr(T, "BLOCK", 37)
    full = T.coupling(A, True).toarray()
    top = T.coupling(A, True, top=5).toarray()
    for i in range(300):
        nz = top[i][top[i] > 0]
        assert np.allclose(nz, full[i][top[i] > 0])                   # same values
        own = np.sort(full[i])[::-1][:5]
        assert set(np.round(own[own > 0], 9)) <= set(np.round(top[i][top[i] > 0], 9))   # its 5 strongest are kept


def test_blockwise_text_neighbours_equal_the_full_computation(monkeypatch):
    rng = np.random.default_rng(1)
    V = rng.normal(size=(120, 16)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    has = np.ones(120, bool)
    a = T.text_graph(V, has, 7).toarray()
    monkeypatch.setattr(T, "BLOCK", 13)
    b = T.text_graph(V, has, 7).toarray()
    assert np.allclose(a, b)
