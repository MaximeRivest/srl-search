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
    # the map code changed: the selected field is kept, only the maps are made again; meanwhile the map made by the
    # older code opens as it is (groups too), and the remake is queued once
    (tmp_path / key / "field.json").write_text("{}")
    (tmp_path / key / "job.json").write_text(json.dumps({"status": "done", "summary": {"papers": 9}}))
    old = st.get("x y", None, None, "v2", None)
    assert old["groups"] == [{"id": 1}, {"id": 2}] and old["field"]["rebuilding"] is True
    assert st.get("x y", None, None, "v2", 1)["parent"]["group"] == 1
    job = json.loads((tmp_path / key / "job.json").read_text())
    assert job["kind"] == "map" and job["status"] == "queued" and job["summary"] == {"papers": 9}
    assert st.queued() == [key]
    st.get("x y", None, None, "v2", None)
    assert st.queued() == [key]
    # a remake that failed is not asked again on every visit
    (tmp_path / "queue" / f"{key}.job").unlink()
    (tmp_path / key / "job.json").write_text(json.dumps({**job, "status": "failed", "version": "v2"}))
    st.get("x y", None, None, "v2", None)
    assert st.queued() == []
    # once the new code's map is complete, it is the one served
    (tmp_path / key / "maps" / "v2").mkdir()
    (tmp_path / key / "maps" / "v2" / "main.json").write_text(json.dumps({"groups": [{"id": 1}]}))
    assert st.get("x y", None, None, "v2", None) == {"groups": [{"id": 1}]}


def _ask(st, q, asker=None):
    tok = F.ASKER.set(asker)
    try:
        st.get(q, None, None, "v1", None)
    except F.Pending as e:
        return e.status
    finally:
        F.ASKER.reset(tok)


def test_free_line_limits(tmp_path, monkeypatch):
    st = F.FieldStore(tmp_path)
    assert _ask(st, "a b", "r1")["position"] == 1 and _ask(st, "c d", "r1")["position"] == 2
    with pytest.raises(HTTPException) as e:                      # 2 waiting per reader
        _ask(st, "e f", "r1")
    assert e.value.status_code == 429
    assert _ask(st, "e f", "r2")["lane"] == "free"
    assert _ask(st, "a b", "r3")["position"] == 1                # asking again: the same place, not a new one
    monkeypatch.setattr(F, "FREE_MAX_QUEUED", 3)
    with pytest.raises(HTTPException) as e:
        _ask(st, "g h", "r4")
    assert e.value.status_code == 503 and "full" in e.value.detail


def test_spent_budget_moves_the_free_line_to_tomorrow_and_paid_maps_go_first(tmp_path, monkeypatch):
    from srl_search import judges as J
    monkeypatch.setattr(J, "LOCAL", False)                      # paid judges: a budget in dollars
    st = F.FieldStore(tmp_path)
    for q in ("a b", "c d"):
        _ask(st, q)
    (tmp_path / "spend_lanes.json").write_text(json.dumps({date.today().isoformat(): {"free": F.FREE_USD}}))
    pl = st.plan()
    assert pl["waiting"][0]["start"] >= F._next_midnight(pl["now"]) - 1
    key = F.field_key("c d")
    (tmp_path / "paid").mkdir()
    (tmp_path / "paid" / f"{key}.json").write_text(json.dumps({"accepted": True, "paid_at": F._now()}))
    pl = st.plan()
    assert pl["waiting"][0]["key"] == key and pl["waiting"][0]["start"] - pl["now"] < 120
    s = st.status(key)
    assert s["lane"] == "paid" and s["position"] == 1 and "paid_eta_s" not in s and "asker" not in s
    assert "paid_eta_s" in st.status(F.field_key("a b"))
    view = st.queue_view()
    assert [w["line"] for w in view["waiting"]] == ["paid", "free"] and view["waiting"][0]["topic"] == "c d"


def test_worker_tries_a_failed_paid_map_once_more(tmp_path, monkeypatch):
    st = F.FieldStore(tmp_path)
    _ask(st, "a b")
    key = F.field_key("a b")
    (tmp_path / "paid").mkdir()
    (tmp_path / "paid" / f"{key}.json").write_text(json.dumps({"accepted": True, "paid_at": F._now()}))
    runs = []

    def fake_run(ctx, store, k):
        runs.append(k)
        j = json.loads((store / k / "job.json").read_text())
        j.update({"status": "failed", "final": False, "error": "boom"})
        (store / k / "job.json").write_text(json.dumps(j))
    monkeypatch.setattr(F, "run_job", fake_run)
    monkeypatch.setattr(F, "Ctx", lambda: object())
    F.worker(tmp_path)
    assert runs == [key, key] and not list((tmp_path / "queue").glob("*.job"))
    assert json.loads((tmp_path / key / "job.json").read_text())["retried"] is True


def test_a_topic_that_cannot_be_mapped_says_why_once(tmp_path):
    key = F.field_key("everything")
    (tmp_path / key).mkdir(parents=True)
    (tmp_path / key / "job.json").write_text(json.dumps({"status": "failed", "final": True, "error": "too broad"}))
    with pytest.raises(HTTPException) as e:
        F.FieldStore(tmp_path).get("everything", None, None, "v1", None)
    assert e.value.status_code == 422 and e.value.detail == "too broad"


def test_definition_is_part_of_the_identity():
    assert F.field_key("bats") != F.field_key("bats", "the flying mammals")
    from srl_search.judges import instruction, paper_text
    assert "the flying mammals" in instruction("bats", "the flying mammals")
    assert paper_text("<i>T</i>", None) == "Title: T\n(no abstract)" and "Abstract: a b" in paper_text("T", "a\n b")


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


def test_paused_blocks_new_fields_but_not_map_rebuilds(tmp_path):
    (tmp_path / "PAUSED").write_text("Paused while the judge changes.")
    st = F.FieldStore(tmp_path)
    with pytest.raises(HTTPException) as e:
        st.get("new topic", None, None, "v1", None)
    assert e.value.status_code == 503 and "Paused while the judge changes." in e.value.detail
    key = F.field_key("old topic")
    (tmp_path / key).mkdir()
    (tmp_path / key / "field.json").write_text("{}")
    with pytest.raises(F.Pending) as p:
        st.get("old topic", None, None, "v2", None)
    assert p.value.status["kind"] == "map"


def test_leiden_runs_side_by_side_give_the_same_groups(monkeypatch):
    rng = np.random.default_rng(3)
    n = 600
    blocks = np.repeat(np.arange(4), n // 4)
    r, c = np.nonzero(np.triu((rng.random((n, n)) < np.where(blocks[:, None] == blocks[None, :], 0.05, 0.004)), 1))
    W = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    W = (W + W.T).tocsr()
    serial = T.cluster(W, T.Options(), 20)
    monkeypatch.setattr(T, "PARALLEL_MIN", 10)
    parallel = T.cluster(W, T.Options(workers=8), 20)
    assert (serial[0] == parallel[0]).all() and serial[2]["resolution"] == parallel[2]["resolution"]


def test_general_judge_quota_counts_free_maps_made_today(tmp_path, monkeypatch):
    from srl_search import judges as J
    monkeypatch.setattr(J, "LOCAL", True)
    monkeypatch.setattr(F, "FREE_PER_DAY", 2)
    st = F.FieldStore(tmp_path)
    for i in range(2):                                          # two free maps already made today
        d = tmp_path / f"done{i}"
        d.mkdir()
        (d / "job.json").write_text(json.dumps({"status": "done", "kind": "full", "lane": "free", "finished": F._now()}))
    _ask(st, "a b")
    pl = st.plan()
    assert pl["waiting"][0]["start"] >= F._next_midnight(pl["now"]) - 1
    assert st.queue_view()["free_per_day"] == 2
    monkeypatch.setattr(F, "FREE_PER_DAY", 3)
    pl = st.plan()
    assert pl["waiting"][0]["start"] - pl["now"] < 120
