import json

import numpy as np

from srl_search.engine import passes, rank
from srl_search.quality import FEATURES, Quality


def make_quality(root, with_model=True):
    ids = np.array([10, 20, 30], dtype=np.int64)
    cols = {
        "is_xpac": np.array([0, 1, 0], bool), "is_paratext": np.zeros(3, bool),
        "source_type": np.array([1, 2, 0], np.uint8), "source_is_core": np.array([1, 0, 0], bool),
        "source_in_doaj": np.zeros(3, bool), "indexed_in": np.array([3, 0, 0], np.uint8),
        "refs": np.array([40, 0, 5], np.int16), "authors": np.array([5, 1, 2], np.int16),
        "locations": np.array([3, 1, 1], np.int16), "citation_pct": np.array([0.9, np.nan, 0.1], np.float32),
        "source": np.array([7, 9, 0], np.int64),
    }
    np.save(root / "ids.npy", ids)
    for k, v in cols.items():
        np.save(root / f"{k}.npy", v)
    np.savez(root / "sources.npz", source=np.array([7, 9]), works=np.array([1000, 50]),
             no_refs=np.array([0.1, 0.9], np.float32), single_author=np.array([0.2, 0.8], np.float32),
             uncited=np.array([0.3, 0.95], np.float32), xpac=np.array([0.0, 0.7], np.float32))
    (root / "manifest.json").write_text(json.dumps({"works": 3, "built_utc": "x"}))
    if with_model:
        n = len(FEATURES)
        w = np.zeros(n, np.float32)
        w[FEATURES.index("no_refs")] = 2.0
        np.savez(root / "model.npz", features=np.array(FEATURES), mean=np.zeros(n, np.float32),
                 scale=np.ones(n, np.float32), w_meta=w, w_vec=np.zeros(4, np.float32), b=np.float32(-1),
                 w_meta_only=w, b_meta_only=np.float32(-1))
    return Quality(root)


def test_features_follow_the_records(tmp_path):
    q = make_quality(tmp_path)
    f = q.features(np.array([30, 10, 20, 99]))
    col = {name: f[:, i] for i, name in enumerate(FEATURES)}
    assert list(col["no_refs"]) == [0, 0, 1, 0]
    assert list(col["single_author"]) == [0, 0, 1, 0]
    assert list(col["source_type=1"]) == [0, 1, 0, 0]
    assert list(col["no_source"]) == [1, 0, 0, 0]          # source 0: none
    assert col["src_no_refs"][2] == np.float32(0.9)
    assert list(col["unknown_record"]) == [0, 0, 0, 1]      # W99 is not in the corpus: all else zero
    assert f[3, :-1].sum() == 0


def test_junk_probability_uses_the_model(tmp_path):
    q = make_quality(tmp_path)
    p = q.junk(np.array([10, 20]), np.zeros((2, 4), np.float32))
    assert p[1] > p[0]                                     # no references -> more likely junk
    assert np.allclose(p, q.junk(np.array([10, 20])))      # record-only model here has the same weights


def test_rank_demotes_junk_but_keeps_similarity(tmp_path):
    q = make_quality(tmp_path)
    wids = np.array([20, 10, 30])
    sims = np.array([0.80, 0.75, 0.50], np.float32)
    hits, _ = rank(wids, sims, np.zeros((3, 4), np.float32), q, 0.3, exclude=None)
    assert [h[0] for h in hits] == [10, 20, 30]            # W20 (junk-like) falls below W10
    assert hits[1][2] == np.float32(0.80)                  # reported similarity unchanged
    plain, _ = rank(wids, sims, np.zeros((3, 4), np.float32), q, 0.0, exclude=20)
    assert [h[0] for h in plain] == [10, 30]               # penalty 0 = similarity order; exclude works
    assert plain[0][3] is not None                         # junk is still reported
    none, t = rank(wids, sims, None, None, 0.3, exclude=None)
    assert [h[0] for h in none] == [20, 10, 30] and t is None and none[0][3] is None


def test_retracted_titles_are_left_out():
    d = {"year": 2025, "type": "article", "is_oa": 1, "license": "cc-by", "pmc_oa": 1, "is_retracted": 0}
    assert not passes({**d, "title": "[RETRACTED] MetaBoost Connection Weight Loss System"}, {})
    assert not passes({**d, "title": "Retraction Note: Effects of x on y"}, {})
    assert passes({**d, "title": "Retracted papers in medicine: a bibliometric study"}, {})
    assert passes({**d, "title": "[RETRACTED] x"}, {"exclude_retracted": False})
