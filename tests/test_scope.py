import sys
from pathlib import Path

from srl_search.scope import SCHOLARLY_TYPES


def test_same_publication_types_as_the_atlas():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "citewalk" / "science"))
    import scope
    assert SCHOLARLY_TYPES == scope.SCHOLARLY_TYPES


def test_non_publications_are_left_out_unless_asked():
    from srl_search.engine import passes
    d = {"year": 2020, "type": "dataset", "is_oa": True, "license": None, "pmc_oa": 0, "is_retracted": False}
    assert not passes(d, {})
    assert passes(d, {"type": "dataset"})
    assert passes({**d, "type": "article"}, {})


def test_every_openalex_type_is_classified():
    import json
    from srl_search.scope import NON_PUBLICATION_TYPES
    meta = Path("/mnt/fast/citewalk/20261007T003120Z-50ce3865/meta.json")
    types = set(json.loads(meta.read_text())["types"])
    assert SCHOLARLY_TYPES.isdisjoint(NON_PUBLICATION_TYPES)
    assert SCHOLARLY_TYPES | set(NON_PUBLICATION_TYPES) == types
