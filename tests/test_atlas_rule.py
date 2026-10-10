"""The search API places papers with the same rule as the atlas pipeline (citewalk/science/placement.py)."""
import sys
from pathlib import Path

import numpy as np

from srl_search import atlas

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "citewalk" / "science"))
import placement  # noqa: E402


def test_mode_rule_matches_pipeline():
    rng = np.random.default_rng(0)
    Y = rng.random((500, 2)); spacing = rng.random(500) * 1e-3
    sims = np.sort(rng.random((40, 30)) * 0.3 + 0.6, 1)[:, ::-1]
    nn = rng.integers(0, 500, (40, 30)); works = rng.integers(1, 2**40, 40).astype(np.uint64)
    a = atlas.mode_place(sims, nn, Y, spacing, 0.05, 7.6e-4, 0.25, works)
    b = placement.mode_place(sims, nn, Y, spacing, 0.05, 7.6e-4, 0.25, works)
    assert np.array_equal(a, b)
    assert np.array_equal(atlas.offsets(works), placement.offsets(works))


def test_question_spot_sits_in_the_main_group():
    rng = np.random.default_rng(1)
    main = 0.3 + rng.normal(0, 0.005, (14, 2))
    stray = rng.random((6, 2))
    spot = atlas.question_spot(np.vstack([main[:3], stray, main[3:]]))
    assert np.linalg.norm(spot - 0.3) < 0.01
