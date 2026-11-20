"""Checks for the speaker-range calculation behind phase-1 figure 3.

    python tests/test_budget_range.py      (or: python -m pytest tests)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eda.run_eda import speaker_range  # noqa: E402


def test_example_from_review():
    # capacities 5, 5, 1, 1; at most 5 clips each; a budget of 6 clips
    assert speaker_range([5, 5, 1, 1], k=5, budget=6) == (2, 4)


def test_cap_raises_the_minimum():
    # with at most 1 clip each, every clip needs its own speaker
    assert speaker_range([5, 5, 1, 1], k=1, budget=4) == (4, 4)
    assert speaker_range([5, 5, 1, 1], k=2, budget=4) == (2, 4)


def test_budget_bounds_the_maximum():
    assert speaker_range([3, 3, 3, 3, 3], k=3, budget=2) == (1, 2)


def test_infeasible_budget():
    assert speaker_range([5, 5, 1, 1], k=2, budget=7) is None
    assert speaker_range([], k=3, budget=1) is None


def test_exact_fill():
    assert speaker_range([4, 4], k=4, budget=8) == (2, 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
