"""
The two halves of the split-half reliability check must be disjoint sets of
exactly m clips per speaker, for every configured profile size.

    python tests/test_split_half.py      (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from patterns.speaker_clusters import split_halves  # noqa: E402
from utils.common import load_config  # noqa: E402

SIZES = load_config("configs/phase2.yaml")["clustering"]["reliability_clips"]


def test_halves_are_disjoint_and_complete():
    rng = np.random.default_rng(0)
    rows = {f"spk{i}": np.arange(100 * i, 100 * i + 2 * max(SIZES) + i) for i in range(30)}
    for m in SIZES:
        a, b = split_halves(rows, m, rng)
        assert a.keys() == b.keys() == rows.keys()
        for s in rows:
            assert len(a[s]) == len(b[s]) == m, (m, s)
            assert len(set(a[s])) == m and len(set(b[s])) == m
            assert not set(a[s]) & set(b[s]), (m, s)
            assert set(a[s]) | set(b[s]) <= set(rows[s])


def test_m_equal_one():
    rows = {"x": np.array([7, 8])}
    for seed in range(50):
        a, b = split_halves(rows, 1, np.random.default_rng(seed))
        assert sorted([a["x"][0], b["x"][0]]) == [7, 8]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
