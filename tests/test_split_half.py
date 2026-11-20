"""
The two halves of the split-half reliability check must be disjoint sets of
exactly m clips per speaker, for every configured profile size, and their
agreement must be scored as Cohen's kappa, with chance agreement taken from
the cluster shares of each half separately.

    python tests/test_split_half.py      (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from patterns.speaker_clusters import agreement, split_halves  # noqa: E402
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


def test_agreement_is_cohens_kappa():
    # The two halves put different shares of speakers in cluster 1 (3/10 and
    # 6/10), the case where pooling both halves into one distribution would
    # give a different chance agreement.
    ca = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    cb = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    observed, chance, kappa = agreement(ca, cb, 2)
    assert observed == 0.7
    assert np.isclose(chance, 0.7 * 0.4 + 0.3 * 0.6)
    assert np.isclose(kappa, (observed - chance) / (1 - chance))
    assert np.isclose(kappa, cohen_kappa_score(ca, cb))
    pooled = np.bincount(np.concatenate([ca, cb]), minlength=2) / 20
    assert not np.isclose(chance, np.sum(pooled**2))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
