"""
Zero padding must not change what the clip statistics say about the recording.

A clip shorter than one second is padded with zeros at the end. The padded
clip and the bare recording must give the same clip vector (everything
except the log-mel, which covers the whole padded second on purpose), and
the same holds for level normalization.

    python tests/test_padding_invariance.py      (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.audio_features import FeatureExtractor  # noqa: E402
from features.waveform import standardize  # noqa: E402
from utils.common import load_config  # noqa: E402

CFG = load_config("configs/phase2.yaml")
FX = FeatureExtractor(CFG)
SR = CFG["signal"]["sample_rate"]


def tone(n: int, hz: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def word_like(n: int, seed: int = 0) -> np.ndarray:
    """A voiced burst with silence on both sides, plus a little noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, n).astype(np.float32)
    a, b = n // 4, 3 * n // 4
    x[a:b] += tone(b - a, 150.0, 0.4) * np.hanning(b - a).astype(np.float32)
    return x


def pad(x: np.ndarray, total: int) -> np.ndarray:
    return np.concatenate([x, np.zeros(total - len(x), dtype=np.float32)])


def vector(x: np.ndarray, n_valid: int | None) -> np.ndarray:
    y, _ = standardize(x, CFG, n_valid)
    return FX.clip_vector(y, 0.0, n_valid)[0]


def test_tone_statistics_ignore_padding():
    x = tone(8000)
    v = vector(pad(x, 16000), 8000)
    names = FX.names
    assert v[names.index("active_frac")] == 1.0
    assert v[names.index("voiced_frac")] == 1.0
    assert abs(v[names.index("f0_hz")] - 200.0) < 3.0


def test_padded_equals_unpadded():
    for x in (tone(8000), word_like(9000), word_like(12345, seed=3)):
        bare = vector(x, None)
        for total in (len(x) + 1, 12800, 16000):
            padded = vector(pad(x, total), len(x))
            np.testing.assert_allclose(padded, bare, rtol=1e-5, atol=1e-5)


def test_level_normalization_ignores_padding():
    x = word_like(7000)
    a, gain_a = standardize(pad(x, 16000), CFG, 7000)
    b, gain_b = standardize(x, CFG, None)
    assert abs(gain_a - gain_b) < 1e-6
    np.testing.assert_allclose(a[:7000], b, rtol=1e-6, atol=1e-7)
    assert not a[7000:].any()


def test_clip_shorter_than_a_pitch_window():
    x = tone(500)  # 31 ms: shorter than the 40 ms pitch window
    v = vector(pad(x, 16000), 500)
    names = FX.names
    assert np.isnan(v[names.index("f0_hz")])
    assert v[names.index("voiced_frac")] == 0.0
    assert np.isfinite(np.delete(v, names.index("f0_hz"))).all()


def test_full_length_clip_unchanged_by_n_valid():
    x = word_like(16000)
    np.testing.assert_array_equal(vector(x, None), vector(x, 16000))


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "-", type(e).__name__, str(e).splitlines()[0][:120] if str(e) else "")
    sys.exit(1 if failed else 0)
