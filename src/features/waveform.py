"""
Signal standardization: every clip leaves this module as a mono float32
array of exactly `n_samples` samples at `sample_rate`.

Two decisions are made here and nowhere else:

* Length. Clips shorter than the target are zero-padded at the end; longer
  ones keep their first `n_samples`. 8.66% of the subset is shorter than one
  second and nothing is longer, so cropping never actually discards speech.
* Level. Each clip is scaled to a common RMS. Recording level in this
  dataset reflects the microphone and the distance to it, not the word, and
  the silence class spans a 150x range of levels across its source files.
  Removing level as a cue stops a model from separating classes on loudness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import fix_length, read_wav  # noqa: E402


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def load_clip(path, sample_rate: int, n_samples: int) -> tuple[np.ndarray, int]:
    """
    Read a clip and bring it to the target length.
    Returns (signal, original_length). The sample rate is checked rather than
    converted: phase 1 established that every file is already 16 kHz, so a
    mismatch means the wrong data, not something to silently resample.
    """
    sig, sr = read_wav(path)
    if sr != sample_rate:
        raise ValueError(f"{path}: sample rate {sr}, expected {sample_rate}")
    return fix_length(sig, n_samples), int(sig.shape[0])


def normalize_level(
    x: np.ndarray, target_rms: float, max_gain_db: float, n_valid: int | None = None
) -> tuple[np.ndarray, float]:
    """
    Scale `x` to `target_rms`, never applying more than `max_gain_db` of gain.
    The level is measured on the first `n_valid` samples (the recording
    without its zero padding), so padding a clip does not make it louder.
    Returns (scaled signal, applied gain in dB). A silent recording is
    returned unchanged with a gain of 0 dB.
    """
    level = rms(x[:n_valid] if n_valid is not None else x)
    if level <= 0.0:
        return x.astype(np.float32), 0.0
    gain = min(target_rms / level, 10.0 ** (max_gain_db / 20.0))
    return (x * gain).astype(np.float32), float(20.0 * np.log10(gain))


def standardize(x: np.ndarray, cfg: dict, n_valid: int | None = None) -> tuple[np.ndarray, float]:
    """
    Apply the level rule from the config to an already length-fixed clip.
    `n_valid` is the clip's length before padding; None means no padding.
    """
    ln = cfg["signal"]["level_norm"]
    if not ln["enabled"]:
        return x.astype(np.float32), 0.0
    return normalize_level(x, ln["target_rms"], ln["max_gain_db"], n_valid)
