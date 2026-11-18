#!/usr/bin/env python3
"""
Waveform augmentation for training clips.

Three transforms, each with its strength in configs/phase2.yaml:

    time shift   the word is not always at the same offset in a recording
    gain         the same voice at a different distance from the microphone
    added noise  the same word in a noisier room

They act on the raw, length-fixed waveform, before level normalization and
feature extraction, and are meant for the training split only: an augmented
copy of a validation or test clip would make the evaluation easier than the
real task. The noise comes exclusively from the background files that phase
1 assigned to the training split, for the same reason.

Run as a script to write the settings and an example figure:
    python src/features/augment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.waveform import rms  # noqa: E402
from utils.common import load_config, read_wav, resolve  # noqa: E402


def time_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift by `shift` samples (positive = later), filling the gap with zeros."""
    if shift == 0:
        return x
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add `noise` to `x` so that the ratio of their RMS levels is `snr_db`.
    The level of `x` is measured over the whole clip. If either signal is
    silent the clip is returned unchanged.
    """
    sx, sn = rms(x), rms(noise)
    if sx <= 0.0 or sn <= 0.0:
        return x
    scale = sx / (sn * 10.0 ** (snr_db / 20.0))
    return (x + scale * noise).astype(np.float32)


class NoiseBank:
    """Background recordings of one split, served as random one-clip segments."""

    def __init__(self, cfg: dict, split: str = "train"):
        summary = json.loads(
            (resolve(cfg["paths"]["phase1_reports"]) / "subset_summary.json").read_text(encoding="utf-8")
        )
        self.files = list(summary["silence_noise_file_assignment"][split])
        noise_dir = resolve(cfg["paths"]["noise_dir"])
        self.n_samples = int(cfg["signal"]["n_samples"])
        self.signals = []
        for name in self.files:
            sig, sr = read_wav(noise_dir / name)
            if sr != cfg["signal"]["sample_rate"]:
                raise ValueError(f"{name}: sample rate {sr}")
            self.signals.append(sig)

    def sample(self, rng: np.random.Generator, source: int | None = None) -> np.ndarray:
        sig = self.signals[rng.integers(len(self.signals)) if source is None else source]
        start = rng.integers(0, len(sig) - self.n_samples)
        return sig[start : start + self.n_samples]


class TrainAugmenter:
    """Random time shift, gain and noise, drawn independently for every call."""

    def __init__(self, cfg: dict, noise_bank: NoiseBank | None = None):
        a = cfg["augmentation"]
        self.max_shift = int(round(a["time_shift_ms"] * cfg["signal"]["sample_rate"] / 1000.0))
        self.gain_db = float(a["gain_db"])
        self.noise_prob = float(a["noise_prob"])
        self.snr_lo, self.snr_hi = map(float, a["noise_snr_db"])
        self.noise_bank = noise_bank

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = time_shift(x, int(rng.integers(-self.max_shift, self.max_shift + 1)))
        x = apply_gain(x, rng.uniform(-self.gain_db, self.gain_db))
        if self.noise_bank is not None and rng.random() < self.noise_prob:
            x = mix_at_snr(x, self.noise_bank.sample(rng), rng.uniform(self.snr_lo, self.snr_hi))
        return x

    def settings(self) -> dict:
        return {
            "applied_to": "train split only",
            "domain": "raw waveform, before level normalization",
            "time_shift_samples": [-self.max_shift, self.max_shift],
            "gain_db": [-self.gain_db, self.gain_db],
            "noise_probability": self.noise_prob,
            "noise_snr_db": [self.snr_lo, self.snr_hi],
            "noise_files": self.noise_bank.files if self.noise_bank else [],
        }


def _demo() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from features.audio_features import FeatureExtractor
    from features.waveform import load_clip, standardize
    from utils.common import clip_path, ensure_dir

    cfg = load_config("configs/phase2.yaml")
    rng = np.random.default_rng(cfg["seed"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    rep_dir = ensure_dir(cfg["paths"]["reports_dir"])

    bank = NoiseBank(cfg, "train")
    aug = TrainAugmenter(cfg, bank)
    fx = FeatureExtractor(cfg)
    sr, n = cfg["signal"]["sample_rate"], cfg["signal"]["n_samples"]

    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    train = man[(man["split"] == "train") & (man["source"] == "keyword")]
    row = train.sample(1, random_state=cfg["seed"]).iloc[0]
    x, _ = load_clip(clip_path(row["filepath"]), sr, n)

    shift = aug.max_shift
    variants = [
        ("original", x),
        (f"shift +{cfg['augmentation']['time_shift_ms']} ms", time_shift(x, shift)),
        (f"gain -{aug.gain_db:.0f} dB", apply_gain(x, -aug.gain_db)),
        (f"noise at {aug.snr_lo:.0f} dB SNR", mix_at_snr(x, bank.sample(rng), aug.snr_lo)),
        ("random draw", aug(x, rng)),
    ]
    mels = [fx.log_mel(standardize(v, cfg)[0]) for _, v in variants]
    vmin, vmax = min(m.min() for m in mels), max(m.max() for m in mels)

    fig, axes = plt.subplots(2, len(variants), figsize=(2.9 * len(variants), 4.2), squeeze=False)
    t = np.arange(n) / sr
    for j, ((title, v), lm) in enumerate(zip(variants, mels)):
        axes[0, j].plot(t, v, lw=0.5)
        axes[0, j].set_ylim(-1, 1)
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].set_xticks([])
        axes[1, j].imshow(lm, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, extent=[0, 1, 0, fx.n_mels])
        axes[1, j].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("amplitude")
    axes[1, 0].set_ylabel("mel band")
    fig.suptitle(f"Augmentation of one training clip ('{row['label']}'); log-mel after level normalization", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_augmentation_examples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Level normalization follows augmentation, so a pure gain change must
    # leave the features untouched. Checked here rather than assumed.
    base = fx.log_mel(standardize(x, cfg)[0])
    gained = fx.log_mel(standardize(apply_gain(x, -aug.gain_db), cfg)[0])
    settings = aug.settings()
    settings["gain_invariance_max_abs_logmel_diff"] = float(np.max(np.abs(base - gained)))
    (rep_dir / "augmentation_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
