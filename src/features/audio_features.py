"""
Audio features for one standardized clip.

Two representations come out of this module:

* the log-mel spectrogram (n_mels x frames), the input of the convolutional
  model, and
* a fixed-length clip vector of 48 hand-built features, the input of the
  classical models and the raw material for the speaker profiles.

Everything is plain numpy so the numbers do not depend on a particular
librosa version; the mel filterbank is the same one phase 1 used.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-10

# Version of the extraction code. It is stored with every feature build and
# checked when features are loaded, so arrays computed by an older version
# of this file are refused instead of being read silently.
#   1  first version
#   2  clip statistics and level normalization use only the recording, not its padding
EXTRACTOR_VERSION = 2


# ------------------------------------------------------------------ mel
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular, area-normalized mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    freqs = _mel_to_hz(mels)
    bins = np.clip(np.floor((n_fft + 1) * freqs / sr).astype(int), 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c == lo:
            c = min(lo + 1, n_bins - 1)
        if hi == c:
            hi = min(c + 1, n_bins - 1)
        for k in range(lo, c):
            fb[m - 1, k] = (k - lo) / max(c - lo, 1)
        for k in range(c, hi):
            fb[m - 1, k] = (hi - k) / max(hi - c, 1)
    fb *= (2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels]))[:, None]
    return fb


def dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    """Orthonormal DCT-II basis of shape (n_mfcc, n_mels)."""
    k = np.arange(n_mels)
    basis = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    basis[0] *= 1.0 / np.sqrt(2.0)
    return (np.sqrt(2.0 / n_mels) * basis).astype(np.float32)


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """
    First-order regression delta along the last axis (time), with edge
    padding. `feat` has shape (n_coeffs, n_frames).
    """
    padded = np.pad(feat, ((0, 0), (width, width)), mode="edge")
    n = feat.shape[1]
    num = sum(w * (padded[:, width + w : width + w + n] - padded[:, width - w : width - w + n]) for w in range(1, width + 1))
    return num / (2.0 * sum(w * w for w in range(1, width + 1)))


# ------------------------------------------------------------------ catalog
N_MFCC_DEFAULT = 13


def feature_names(n_mfcc: int = N_MFCC_DEFAULT) -> list[str]:
    names = [f"mfcc_mean_{i}" for i in range(n_mfcc)]
    names += [f"mfcc_std_{i}" for i in range(n_mfcc)]
    names += [f"dmfcc_std_{i}" for i in range(n_mfcc)]
    names += [
        "centroid_hz",
        "flatness",
        "zcr",
        "noise_floor_db",
        "dyn_range_db",
        "active_frac",
        "f0_hz",
        "voiced_frac",
        "level_db",
    ]
    return names


# What each group of features is, and what it adds that the others do not.
# This table is written out as reports/phase2/feature_catalog.csv.
FEATURE_CATALOG = [
    {
        "group": "log-mel spectrogram",
        "features": "40 mel bands x 98 frames",
        "definition": "log of mel-filtered power spectrum, 25 ms window, 10 ms hop, 20-7600 Hz",
        "adds": "the full time-frequency picture of the clip; the only representation that keeps the order of sounds inside the word",
        "used_by": "convolutional model",
    },
    {
        "group": "MFCC mean",
        "features": "mfcc_mean_0..12",
        "definition": "time average of the first 13 cepstral coefficients",
        "adds": "the average spectral envelope: which vowel-like shape dominates the clip, plus the long-term coloration of the voice and microphone",
        "used_by": "classical models, speaker profiles",
    },
    {
        "group": "MFCC spread",
        "features": "mfcc_std_0..12",
        "definition": "standard deviation over time of the same 13 coefficients",
        "adds": "how much the envelope moves within the clip; separates a steady sound (noise, a held vowel) from a word with several distinct segments",
        "used_by": "classical models, speaker profiles",
    },
    {
        "group": "delta-MFCC spread",
        "features": "dmfcc_std_0..12",
        "definition": "standard deviation over time of the frame-to-frame MFCC slope (regression over +-2 frames)",
        "adds": "the speed of spectral change, which averaging over time throws away; fast transitions mark stop consonants such as the /p/ in 'up' or 'stop'",
        "used_by": "classical models, speaker profiles",
    },
    {
        "group": "spectral shape",
        "features": "centroid_hz, flatness, zcr",
        "definition": "power-weighted mean frequency and geometric/arithmetic mean ratio over active frames; zero-crossing rate over the whole clip",
        "adds": "brightness and noisiness in directly readable units; flatness is close to 1 for broadband noise and close to 0 for voiced speech, so it speaks to the silence class without using loudness",
        "used_by": "classical models, speaker profiles",
    },
    {
        "group": "recording condition",
        "features": "noise_floor_db, dyn_range_db, level_db",
        "definition": "10th percentile of frame energy and its distance to the 95th percentile (both after level normalization); RMS of the clip before normalization",
        "adds": "a description of the channel rather than the voice: how noisy the room or microphone is and how loud the recording was; needed to tell whether speaker groups are voice groups or microphone groups",
        "used_by": "speaker profiles (level_db is excluded from every classifier input)",
    },
    {
        "group": "timing",
        "features": "active_frac",
        "definition": "share of frames within 20 dB of the loudest frame",
        "adds": "how much of the second is occupied by sound: a proxy for word duration and speaking rate",
        "used_by": "classical models, speaker profiles",
    },
    {
        "group": "voicing and pitch",
        "features": "f0_hz, voiced_frac",
        "definition": "median autocorrelation pitch over voiced active frames (60-400 Hz); share of frames judged voiced",
        "adds": "the one property of a voice that MFCCs deliberately smooth away; undefined, by construction, for clips with no voiced frame",
        "used_by": "speaker profiles (f0_hz), classical models (voiced_frac)",
    },
]


# ------------------------------------------------------------------ extractor
class FeatureExtractor:
    """Feature computation with the parameters of one config; reusable across clips."""

    def __init__(self, cfg: dict):
        f = cfg["features"]
        self.sr = int(cfg["signal"]["sample_rate"])
        self.n_fft = int(f["n_fft"])
        self.hop = int(f["hop_length"])
        self.n_mels = int(f["n_mels"])
        self.n_mfcc = int(f["n_mfcc"])
        self.log_floor = float(f["log_floor"])
        self.delta_width = int(f["delta_width"])
        self.activity_db = float(f["activity_db"])
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.fb = mel_filterbank(self.sr, self.n_fft, self.n_mels, f["fmin"], min(f["fmax"], self.sr // 2))
        self.dct = dct_basis(self.n_mfcc, self.n_mels)
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr).astype(np.float32)

        p = f["f0"]
        self.f0_frame = int(p["frame_length"])
        self.f0_lag_min = int(self.sr / p["fmax"])
        self.f0_lag_max = int(self.sr / p["fmin"])
        self.f0_threshold = float(p["voicing_threshold"])
        self.f0_nfft = int(2 ** np.ceil(np.log2(2 * self.f0_frame)))
        self.names = feature_names(self.n_mfcc)

    # -- spectra
    def n_frames(self, n_samples: int) -> int:
        return 1 + max(n_samples - self.n_fft, 0) // self.hop

    def power_spectrogram(self, x: np.ndarray) -> np.ndarray:
        """(frames, n_fft // 2 + 1) power spectrum; frames are not centered."""
        frames = sliding_window_view(x, self.n_fft)[:: self.hop]
        return np.abs(np.fft.rfft(frames * self.window, axis=1)) ** 2

    def log_mel(self, x: np.ndarray, power: np.ndarray | None = None) -> np.ndarray:
        """(n_mels, frames) log-mel spectrogram."""
        if power is None:
            power = self.power_spectrogram(x)
        return np.log(power @ self.fb.T + self.log_floor).T.astype(np.float32)

    # -- pitch
    def f0_track(self, x: np.ndarray, active: np.ndarray) -> np.ndarray:
        """
        Pitch per frame by normalized autocorrelation, NaN where the frame is
        unvoiced or inactive. `active` is the activity mask of the spectral
        frames; the pitch frames are longer but share the hop, so frame i of
        both starts at the same sample.
        """
        frames = sliding_window_view(x, self.f0_frame)[:: self.hop]
        frames = frames - frames.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(frames, n=self.f0_nfft, axis=1)) ** 2
        ac = np.fft.irfft(spec, axis=1)[:, : self.f0_lag_max + 1]
        r0 = ac[:, :1]
        ac = ac / np.maximum(r0, EPS)
        seg = ac[:, self.f0_lag_min :]
        lag = seg.argmax(axis=1)
        peak = seg[np.arange(seg.shape[0]), lag]
        f0 = self.sr / (lag + self.f0_lag_min).astype(np.float64)
        voiced = (peak >= self.f0_threshold) & active[: f0.shape[0]]
        f0[~voiced] = np.nan
        return f0

    # -- clip vector
    def valid_frames(self, n_valid: int, frame_length: int) -> int:
        """Frames of `frame_length` samples (at the common hop) that lie wholly inside the recording."""
        return 0 if n_valid < frame_length else 1 + (n_valid - frame_length) // self.hop

    def clip_vector(self, x: np.ndarray, level_db: float, n_valid: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (feature vector, log-mel).

        `x` is the standardized clip, `level_db` its RMS level before
        normalization and `n_valid` its length before padding (None: no
        padding). The log-mel covers the whole padded second, because that
        is what a model receives. Every summary statistic is computed only
        from frames that lie wholly inside the recording, each with the
        length of its own window: 25 ms for the spectral statistics, 40 ms
        for pitch. Padding a recording therefore leaves the clip vector
        unchanged (checked in tests/test_padding_invariance.py).

        Edge cases
          * A recording shorter than one spectral window (25 ms) keeps its
            first frame, which then includes padding.
          * A recording shorter than one pitch window (40 ms) has no
            measurable pitch: f0_hz is NaN and voiced_frac is 0.
          * A recording without a voiced frame has f0_hz NaN and
            voiced_frac 0; the missing pitch is handled where it is used.
        """
        n_valid = len(x) if n_valid is None else int(min(n_valid, len(x)))
        power = self.power_spectrogram(x)
        lm = self.log_mel(x, power)
        nv = max(1, min(lm.shape[1], self.valid_frames(n_valid, self.n_fft)))

        mf = self.dct @ lm[:, :nv]
        d = delta(mf, self.delta_width)

        frames = sliding_window_view(x, self.n_fft)[:: self.hop][:nv]
        energy_db = 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
        active = energy_db >= energy_db.max() - self.activity_db

        p_act = power[:nv][active] + EPS
        centroid = float(np.mean((p_act @ self.freqs) / p_act.sum(axis=1)))
        flatness = float(np.mean(np.exp(np.mean(np.log(p_act), axis=1)) / np.mean(p_act, axis=1)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x[:n_valid]).astype(np.int8))))) if n_valid > 1 else 0.0
        p10, p95 = np.percentile(energy_db, [10, 95])

        n_pitch = min(self.valid_frames(n_valid, self.f0_frame), nv)
        if n_pitch:
            f0 = self.f0_track(x[: (n_pitch - 1) * self.hop + self.f0_frame], active[:n_pitch])
            n_voiced = int(np.sum(~np.isnan(f0)))
        else:
            f0, n_voiced = np.array([]), 0
        f0_med = float(np.nanmedian(f0)) if n_voiced else np.nan

        vec = np.concatenate(
            [
                mf.mean(axis=1),
                mf.std(axis=1),
                d.std(axis=1),
                [
                    centroid,
                    flatness,
                    zcr,
                    p10,
                    p95 - p10,
                    float(active.mean()),
                    f0_med,
                    n_voiced / n_pitch if n_pitch else 0.0,
                    level_db,
                ],
            ]
        ).astype(np.float32)
        return vec, lm


def column_index(names: list[str], wanted: list[str]) -> np.ndarray:
    pos = {n: i for i, n in enumerate(names)}
    return np.array([pos[w] for w in wanted], dtype=int)


def feature_sets(names: list[str]) -> dict[str, list[str]]:
    """
    Named subsets of the clip vector.

    mfcc_stats : the 26 MFCC mean/std values, the baseline's input.
    classifier : everything a word classifier may see. level_db is left out
                 because recording level is a property of the microphone, and
                 f0_hz because it is undefined for unvoiced clips.
    profile    : everything used to describe a speaker.
    """
    mfcc_stats = [n for n in names if n.startswith(("mfcc_mean_", "mfcc_std_"))]
    classifier = [n for n in names if n not in ("level_db", "f0_hz")]
    return {"mfcc_stats": mfcc_stats, "classifier": classifier, "profile": list(names)}
