"""
Shared utilities: config loading, paths, seeding, and metadata extraction
from Speech Commands filenames.

Official dataset naming convention:
    <label>/<speaker_id>_nohash_<utterance_index>.wav
Example: yes/0a7c2a8d_nohash_0.wav
So speaker_id is the part before "_nohash_" and is an anonymous hash. No
attempt is made to deanonymize it; it is used only to design a
speaker-independent split and to analyze generalization.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Repo root = two levels above this file (src/utils/common.py -> src/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Official Speech Commands filename pattern
_FILENAME_RE = re.compile(r"^(?P<speaker>[0-9a-f]+)_nohash_(?P<utt>\d+)\.wav$")


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML config file. Default path: configs/phase1.yaml"""
    if path is None:
        path = REPO_ROOT / "configs" / "phase1.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Guard against a classic YAML 1.1 trap: unquoted yes/no/on/off values
    # get coerced to booleans, which would silently drop four of this
    # project's ten keywords. Fail loudly instead.
    kws = cfg.get("subset", {}).get("keywords", [])
    bad = [k for k in kws if not isinstance(k, str)]
    if bad:
        raise ValueError(
            f"Keywords {bad} were coerced from strings to booleans in the config. "
            'Quote every keyword, e.g.: - "yes"'
        )
    if len(set(kws)) != len(kws):
        raise ValueError("Duplicate keyword found in config.")
    return cfg


def resolve(path_like: str | os.PathLike) -> Path:
    """Resolve a path from config (relative) into an absolute path under the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def set_seed(seed: int) -> None:
    """Fix the seed for reproducibility; the value used is recorded in the config."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_speaker_id(filename: str) -> str | None:
    """
    Extract the speaker id from a filename.
    None means the filename does not match the official naming convention
    (reported during the audit).
    """
    m = _FILENAME_RE.match(Path(filename).name)
    return m.group("speaker") if m else None


def parse_utterance_index(filename: str) -> int | None:
    """Extract the utterance index (how many times this speaker said this word before)."""
    m = _FILENAME_RE.match(Path(filename).name)
    return int(m.group("utt")) if m else None


def relpath_key(label: str, filename: str) -> str:
    """
    The unique key for a file exactly as it appears in
    validation_list.txt / testing_list.txt: "<label>/<filename>".
    Used to match against Google's official split.
    """
    return f"{label}/{Path(filename).name}"


def ensure_dir(path: str | os.PathLike) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_wav(path: str | os.PathLike):
    """
    Read a wav file as (signal: np.float32 in [-1, 1], sample_rate: int).
    Tries soundfile first, falling back to scipy.io.wavfile so this module
    doesn't hard-depend on either.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        data = data.astype(np.float32)
        if np.issubdtype(np.dtype(data.dtype), np.floating) is False:
            pass
        # Normalize 16-bit PCM to [-1, 1]
        if data.dtype != np.float32 or np.max(np.abs(data)) > 1.5:
            data = data / 32768.0
    if data.ndim > 1:  # everything in this dataset is mono, but be safe
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def fix_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Make a signal exactly target_len samples long by zero-padding at the
    end or cropping from the start.
    (Design decision, recorded for phase 2: pad with zeros at the end,
    crop from the start, no centering.)
    """
    n = signal.shape[0]
    if n == target_len:
        return signal
    if n > target_len:
        return signal[:target_len]
    out = np.zeros(target_len, dtype=signal.dtype)
    out[:n] = signal
    return out
