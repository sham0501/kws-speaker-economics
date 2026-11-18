"""
Access to the arrays written by build_features.py.

Every consumer goes through `load_features`, which refuses to pair a
feature array with a manifest it was not built from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import manifest_fingerprint  # noqa: E402
from utils.common import resolve  # noqa: E402


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["manifest"]))


def load_features(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns (manifest, clip feature matrix, feature names), row-aligned."""
    proc = resolve(cfg["paths"]["processed_dir"])
    meta_path = proc / "features_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run src/features/build_features.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    man = load_manifest(cfg)
    if meta.get("limit"):
        man = man.iloc[: meta["limit"]].reset_index(drop=True)
    if manifest_fingerprint(man) != meta["manifest_sha1"]:
        raise RuntimeError("clip features were built from a different manifest; rebuild them")
    feats = np.load(proc / "clip_features.npy")
    return man, feats, meta["feature_names"]


def load_waveforms(cfg: dict, mmap: bool = True) -> np.ndarray:
    """int16 array (n_clips, n_samples), row-aligned with the manifest."""
    path = resolve(cfg["paths"]["processed_dir"]) / "waveforms.npy"
    return np.load(path, mmap_mode="r" if mmap else None)


def load_lengths(cfg: dict) -> np.ndarray:
    """Number of samples each clip had before padding."""
    return np.load(resolve(cfg["paths"]["processed_dir"]) / "clip_lengths.npy")


def class_order(cfg: dict) -> list[str]:
    lab = cfg["labels"]
    return list(lab["keywords"]) + [lab["unknown"], lab["silence"]]
