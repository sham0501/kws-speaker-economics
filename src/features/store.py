"""
Access to the arrays written by build_features.py.

Every consumer goes through these functions. They read the full build only
(a --limit smoke test lives in its own folder and is read only on request),
and refuse arrays that were built from another manifest, with other signal
or feature settings, or by an older version of the extraction code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import manifest_fingerprint, settings_fingerprint  # noqa: E402
from utils.common import resolve  # noqa: E402


class StaleFeatures(RuntimeError):
    pass


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["manifest"]))


def _build_dir(cfg: dict, limited: bool) -> Path:
    return resolve(cfg["paths"]["processed_dir"]) / ("limited" if limited else "")


def _checked_meta(cfg: dict, limited: bool) -> tuple[dict, pd.DataFrame]:
    proc = _build_dir(cfg, limited)
    meta_path = proc / "features_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run src/features/build_features.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rebuild = "rebuild them with: python src/features/build_features.py"
    if bool(meta.get("limit")) != limited:
        raise StaleFeatures(f"{proc} holds a {'limited' if meta.get('limit') else 'full'} build; {rebuild}")
    if meta.get("settings_sha1") != settings_fingerprint(cfg):
        raise StaleFeatures(
            "clip features were built with other signal/feature settings or an older version of "
            f"the extraction code (version {meta.get('extractor_version', 'unknown')}); {rebuild}"
        )
    man = load_manifest(cfg)
    if manifest_fingerprint(man) != meta["manifest_sha1"]:
        raise StaleFeatures(f"clip features were built from a different manifest; {rebuild}")
    if meta.get("manifest_rows") is not None:
        man = man.iloc[meta["manifest_rows"]].reset_index(drop=True)
    if len(man) != meta["n_clips"]:
        raise StaleFeatures(f"feature build has {meta['n_clips']} rows, manifest selection has {len(man)}; {rebuild}")
    return meta, man


def load_features(cfg: dict, limited: bool = False) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns (manifest rows, clip feature matrix, feature names), row-aligned."""
    meta, man = _checked_meta(cfg, limited)
    feats = np.load(_build_dir(cfg, limited) / "clip_features.npy")
    if feats.shape != (len(man), len(meta["feature_names"])):
        raise StaleFeatures(f"clip_features.npy has shape {feats.shape}, expected {(len(man), len(meta['feature_names']))}")
    return man, feats, meta["feature_names"]


def load_waveforms(cfg: dict, mmap: bool = True, limited: bool = False) -> np.ndarray:
    """int16 array (n_clips, n_samples), row-aligned with the manifest rows of the same build."""
    meta, _ = _checked_meta(cfg, limited)
    waves = np.load(_build_dir(cfg, limited) / "waveforms.npy", mmap_mode="r" if mmap else None)
    if waves.shape != (meta["n_clips"], cfg["signal"]["n_samples"]):
        raise StaleFeatures(f"waveforms.npy has shape {waves.shape}; rebuild with build_features.py")
    return waves


def load_lengths(cfg: dict, limited: bool = False) -> np.ndarray:
    """Number of samples each clip had before padding."""
    _checked_meta(cfg, limited)
    return np.load(_build_dir(cfg, limited) / "clip_lengths.npy")


def class_order(cfg: dict) -> list[str]:
    lab = cfg["labels"]
    return list(lab["keywords"]) + [lab["unknown"], lab["silence"]]
