#!/usr/bin/env python3
"""
Run the preparation pipeline over the whole subset manifest.

For every clip: read -> fix length -> record level -> normalize level ->
log-mel / MFCC / descriptors. Row i of every output array corresponds to
row i of data/subset_manifest.csv.

Outputs
    data/processed/phase2/waveforms.npy       int16  (N, 16000)  length-fixed, original level
    data/processed/phase2/clip_features.npy   float32 (N, 48)
    data/processed/phase2/clip_lengths.npy    int64  (N,)        samples before padding
    data/processed/phase2/features_meta.json  feature names + manifest fingerprint
    reports/phase2/features_summary.json      what was done to how many clips
    reports/phase2/feature_catalog.csv        the feature list with the reason for each group

Usage
    python src/features/build_features.py
    python src/features/build_features.py --limit 2000 --jobs 1   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_features import FEATURE_CATALOG, FeatureExtractor  # noqa: E402
from features.waveform import load_clip, rms, standardize  # noqa: E402
from utils.common import clip_path, ensure_dir, load_config, resolve  # noqa: E402

_CFG = None
_FX = None


def _init(cfg):
    global _CFG, _FX
    _CFG = cfg
    _FX = FeatureExtractor(cfg)


def _process(chunk):
    """chunk: list of (row index, manifest filepath). Runs in a worker process."""
    sr, n = _CFG["signal"]["sample_rate"], _CFG["signal"]["n_samples"]
    idx, waves, feats, lengths, gains = [], [], [], [], []
    for i, fp in chunk:
        sig, orig_len = load_clip(clip_path(fp), sr, n)
        level = rms(sig[:orig_len])
        level_db = 20.0 * np.log10(level + 1e-10)
        x, gain_db = standardize(sig, _CFG)
        vec, _ = _FX.clip_vector(x, level_db, min(orig_len, n))
        idx.append(i)
        waves.append(np.round(sig * 32768.0).clip(-32768, 32767).astype(np.int16))
        feats.append(vec)
        lengths.append(orig_len)
        gains.append(gain_db)
    return idx, np.stack(waves), np.stack(feats), lengths, gains


def manifest_fingerprint(man: pd.DataFrame) -> str:
    return hashlib.sha1("\n".join(man["filepath"].astype(str)).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N manifest rows")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    cfg = load_config(args.config)
    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    if args.limit:
        man = man.iloc[: args.limit].reset_index(drop=True)
    n_clips, n_samples = len(man), cfg["signal"]["n_samples"]

    out_dir = ensure_dir(cfg["paths"]["processed_dir"])
    rep_dir = ensure_dir(cfg["paths"]["reports_dir"])
    fx = FeatureExtractor(cfg)

    missing = [fp for fp in man["filepath"] if not clip_path(fp).exists()]
    if missing:
        print(f"{len(missing)} manifest files are not on disk, e.g. {missing[0]}")
        print("Run src/data/download_dataset.py and src/data/build_subset.py first.")
        return 1

    waves = np.lib.format.open_memmap(
        out_dir / "waveforms.npy", mode="w+", dtype=np.int16, shape=(n_clips, n_samples)
    )
    feats = np.zeros((n_clips, len(fx.names)), dtype=np.float32)
    lengths = np.zeros(n_clips, dtype=np.int64)
    gains = np.zeros(n_clips, dtype=np.float32)

    rows = list(enumerate(man["filepath"]))
    chunks = [rows[i : i + 256] for i in range(0, n_clips, 256)]
    print(f"{n_clips} clips, {len(fx.names)} clip features, {args.jobs} worker(s)")
    t0 = time.time()
    done = 0
    with Pool(args.jobs, initializer=_init, initargs=(cfg,)) as pool:
        for idx, w, f, ln, g in pool.imap_unordered(_process, chunks):
            waves[idx] = w
            feats[idx] = f
            lengths[idx] = ln
            gains[idx] = g
            done += len(idx)
            if done % (256 * 20) < 256 or done == n_clips:
                print(f"  {done:6d}/{n_clips}  {time.time() - t0:6.0f}s", flush=True)
    waves.flush()
    del waves
    elapsed = time.time() - t0

    np.save(out_dir / "clip_features.npy", feats)
    np.save(out_dir / "clip_lengths.npy", lengths)
    meta = {
        "feature_names": fx.names,
        "n_clips": n_clips,
        "manifest_sha1": manifest_fingerprint(man),
        "limit": args.limit,
    }
    (out_dir / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- what the pipeline did, in numbers -------------------------------
    names = fx.names
    f0 = feats[:, names.index("f0_hz")]
    max_gain = cfg["signal"]["level_norm"]["max_gain_db"]
    non_finite = {n: int(np.sum(~np.isfinite(feats[:, j]))) for j, n in enumerate(names)}
    summary = {
        "n_clips": n_clips,
        "signal": {
            "sample_rate": cfg["signal"]["sample_rate"],
            "n_samples": n_samples,
            "clips_zero_padded": int(np.sum(lengths < n_samples)),
            "clips_cropped": int(np.sum(lengths > n_samples)),
            "shortest_clip_samples": int(lengths.min()),
        },
        "level_normalization": {
            **cfg["signal"]["level_norm"],
            "gain_db_quantiles": {
                q: float(np.quantile(gains, v)) for q, v in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
            },
            "clips_at_gain_cap": int(np.sum(gains >= max_gain - 1e-3)),
            "clips_at_gain_cap_by_label": man.loc[gains >= max_gain - 1e-3, "label"].value_counts().to_dict(),
        },
        "features": {
            **{k: v for k, v in cfg["features"].items() if k != "f0"},
            "f0": cfg["features"]["f0"],
            "frames_per_clip": fx.n_frames(n_samples),
            "clip_vector_length": len(names),
        },
        # The only value that can be undefined is the pitch of a clip with no
        # voiced frame. It is left as NaN here and handled where it is used.
        "undefined_values": {
            "non_finite_by_feature": {k: v for k, v in non_finite.items() if v},
            "f0_undefined_total": int(np.isnan(f0).sum()),
            "f0_undefined_by_label": man.loc[np.isnan(f0), "label"].value_counts().to_dict(),
            "f0_undefined_share_by_label": (
                man.assign(u=np.isnan(f0)).groupby("label")["u"].mean().round(4).to_dict()
            ),
        },
        "runtime_seconds": round(elapsed, 1),
        "workers": args.jobs,
        "limit": args.limit,
    }
    (rep_dir / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(FEATURE_CATALOG).to_csv(rep_dir / "feature_catalog.csv", index=False)

    print(f"done in {elapsed:.0f}s")
    print(f"  zero-padded clips : {summary['signal']['clips_zero_padded']}")
    print(f"  clips at gain cap : {summary['level_normalization']['clips_at_gain_cap']}")
    print(f"  clips without f0  : {summary['undefined_values']['f0_undefined_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
