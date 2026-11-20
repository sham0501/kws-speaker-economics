#!/usr/bin/env python3
"""
Where each silence clip comes from, and how much the clips overlap.

The silence class is cut from the six background recordings of the dataset.
Each recording is about a minute long, so it holds only about sixty
disjoint one-second windows, while a split may take hundreds of clips from
it. Clips from one recording therefore share audio, and the effective
number of independent silence examples is much smaller than the clip count.

This script measures that on the subset as it is on disk, without
rebuilding it: every silence clip is located in its source recording by
exact sample match (the clips are verbatim copies of 16-bit source
samples), which gives its start sample. If build_subset.py already wrote
the start samples, they are checked against the match.

Per source recording it reports
    duration                 length of the recording in seconds
    clips                    silence clips cut from it
    distinct_seconds         audio actually covered by at least one clip
    clip_seconds_per_distinct_second
                             how many times, on average, each covered second
                             appears in the class (1.0 = no overlap)
    clips_overlapping_another
                             share of clips that share samples with another
    max_clips_on_one_sample  the most clips any single sample appears in

Outputs
    reports/phase1/silence_provenance.csv   filepath, split, source_file, start_sample, ...
    reports/phase1/silence_overlap.json
    figures/phase1/fig10_silence_coverage.png

Usage
    python src/data/silence_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import clip_path, ensure_dir, load_config, resolve  # noqa: E402

BG_DIR = "_background_noise_"
PROBE = 64  # samples compared before a full-window check


def locate(clip: np.ndarray, source: np.ndarray, index: dict) -> int | None:
    """Start sample of `clip` inside `source`, or None if it is not a verbatim excerpt."""
    n = len(clip)
    for start in index.get(int(clip[0]), ()):
        if start + n <= len(source) and np.array_equal(source[start : start + PROBE], clip[:PROBE]):
            if np.array_equal(source[start : start + n], clip):
                return int(start)
    return None


def coverage(starts: np.ndarray, clip_len: int, n_samples: int) -> np.ndarray:
    """Number of clips covering each sample of the source recording."""
    diff = np.zeros(n_samples + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, np.minimum(starts + clip_len, n_samples), -1)
    return np.cumsum(diff)[:n_samples]


def main() -> int:
    cfg = load_config()
    sr = int(cfg["audio"]["target_sample_rate"])
    clip_len = int(round(cfg["subset"]["silence_clip_seconds"] * sr))
    rep = ensure_dir(cfg["paths"]["reports_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    bg = resolve(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["extract_dirname"] / BG_DIR

    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    sil = man[man["label"] == cfg["subset"]["silence_label"]].copy()
    sil["source_file"] = sil["speaker_id"].str.replace("noise::", "", regex=False) + ".wav"
    recorded = rep / "silence_provenance.csv"
    previous = pd.read_csv(recorded).set_index("filepath")["start_sample"] if recorded.exists() else None

    rows, summary = [], {}
    for name, group in sil.groupby("source_file"):
        source, file_sr = sf.read(bg / name, dtype="int16")
        if file_sr != sr:
            raise ValueError(f"{name}: sample rate {file_sr}")
        order = np.argsort(source, kind="stable")
        values, first = np.unique(source[order], return_index=True)
        bounds = np.append(first, len(order))
        index = {int(v): order[bounds[i] : bounds[i + 1]] for i, v in enumerate(values)}
        starts = []
        for fp, split in zip(group["filepath"], group["split"]):
            clip, _ = sf.read(clip_path(fp), dtype="int16")
            st = locate(clip, source, index)
            starts.append(st)
            rows.append({"filepath": fp, "split": split, "source_file": name, "start_sample": st, "source_samples": len(source)})
        found = np.array([s for s in starts if s is not None], dtype=np.int64)
        cov = coverage(found, clip_len, len(source))
        covered = int((cov > 0).sum())
        srt = np.sort(found)
        gaps = np.diff(srt)
        overlapping = np.zeros(len(srt), dtype=bool)
        overlapping[:-1] |= gaps < clip_len
        overlapping[1:] |= gaps < clip_len
        summary[name] = {
            "split": sorted(group["split"].unique()),
            "duration_s": round(len(source) / sr, 2),
            "clips": int(len(group)),
            "clips_located": int(len(found)),
            "distinct_seconds": round(covered / sr, 2),
            "share_of_recording_used": round(covered / len(source), 4),
            "clip_seconds_per_distinct_second": round(len(found) * clip_len / max(covered, 1), 2),
            "clips_overlapping_another": round(float(overlapping.mean()), 4) if len(srt) else None,
            "max_clips_on_one_sample": int(cov.max()) if len(cov) else 0,
            "disjoint_one_second_windows_in_recording": int(len(source) // clip_len),
        }
        print(f"{name:24s} {summary[name]['split']}  {len(group):4d} clips from {len(source) / sr:5.1f} s"
              f"  -> each covered second appears {summary[name]['clip_seconds_per_distinct_second']:.1f}x")

    prov = pd.DataFrame(rows)
    prov["start_sample"] = prov["start_sample"].astype("Int64")
    if previous is not None:
        both = prov.set_index("filepath")["start_sample"].reindex(previous.index)
        mismatch = int((both.astype("float") != previous.astype("float")).sum())
        print(f"start samples recorded by build_subset.py: {len(previous)}, disagreeing with the audio: {mismatch}")
    prov.to_csv(recorded, index=False)

    by_split = (
        prov.groupby("split")
        .agg(clips=("filepath", "size"), source_recordings=("source_file", "nunique"))
        .to_dict(orient="index")
    )
    out = {
        "clip_seconds": cfg["subset"]["silence_clip_seconds"],
        "clips_not_located": int(prov["start_sample"].isna().sum()),
        "by_split": by_split,
        "by_source_recording": summary,
    }
    (rep / "silence_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (name, g) in enumerate(prov.groupby("source_file")):
        st = g["start_sample"].dropna().to_numpy(dtype=np.int64)
        cov = coverage(st, clip_len, int(g["source_samples"].iat[0]))
        t = np.arange(len(cov)) / sr
        ax.plot(t[::160], cov[::160], lw=1, label=f"{name} ({g['split'].iat[0]}, {len(g)} clips)", color=plt.get_cmap("tab10")(i))
    ax.set_xlabel("time in the source recording (s)")
    ax.set_ylabel("silence clips containing this sample")
    ax.set_title("How often each part of a background recording is reused in the silence class")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(fig_dir / "fig10_silence_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"clips not located: {out['clips_not_located']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
