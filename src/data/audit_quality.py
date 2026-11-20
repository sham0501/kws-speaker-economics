#!/usr/bin/env python3
"""
Audio quality audit and speaker audit for phase 1.

Covers four checks on the subset manifest:
  * audio-file audit — sample rate, duration, sample count per class,
    signal amplitude, corrupt or very short files
  * speaker audit — samples per speaker, broken down by class
  * noise and silence check — background-noise and low-energy clips, and
    how they enter the classification problem
  * table-level audit — missing values, duplicate rows, impossible values

Outputs (all in reports/phase1/):
  audit_per_file.csv        — one row per file with signal metrics
  audit_flags.csv           — only flagged files, with the reason
  audit_per_speaker.csv     — per-speaker sample counts by class
  audit_summary.json        — numeric summary for the write-up

Usage:
    python src/data/audit_quality.py
    python src/data/audit_quality.py --limit 2000    # quick audit on a random sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import clip_path, ensure_dir, load_config, resolve, set_seed  # noqa: E402


def probe(path: Path) -> dict:
    """
    Read one file and compute quality metrics.
    Any error is recorded as corrupt=True instead of crashing (required:
    identify corrupt files).
    """
    out = {
        "sample_rate": np.nan,
        "n_samples": np.nan,
        "duration_s": np.nan,
        "peak_abs": np.nan,
        "rms": np.nan,
        "dc_offset": np.nan,
        "clipped_ratio": np.nan,
        "is_all_zero": False,
        "corrupt": False,
        "error": "",
    }
    try:
        from utils.common import read_wav

        sig, sr = read_wav(path)
        n = int(sig.shape[0])
        out.update(
            sample_rate=sr,
            n_samples=n,
            duration_s=n / sr if sr else np.nan,
        )
        if n == 0:
            out["is_all_zero"] = True
            return out
        absx = np.abs(sig)
        peak = float(absx.max())
        out.update(
            peak_abs=peak,
            rms=float(np.sqrt(np.mean(sig.astype(np.float64) ** 2))),
            dc_offset=float(np.mean(sig)),
            # Fraction of samples pinned to near-max amplitude — a sign
            # of clipping during recording.
            clipped_ratio=float(np.mean(absx >= 0.999)) if peak >= 0.999 else 0.0,
            is_all_zero=bool(peak == 0.0),
        )
    except Exception as e:  # noqa: BLE001
        out["corrupt"] = True
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio quality audit and speaker audit")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only N random files (0 = all); written to reports/phase1/limited/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    acfg = cfg["audit"]

    man_path = resolve(cfg["paths"]["manifest"])
    if not man_path.exists():
        print(f"[error] manifest not found: {man_path}", file=sys.stderr)
        print("        run first: python src/data/build_subset.py", file=sys.stderr)
        return 1

    man = pd.read_csv(man_path)
    print(f"[info] {len(man)} rows in manifest.")

    # ---------- 1) table-level audit (before opening any files) ----------
    table_issues = {
        "missing_values_by_column": man.isna().sum().to_dict(),
        "duplicate_filepath_rows": int(man["filepath"].duplicated().sum()),
        "duplicate_full_rows": int(man.duplicated().sum()),
        "labels_present": sorted(man["label"].unique().tolist()),
        "splits_present": sorted(man["split"].unique().tolist()),
        "n_rows_with_missing_speaker": int(man["speaker_id"].isna().sum()),
    }
    # "Impossible" value: negative utterance_index
    bad_utt = man[pd.to_numeric(man["utterance_index"], errors="coerce").fillna(-1) < 0]
    table_issues["rows_with_invalid_utterance_index"] = int(len(bad_utt))

    # ---------- 2) file-level audit ----------
    work = man.sample(n=min(args.limit, len(man)), random_state=cfg["seed"]) if args.limit else man
    print(f"[info] auditing {len(work)} files ... (this takes a few minutes)")

    recs = []
    for i, (_, row) in enumerate(work.iterrows(), 1):
        p = clip_path(row["filepath"])
        if not p.exists():
            rec = {"corrupt": True, "error": "FileNotFound", "sample_rate": np.nan}
        else:
            rec = probe(p)
        rec.update(
            filepath=row["filepath"],
            label=row["label"],
            word=row["word"],
            speaker_id=row["speaker_id"],
            split=row["split"],
        )
        recs.append(rec)
        if i % 1000 == 0 or i == len(work):
            print(f"\r[audit] {i}/{len(work)}", end="", flush=True)
    print()

    df = pd.DataFrame(recs)

    # ---------- 3) flagging ----------
    sr_target = cfg["audio"]["target_sample_rate"]
    df["flag_corrupt"] = df["corrupt"]
    df["flag_wrong_sample_rate"] = df["sample_rate"].notna() & (df["sample_rate"] != sr_target)
    df["flag_too_short"] = df["duration_s"] < acfg["min_duration_seconds"]
    df["flag_low_energy"] = df["rms"] < acfg["low_energy_rms_threshold"]
    df["flag_all_zero"] = df["is_all_zero"]
    df["flag_clipping"] = df["clipped_ratio"] > acfg["clipping_ratio_threshold"]
    # Loudness outlier: outside 1.5x IQR on log10(RMS).
    valid_rms = df.loc[df["rms"] > 0, "rms"]
    if len(valid_rms) > 10:
        lr = np.log10(valid_rms)
        q1, q3 = lr.quantile(0.25), lr.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df["flag_loudness_outlier"] = (np.log10(df["rms"].where(df["rms"] > 0)) < lo) | (
            np.log10(df["rms"].where(df["rms"] > 0)) > hi
        )
        df["flag_loudness_outlier"] = df["flag_loudness_outlier"].fillna(False)
        loudness_bounds = {"log10_rms_low": float(lo), "log10_rms_high": float(hi)}
    else:
        df["flag_loudness_outlier"] = False
        loudness_bounds = {}

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["n_flags"] = df[flag_cols].sum(axis=1)

    # ---------- 4) speaker audit ----------
    words = df[~df["speaker_id"].astype(str).str.startswith("noise::")]
    per_speaker = (
        words.groupby(["speaker_id", "split"])
        .agg(
            n_samples=("filepath", "count"),
            n_distinct_labels=("label", "nunique"),
            n_distinct_words=("word", "nunique"),
            mean_rms=("rms", "mean"),
            mean_duration_s=("duration_s", "mean"),
        )
        .reset_index()
    )
    # Speaker x class cross-tab: how many samples each speaker has in which class.
    spk_by_label = (
        words.pivot_table(
            index="speaker_id", columns="label", values="filepath", aggfunc="count", fill_value=0
        )
        .reset_index()
    )
    per_speaker = per_speaker.merge(spk_by_label, on="speaker_id", how="left")

    counts = per_speaker.groupby("split")["n_samples"]
    speaker_stats = {
        s: {
            "n_speakers": int(g.count()),
            "samples_per_speaker_min": int(g.min()),
            "samples_per_speaker_p25": float(g.quantile(0.25)),
            "samples_per_speaker_median": float(g.median()),
            "samples_per_speaker_p75": float(g.quantile(0.75)),
            "samples_per_speaker_max": int(g.max()),
            "samples_per_speaker_mean": float(g.mean()),
            "gini": float(_gini(g.to_numpy())),
        }
        for s, g in counts
    }

    # ---------- 5) write output ----------
    # A limited run is a smoke test; it must never overwrite the full audit
    # that the report and later phases read.
    rep = ensure_dir(resolve(cfg["paths"]["reports_dir"]) / "limited" if args.limit else cfg["paths"]["reports_dir"])
    df.to_csv(rep / "audit_per_file.csv", index=False)
    flagged = df[df["n_flags"] > 0].sort_values("n_flags", ascending=False)
    flagged.to_csv(rep / "audit_flags.csv", index=False)
    per_speaker.to_csv(rep / "audit_per_speaker.csv", index=False)

    summary = {
        "n_files_audited": int(len(df)),
        "audited_fraction": float(len(df) / len(man)),
        "table_level_issues": table_issues,
        "sample_rate_counts": df["sample_rate"].value_counts(dropna=False).to_dict(),
        "duration_seconds": {
            "min": _f(df["duration_s"].min()),
            "p01": _f(df["duration_s"].quantile(0.01)),
            "median": _f(df["duration_s"].median()),
            "p99": _f(df["duration_s"].quantile(0.99)),
            "max": _f(df["duration_s"].max()),
            "fraction_exactly_1s": _f((df["n_samples"] == sr_target).mean()),
            "fraction_shorter_than_1s": _f((df["n_samples"] < sr_target).mean()),
        },
        "amplitude": {
            "peak_abs_median": _f(df["peak_abs"].median()),
            "peak_abs_p99": _f(df["peak_abs"].quantile(0.99)),
            "rms_median": _f(df["rms"].median()),
            "rms_p01": _f(df["rms"].quantile(0.01)),
            "dc_offset_abs_max": _f(df["dc_offset"].abs().max()),
            "loudness_outlier_bounds_log10_rms": loudness_bounds,
        },
        "flags": {c: int(df[c].sum()) for c in flag_cols},
        "flagged_files_total": int((df["n_flags"] > 0).sum()),
        "counts_by_label": df["label"].value_counts().to_dict(),
        "counts_by_split_label": {
            s: g["label"].value_counts().to_dict() for s, g in df.groupby("split")
        },
        "low_energy_by_label": df.groupby("label")["flag_low_energy"].sum().astype(int).to_dict(),
        "speaker_stats_by_split": speaker_stats,
        "thresholds_used": acfg,
    }
    (rep / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- 6) print summary ----------
    print()
    print("=== audit summary ===")
    print(f"files audited: {len(df)}")
    print(f"sample rate: {df['sample_rate'].value_counts(dropna=False).to_dict()}")
    print(
        f"duration: median {summary['duration_seconds']['median']}s, "
        f"{100 * (summary['duration_seconds']['fraction_shorter_than_1s'] or 0):.1f}% shorter than 1s"
    )
    for c in flag_cols:
        n = int(df[c].sum())
        if n:
            print(f"  {c}: {n} ({100 * n / len(df):.2f}%)")
    print(f"total flagged files: {int((df['n_flags'] > 0).sum())}")
    print()
    print("speakers:")
    for s, st in speaker_stats.items():
        print(
            f"  {s:5s}  {st['n_speakers']:4d} speakers  |  samples/speaker: "
            f"min={st['samples_per_speaker_min']} median={st['samples_per_speaker_median']:.0f} "
            f"max={st['samples_per_speaker_max']}  Gini={st['gini']:.3f}"
        )
    print()
    print(f"outputs in: {rep}")
    return 0


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of the per-speaker sample-count distribution — an imbalance metric."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _f(v):
    """Convert numpy/NaN values into something json can serialize."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())