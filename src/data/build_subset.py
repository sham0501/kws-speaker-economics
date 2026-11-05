#!/usr/bin/env python3
"""
Build an explainable subset of Speech Commands v0.02 with a
speaker-independent split.

Output: data/subset_manifest.csv with columns
    filepath, label, speaker_id, utterance_index, split, source
and a report at reports/phase1/subset_summary.json

Classes (12 total):
    the 10 classic benchmark keywords + unknown + silence

Split logic (the key design decision in this script):
    Google's official `validation_list.txt` and `testing_list.txt` are
    used directly. These files are built from the `which_set()` hash
    function in TensorFlow's reference code, which hashes the speaker_id
    portion of the filename (before `_nohash_`). The result is that
    **all of a given speaker's samples always land in the same set**, so
    no speaker leakage occurs. This script only filters those lists down
    to the 12 subset classes — it does not invent a new split, so the
    official split stays intact.

    The same rule applies to the unknown class (non-keyword words are
    also in the official lists). The silence class has no ready-made
    files, so its split is done **at the level of the source noise
    file**: each `_background_noise_` file is assigned wholly to one of
    train/val/test, so that clips cut from the same noise file never end
    up split across sets (which would be a noise leak).

Usage:
    python src/data/build_subset.py
    python src/data/build_subset.py --dry-run     # stats only, no silence files written
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import (  # noqa: E402
    REPO_ROOT,
    ensure_dir,
    fix_length,
    load_config,
    parse_speaker_id,
    parse_utterance_index,
    read_wav,
    relpath_key,
    resolve,
    set_seed,
)

BG_DIR = "_background_noise_"


def load_official_split(root: Path) -> dict[str, str]:
    """
    Read Google's official validation_list.txt and testing_list.txt.
    Returns: a mapping "<label>/<file>.wav" -> "val" | "test"
    Any file not in either list is train (the official convention).
    """
    mapping: dict[str, str] = {}
    for fname, split in (("validation_list.txt", "val"), ("testing_list.txt", "test")):
        p = root / fname
        if not p.exists():
            raise FileNotFoundError(
                f"official split file not found: {p}\n"
                "Run src/data/download_dataset.py first."
            )
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                mapping[line] = split
    return mapping


def collect_rows(root: Path, keywords: list[str], split_map: dict[str, str]) -> pd.DataFrame:
    """Walk every word folder and build a raw DataFrame of the whole dataset (excluding background noise)."""
    rows = []
    bad_names = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        if label == BG_DIR:
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            spk = parse_speaker_id(wav.name)
            utt = parse_utterance_index(wav.name)
            if spk is None:
                bad_names.append(f"{label}/{wav.name}")
                continue
            key = relpath_key(label, wav.name)
            rows.append(
                {
                    "filepath": str(wav.relative_to(REPO_ROOT)),
                    "word": label,
                    "speaker_id": spk,
                    "utterance_index": utt,
                    "split": split_map.get(key, "train"),
                }
            )
    if bad_names:
        print(
            f"[warn] {len(bad_names)} files with names not matching the official "
            f"convention were dropped (example: {bad_names[:3]})"
        )
    return pd.DataFrame(rows)


def build_silence(
    root: Path,
    out_dir: Path,
    n_per_split: dict[str, int],
    clip_seconds: float,
    sr_target: int,
    rng: np.random.Generator,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    """
    Build silence clips from the _background_noise_ files.

    Recorded decision (for the write-up):
      - The silence class has no ready-made files in the dataset; the
        official TensorFlow benchmark builds it from the same noise
        files, so this is consistent with the reference setup.
      - Noise files are split **at the file level**, not the clip level.
        Reason: if two clips from the same noise file ended up in train
        and test, the model could memorize that specific noise pattern
        and test accuracy would be inflated (a form of leakage).
      - Clips are non-overlapping with a random offset.
    """
    bg = root / BG_DIR
    noise_files = sorted(bg.glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"no noise files found in {bg}")

    # Assign noise files to splits: most go to train, one to val, one to test.
    files = list(noise_files)
    rng.shuffle(files)
    if len(files) < 3:
        raise RuntimeError(f"need at least 3 noise files for a file-level split, found {len(files)}.")
    assign = {"val": [files[0]], "test": [files[1]], "train": files[2:]}

    clip_len = int(round(clip_seconds * sr_target))
    rows: list[dict] = []
    provenance: dict = {k: [f.name for f in v] for k, v in assign.items()}

    if not dry_run:
        ensure_dir(out_dir)

    for split, flist in assign.items():
        want = n_per_split[split]
        if want <= 0:
            continue
        # Distribute the clip count across that split's noise files.
        per_file = [want // len(flist)] * len(flist)
        for i in range(want % len(flist)):
            per_file[i] += 1

        for nf, count in zip(flist, per_file):
            if count == 0:
                continue
            sig, sr = read_wav(nf)
            if sr != sr_target:
                raise RuntimeError(f"unexpected sample rate in {nf.name}: {sr}")
            max_start = max(len(sig) - clip_len, 0)
            if max_start == 0:
                starts = np.zeros(count, dtype=int)
            else:
                starts = rng.choice(max_start, size=count, replace=count > max_start)
            for j, st in enumerate(starts):
                clip = fix_length(sig[int(st) : int(st) + clip_len], clip_len)
                name = f"{nf.stem}_{split}_{j:05d}.wav"
                dest = out_dir / name
                if not dry_run:
                    import soundfile as sf

                    sf.write(str(dest), clip, sr_target, subtype="PCM_16")
                rows.append(
                    {
                        "filepath": str(dest.relative_to(REPO_ROOT)),
                        "word": "_silence_",
                        # There is no "speaker" for silence clips; the
                        # source noise filename is used as the grouping
                        # key so it isn't confused with a real speaker.
                        "speaker_id": f"noise::{nf.stem}",
                        "utterance_index": j,
                        "split": split,
                    }
                )
    return rows, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the subset with a speaker-independent split")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="stats only; don't write silence files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    root = resolve(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["extract_dirname"]
    if not root.exists():
        print(f"[error] dataset folder not found: {root}", file=sys.stderr)
        print("        run first: python src/data/download_dataset.py", file=sys.stderr)
        return 1

    keywords = cfg["subset"]["keywords"]
    unk_label = cfg["subset"]["unknown_label"]
    sil_label = cfg["subset"]["silence_label"]

    print("[1/5] reading Google's official split ...")
    split_map = load_official_split(root)
    print(f"      {len(split_map)} files are listed in the official val/test lists.")

    print("[2/5] scanning the full dataset ...")
    full = collect_rows(root, keywords, split_map)
    print(f"      {len(full)} files across {full['word'].nunique()} words.")

    # ---- keywords ----
    kw = full[full["word"].isin(keywords)].copy()
    kw["label"] = kw["word"]
    kw["source"] = "keyword"

    # ---- unknown: balanced sampling from non-keyword words ----
    # Recorded decision: the goal is for unknown's size to be close to
    # the *average size of one keyword class*, so it doesn't dominate.
    # Sampling is done separately within each split, uniformly across
    # non-keyword words, so no single word dominates unknown.
    other = full[~full["word"].isin(keywords)].copy()
    unk_rows = []
    unk_target: dict[str, int] = {}
    for split in ("train", "val", "test"):
        kw_split = kw[kw["split"] == split]
        target = int(round(len(kw_split) / len(keywords) * cfg["subset"]["unknown_target_ratio"]))
        unk_target[split] = target
        pool = other[other["split"] == split]
        if len(pool) == 0:
            continue
        words = sorted(pool["word"].unique())
        per_word = [target // len(words)] * len(words)
        for i in range(target % len(words)):
            per_word[i] += 1
        for w, n in zip(words, per_word):
            sub = pool[pool["word"] == w]
            n = min(n, len(sub))
            if n > 0:
                unk_rows.append(sub.sample(n=n, random_state=cfg["seed"]))
    unk = pd.concat(unk_rows, ignore_index=True) if unk_rows else other.head(0).copy()
    unk["label"] = unk_label
    unk["source"] = "unknown_sampled"

    # ---- silence ----
    sil_target = {}
    for split in ("train", "val", "test"):
        kw_split = kw[kw["split"] == split]
        sil_target[split] = int(
            round(len(kw_split) / len(keywords) * cfg["subset"]["silence_target_ratio"])
        )
    print(f"[3/5] building silence clips (target: {sil_target}) ...")
    sil_dir = resolve(cfg["dataset"]["raw_dir"]) / "silence_clips"
    sil_rows, sil_prov = build_silence(
        root,
        sil_dir,
        sil_target,
        cfg["subset"]["silence_clip_seconds"],
        cfg["audio"]["target_sample_rate"],
        rng,
        dry_run=args.dry_run,
    )
    sil = pd.DataFrame(sil_rows)
    if len(sil):
        sil["label"] = sil_label
        sil["source"] = "silence_generated"

    # ---- combine ----
    cols = ["filepath", "label", "word", "speaker_id", "utterance_index", "split", "source"]
    parts = [kw[cols], unk[cols]] + ([sil[cols]] if len(sil) else [])
    manifest = pd.concat(parts, ignore_index=True)
    manifest = manifest.sort_values(["split", "label", "speaker_id", "utterance_index"]).reset_index(
        drop=True
    )

    # ---- speaker-leakage verification (required for the write-up) ----
    print("[4/5] verifying speaker independence ...")
    spk_by_split = {
        s: set(manifest.loc[(manifest["split"] == s) & (~manifest["speaker_id"].str.startswith("noise::")), "speaker_id"])
        for s in ("train", "val", "test")
    }
    overlaps = {
        "train∩val": sorted(spk_by_split["train"] & spk_by_split["val"]),
        "train∩test": sorted(spk_by_split["train"] & spk_by_split["test"]),
        "val∩test": sorted(spk_by_split["val"] & spk_by_split["test"]),
    }
    leak = {k: v for k, v in overlaps.items() if v}
    if leak:
        print("      [FAIL] speaker leakage found:", file=sys.stderr)
        for k, v in leak.items():
            print(f"             {k}: {len(v)} speakers (example {v[:5]})", file=sys.stderr)
    else:
        print("      [OK] no speaker is shared across train/val/test.")

    # Noise-file leakage verification.
    noise_by_split = defaultdict(set)
    for _, r in manifest[manifest["label"] == sil_label].iterrows():
        noise_by_split[r["split"]].add(r["speaker_id"])
    noise_leak = {
        "train∩val": sorted(noise_by_split["train"] & noise_by_split["val"]),
        "train∩test": sorted(noise_by_split["train"] & noise_by_split["test"]),
        "val∩test": sorted(noise_by_split["val"] & noise_by_split["test"]),
    }
    noise_leak = {k: v for k, v in noise_leak.items() if v}
    print(
        "      [OK] no noise file is shared across sets."
        if not noise_leak
        else f"      [FAIL] noise file leakage: {noise_leak}"
    )

    # ---- write output ----
    print("[5/5] writing manifest and summary ...")
    out_csv = resolve(cfg["paths"]["manifest"])
    ensure_dir(out_csv.parent)
    manifest.to_csv(out_csv, index=False)

    summary = {
        "seed": cfg["seed"],
        "n_total": int(len(manifest)),
        "n_classes": int(manifest["label"].nunique()),
        "keywords": keywords,
        "counts_by_split": manifest["split"].value_counts().to_dict(),
        "counts_by_label": manifest["label"].value_counts().to_dict(),
        "counts_by_split_label": {
            s: g["label"].value_counts().to_dict() for s, g in manifest.groupby("split")
        },
        "n_speakers_by_split": {k: len(v) for k, v in spk_by_split.items()},
        "speaker_overlap": overlaps,
        "speaker_leakage_detected": bool(leak),
        "unknown_source_words": sorted(unk["word"].unique().tolist()) if len(unk) else [],
        "unknown_target_per_split": unk_target,
        "silence_target_per_split": sil_target,
        "silence_noise_file_assignment": sil_prov,
        "silence_leakage_detected": bool(noise_leak),
        "split_policy": (
            "Google's official validation_list.txt/testing_list.txt (hashed on "
            "speaker_id), which guarantees each speaker appears in only one set; "
            "silence is split at the level of the source noise file."
        ),
    }
    out_json = ensure_dir(cfg["paths"]["reports_dir"]) / "subset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"manifest: {out_csv}  ({len(manifest)} rows)")
    print(f"summary:  {out_json}")
    print()
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print()
    print("unique speaker counts:", {k: len(v) for k, v in spk_by_split.items()})
    return 1 if (leak or noise_leak) else 0


if __name__ == "__main__":
    raise SystemExit(main())
