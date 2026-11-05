#!/usr/bin/env python3
"""
Phase-1 exploratory data analysis (EDA).

Figures produced (all under figures/phase1/):
  fig01_class_distribution.png        — class distribution by split (univariate)
  fig02_samples_per_speaker.png       — histogram + cumulative curve of samples/speaker
  fig03_speakers_vs_budget.png        — "budget vs. speaker count" curve (ties directly to the main question)
  fig04_duration_distribution.png     — file-length distribution
  fig05_rms_by_class.png              — energy (RMS) distribution by class (multivariate)
  fig06_waveforms.png                 — time-domain waveform for a few examples per class
  fig07_logmel.png                    — log-mel spectrogram for the same examples
  fig08_mean_logmel_per_class.png     — mean log-mel per class (spectral difference between words)
  fig09_speaker_mfcc_pca.png          — PCA of speakers' MFCC profiles (groundwork for phase-2 clustering)

All text rendered *inside* these figures (titles, axis labels, legends)
is in English, same as the rest of the codebase and the report.

Usage:
    python src/eda/run_eda.py
    python src/eda/run_eda.py --max-files-per-class 300   # faster on a laptop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import (  # noqa: E402
    REPO_ROOT,
    ensure_dir,
    fix_length,
    load_config,
    read_wav,
    resolve,
    set_seed,
)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Fixed class order for every figure (readability and cross-figure comparability).
def class_order(cfg) -> list[str]:
    return list(cfg["subset"]["keywords"]) + [
        cfg["subset"]["unknown_label"],
        cfg["subset"]["silence_label"],
    ]


# ---------------------------------------------------------------- log-mel
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank — implemented directly so this doesn't hard-depend on librosa."""
    n_bins = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    freqs = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = min(l + 1, n_bins - 1)
        if r == c:
            r = min(c + 1, n_bins - 1)
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    # Slaney normalization (unit area) so high bands aren't over-weighted.
    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    fb *= enorm[:, None]
    return fb


def log_mel(sig: np.ndarray, sr: int, cfg) -> np.ndarray:
    """log-mel spectrogram using the parameters recorded in the config file."""
    e = cfg["eda"]
    n_fft, hop, n_mels = e["n_fft"], e["hop_length"], e["n_mels"]
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(len(sig) - n_fft, 0) // hop
    frames = np.stack([sig[i * hop : i * hop + n_fft] * win for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    fb = mel_filterbank(sr, n_fft, n_mels, e["fmin"], min(e["fmax"], sr // 2))
    mel = spec @ fb.T
    return np.log(mel + 1e-10).T  # (n_mels, n_frames)


def mfcc_from_logmel(lm: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """DCT-II on log-mel to get MFCCs (no librosa dependency)."""
    n_mels = lm.shape[0]
    k = np.arange(n_mels)
    basis = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    basis[0] *= 1.0 / np.sqrt(2)
    return np.sqrt(2.0 / n_mels) * (basis @ lm)


# ---------------------------------------------------------------- figures
def fig_class_distribution(man, order, out):
    piv = (
        man.pivot_table(index="label", columns="split", values="filepath", aggfunc="count")
        .reindex(order)
        .fillna(0)
    )
    piv = piv[[c for c in ("train", "val", "test") if c in piv.columns]]
    ax = piv.plot(kind="barh", stacked=True, figsize=(7, 4.5), width=0.78)
    ax.set_xlabel("Clip count")
    ax.set_ylabel("")
    ax.set_title("Class distribution by split (12-class subset)")
    ax.invert_yaxis()
    ax.legend(title="split", loc="lower right")
    for i, total in enumerate(piv.sum(axis=1)):
        ax.text(total * 1.01, i, f"{int(total):,}", va="center", fontsize=8)
    plt.savefig(out / "fig01_class_distribution.png")
    plt.close()


def fig_samples_per_speaker(man, out):
    w = man[~man["speaker_id"].astype(str).str.startswith("noise::")]
    counts = w[w["split"] == "train"].groupby("speaker_id").size().sort_values()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))

    a.hist(counts, bins=40, color="#4C72B0", edgecolor="white")
    a.axvline(counts.median(), color="#C44E52", ls="--", lw=1.5,
              label=f"median = {counts.median():.0f}")
    a.set_xlabel("Clips recorded by one speaker")
    a.set_ylabel("Number of speakers")
    a.set_title("Samples-per-speaker distribution (train)")
    a.legend()

    cum = np.cumsum(counts.to_numpy()[::-1]) / counts.sum()
    frac_spk = np.arange(1, len(counts) + 1) / len(counts)
    b.plot(frac_spk, cum, lw=2, color="#55A868")
    b.plot([0, 1], [0, 1], ls=":", color="gray", lw=1, label="perfectly uniform")
    b.set_xlabel("Cumulative fraction of speakers (highest-count first)")
    b.set_ylabel("Cumulative fraction of clips")
    b.set_title("Data-concentration curve across speakers")
    b.legend()

    fig.suptitle("Sample imbalance across speakers — the basis of the data-economics question", y=1.02)
    plt.savefig(out / "fig02_samples_per_speaker.png")
    plt.close()
    return counts


def fig_speakers_vs_budget(counts, out):
    """
    The figure most directly tied to the research question: with a fixed
    budget of B clips, how many speakers can be covered if we take at
    most k samples from each?
    """
    fig, ax = plt.subplots(figsize=(7, 4.3))
    arr = counts.to_numpy()
    total = int(arr.sum())
    # Budgets are chosen as a fraction of the available data so this
    # figure stays meaningful on any dataset size (small subset or full
    # dataset).
    budgets = sorted({max(int(total * f), 10) for f in (0.05, 0.10, 0.25, 0.50)})
    k_max = int(min(max(np.percentile(arr, 95), 5), arr.max()))
    ks = np.arange(1, k_max + 1)
    for B in budgets:
        n_spk = []
        for k in ks:
            per = np.minimum(arr, k)          # at most k samples per speaker
            avail = np.sort(per)[::-1]
            cum = np.cumsum(avail)
            n_spk.append(int(np.searchsorted(cum, B) + 1) if cum[-1] >= B else np.nan)
        ax.plot(ks, n_spk, marker="o", ms=3,
                label=f"budget = {B:,} clips ({100 * B / total:.0f}% of train)")
    ax.set_xlabel("Max samples per speaker (k)")
    ax.set_ylabel("Speakers that can be covered")
    ax.set_title("Speaker count vs. samples-per-speaker trade-off at a fixed budget")
    ax.legend()
    plt.savefig(out / "fig03_speakers_vs_budget.png")
    plt.close()


def fig_duration(audit, out):
    if audit is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    d = audit["duration_s"].dropna()
    ax.hist(d, bins=60, color="#8172B2", edgecolor="white")
    ax.axvline(1.0, color="#C44E52", ls="--", lw=1.5, label="target length = 1 s")
    ax.set_yscale("log")
    ax.set_xlabel("Clip length (seconds)")
    ax.set_ylabel("File count (log scale)")
    ax.set_title(
        f"File-length distribution — {100 * (d < 1.0).mean():.1f}% shorter than 1 s (needs padding)"
    )
    ax.legend()
    plt.savefig(out / "fig04_duration_distribution.png")
    plt.close()


def fig_rms_by_class(audit, order, out):
    if audit is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data, labels = [], []
    for lab in order:
        v = audit.loc[(audit["label"] == lab) & (audit["rms"] > 0), "rms"]
        if len(v):
            data.append(np.log10(v))
            labels.append(lab)
    # Compatibility note: the `labels` kwarg on boxplot was deprecated in
    # matplotlib 3.9 and removed in 3.11. Labels are set on the axis
    # directly so this works on either version.
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor("#DD8452" if lab in ("unknown", "silence") else "#4C72B0")
        patch.set_alpha(0.75)
    ax.set_ylabel("log₁₀(RMS)")
    ax.set_xlabel("Class")
    ax.set_title("Signal energy distribution by class — silence should be clearly lower")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.savefig(out / "fig05_rms_by_class.png")
    plt.close()


def pick_examples(man, order, n_per_class, seed):
    rows = []
    for lab in order:
        sub = man[(man["label"] == lab) & (man["split"] == "train")]
        if len(sub) == 0:
            sub = man[man["label"] == lab]
        if len(sub):
            rows.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(rows, ignore_index=True) if rows else man.head(0)


def fig_waveforms_and_logmel(examples, order, cfg, out):
    """Required for route D: plot a waveform and a spectral view for a few examples of each class."""
    sr = cfg["audio"]["target_sample_rate"]
    tlen = int(cfg["audio"]["target_duration_seconds"] * sr)
    n_per = cfg["eda"]["examples_per_class"]
    labs = [l for l in order if l in set(examples["label"])]
    if not labs:
        return None

    cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for lab in labs:
        items = []
        for _, r in examples[examples["label"] == lab].head(n_per).iterrows():
            p = REPO_ROOT / r["filepath"]
            if not p.exists():
                continue
            sig, s = read_wav(p)
            sig = fix_length(sig, tlen)
            items.append((sig, log_mel(sig, s, cfg)))
        cache[lab] = items

    for kind, fname, title in (
        ("wave", "fig06_waveforms.png", "Time-domain waveform per class"),
        ("mel", "fig07_logmel.png", "log-mel spectrogram of the same examples"),
    ):
        fig, axes = plt.subplots(
            len(labs),
            n_per,
            figsize=(2.3 * n_per, 1.45 * len(labs)),
            squeeze=False,
            layout="constrained",
        )
        for i, lab in enumerate(labs):
            for j in range(n_per):
                ax = axes[i][j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if j < len(cache[lab]):
                    sig, lm = cache[lab][j]
                    if kind == "wave":
                        ax.plot(np.arange(len(sig)) / sr, sig, lw=0.45, color="#4C72B0")
                        ax.set_ylim(-1.02, 1.02)
                    else:
                        ax.imshow(lm, origin="lower", aspect="auto", cmap="magma")
                if j == 0:
                    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        e = cfg["eda"]
        sub = (
            "x-axis: time, 0 to 1 second"
            if kind == "wave"
            else f"n_mels={e['n_mels']}, n_fft={e['n_fft']} (25ms), hop={e['hop_length']} (10ms)"
        )
        fig.suptitle(f"{title}\n{sub}", fontsize=10)
        plt.savefig(out / fname)
        plt.close()
    return cache


def fig_mean_logmel(cache, cfg, man, out):
    """Mean log-mel per class over a larger sample — shows the spectral difference between words."""
    if not cache:
        return
    sr = cfg["audio"]["target_sample_rate"]
    tlen = int(cfg["audio"]["target_duration_seconds"] * sr)
    order = class_order(cfg)
    labs = [l for l in order if l in set(man["label"])]
    means = {}
    for lab in labs:
        sub = man[(man["label"] == lab) & (man["split"] == "train")].head(60)
        acc, n = None, 0
        for _, r in sub.iterrows():
            p = REPO_ROOT / r["filepath"]
            if not p.exists():
                continue
            sig, s = read_wav(p)
            lm = log_mel(fix_length(sig, tlen), s, cfg)
            acc = lm if acc is None else acc + lm
            n += 1
        if n:
            means[lab] = acc / n
    if not means:
        return
    cols = 4
    rows = int(np.ceil(len(means) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.0 * rows), squeeze=False)
    vmin = min(m.min() for m in means.values())
    vmax = max(m.max() for m in means.values())
    for idx, (lab, m) in enumerate(means.items()):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(m, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for idx in range(len(means), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("Mean log-mel per class (y-axis: mel band, x-axis: time)", y=1.01)
    plt.savefig(out / "fig08_mean_logmel_per_class.png")
    plt.close()


def fig_speaker_pca(man, cfg, out, max_speakers=250, clips_per_speaker=8):
    """
    Per-speaker MFCC profile -> 2D PCA.
    This figure is direct groundwork for phase-2 speaker clustering; in
    phase 1 it's only used to check whether any visible structure exists
    across speakers.
    """
    sr = cfg["audio"]["target_sample_rate"]
    tlen = int(cfg["audio"]["target_duration_seconds"] * sr)
    w = man[(man["split"] == "train") & (~man["speaker_id"].astype(str).str.startswith("noise::"))]
    spk_counts = w.groupby("speaker_id").size().sort_values(ascending=False)
    speakers = spk_counts.head(max_speakers).index.tolist()

    vecs, names, nclips = [], [], []
    for i, spk in enumerate(speakers, 1):
        sub = w[w["speaker_id"] == spk].head(clips_per_speaker)
        feats = []
        for _, r in sub.iterrows():
            p = REPO_ROOT / r["filepath"]
            if not p.exists():
                continue
            sig, s = read_wav(p)
            lm = log_mel(fix_length(sig, tlen), s, cfg)
            mf = mfcc_from_logmel(lm, 13)
            feats.append(np.concatenate([mf.mean(axis=1), mf.std(axis=1)]))
        if feats:
            vecs.append(np.mean(feats, axis=0))
            names.append(spk)
            nclips.append(len(sub))
        if i % 25 == 0:
            print(f"\r[eda] speaker profile {i}/{len(speakers)}", end="", flush=True)
    print()
    if len(vecs) < 5:
        return
    X = np.array(vecs)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # Manual PCA via SVD — no scikit-learn dependency needed here.
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = U[:, :2] * S[:2]
    var = (S**2) / (S**2).sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=nclips, cmap="viridis", s=26, alpha=0.85,
                    edgecolor="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="Clips used to build the profile")
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}% variance)")
    ax.set_title(
        f"Speaker audio-profile space ({len(names)} train speakers)\n"
        "Feature: mean and std of 13 MFCC coefficients"
    )
    plt.savefig(out / "fig09_speaker_mfcc_pca.png")
    plt.close()

    np.savez(
        ensure_dir(cfg["paths"]["reports_dir"]) / "speaker_profiles_phase1.npz",
        speakers=np.array(names),
        features=X,
        pca2=Z,
        explained_variance_ratio=var[:10],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-1 EDA")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-files-per-class", type=int, default=0, help="limit for a fast run")
    ap.add_argument("--skip-speaker-pca", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out = ensure_dir(cfg["paths"]["figures_dir"])
    rep = resolve(cfg["paths"]["reports_dir"])

    man_path = resolve(cfg["paths"]["manifest"])
    if not man_path.exists():
        print(f"[error] manifest not found: {man_path}", file=sys.stderr)
        print("        run first: python src/data/build_subset.py", file=sys.stderr)
        return 1
    man = pd.read_csv(man_path)
    if args.max_files_per_class:
        man = man.groupby(["label", "split"], group_keys=False).head(args.max_files_per_class)

    audit_path = rep / "audit_per_file.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    if audit is None:
        print("[warn] audit_per_file.csv not found; figures 4 and 5 will be skipped. "
              "Run audit_quality.py first to generate it.")

    order = class_order(cfg)

    print("[eda] 1/7 class distribution ...")
    fig_class_distribution(man, order, out)
    print("[eda] 2/7 samples per speaker ...")
    counts = fig_samples_per_speaker(man, out)
    print("[eda] 3/7 budget/speaker trade-off ...")
    fig_speakers_vs_budget(counts, out)
    print("[eda] 4/7 duration distribution ...")
    fig_duration(audit, out)
    print("[eda] 5/7 energy by class ...")
    fig_rms_by_class(audit, order, out)
    print("[eda] 6/7 waveform and log-mel ...")
    ex = pick_examples(man, order, cfg["eda"]["examples_per_class"], cfg["seed"])
    cache = fig_waveforms_and_logmel(ex, order, cfg, out)
    fig_mean_logmel(cache, cfg, man, out)
    if not args.skip_speaker_pca:
        print("[eda] 7/7 speaker-profile PCA ...")
        fig_speaker_pca(man, cfg, out)

    meta = {
        "n_rows_used": int(len(man)),
        "class_order": order,
        "logmel_params": cfg["eda"],
        "example_files": ex[["filepath", "label"]].to_dict(orient="records"),
        "figures": sorted(p.name for p in out.glob("*.png")),
    }
    (ensure_dir(cfg["paths"]["reports_dir"]) / "eda_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] {len(meta['figures'])} figures in {out}")
    for f in meta["figures"]:
        print(f"   - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
