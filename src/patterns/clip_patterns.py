#!/usr/bin/env python3
"""
What structure do the clip features expose before any label is used?

Three views of the training clips:

1. PCA of the standardized clip vector: how many directions carry the
   variance, and what the first two look like.
2. t-SNE of the same data, for a non-linear picture of local neighborhoods.
3. k-means partitions compared against three things a cluster could
   correspond to: the word, the speaker, and (for silence clips) the
   background recording the clip was cut from.

The scaler, the PCA and the word means are all fitted on training clips.

Outputs
    reports/phase2/clip_structure.json
    figures/phase2/fig03_clip_pca_variance.png
    figures/phase2/fig04_clip_embedding.png
    figures/phase2/fig05_clip_structure.png

Usage
    python src/patterns/clip_patterns.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_features import column_index, feature_sets  # noqa: E402
from features.speaker_profiles import word_residuals  # noqa: E402
from features.store import class_order, load_features  # noqa: E402
from utils.common import ensure_dir, load_config  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def reduce(x: np.ndarray, variance: float, seed: int) -> tuple[np.ndarray, PCA]:
    """Standardize, then keep the leading components that reach `variance`."""
    z = StandardScaler().fit_transform(x)
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed).fit(z)
    return pca.transform(z), pca


def partition_agreement(x: np.ndarray, target: np.ndarray, seed: int, variance: float) -> dict:
    """k-means with as many clusters as `target` has values, scored against `target`."""
    k = len(np.unique(target))
    z, _ = reduce(x, variance, seed)
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)
    return {
        "k": int(k),
        "nmi": float(normalized_mutual_info_score(target, pred)),
        "ari": float(adjusted_rand_score(target, pred)),
    }


def main() -> int:
    cfg = load_config("configs/phase2.yaml")
    cp, seed = cfg["clip_patterns"], cfg["seed"]
    variance = cfg["clustering"]["pca_variance"]
    rng = np.random.default_rng(seed)
    rep_dir = ensure_dir(cfg["paths"]["reports_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

    man, feats, names = load_features(cfg)
    order = class_order(cfg)
    cols = column_index(names, feature_sets(names)["classifier"])
    x_all = feats[:, cols].astype(np.float64)
    train = (man["split"] == "train").to_numpy()
    silence = (man["label"] == cfg["labels"]["silence"]).to_numpy()

    # ---- 1. PCA on a class-balanced sample of training clips
    idx = np.concatenate(
        [
            rng.choice(r, size=min(cp["sample_per_class"], len(r)), replace=False)
            for r in (np.flatnonzero(train & (man["label"] == c).to_numpy()) for c in order)
        ]
    )
    labels = man["label"].to_numpy()[idx]
    z = StandardScaler().fit_transform(x_all[idx])
    pca = PCA(svd_solver="full", random_state=seed).fit(z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, variance) + 1)
    pcs = pca.transform(z)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(np.arange(1, len(cum) + 1), pca.explained_variance_ratio_, color="#9ecae1", label="per component")
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#08519c", marker="o", ms=3, label="cumulative")
    ax.axhline(variance, color="grey", lw=0.8, ls="--")
    ax.axvline(n90, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(f"PCA of the clip vector: {n90} of {len(cum)} components reach {variance:.0%}")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "fig03_clip_pca_variance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. PCA plane and t-SNE, colored by class
    sub = rng.choice(len(idx), size=min(cp["tsne_points"], len(idx)), replace=False)
    emb = TSNE(
        n_components=2, perplexity=cp["tsne_perplexity"], init="pca", random_state=seed
    ).fit_transform(pcs[sub, :n90])
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for j, c in enumerate(order):
        m = labels[sub] == c
        axes[0].scatter(pcs[sub][m, 0], pcs[sub][m, 1], s=5, color=cmap(j), alpha=0.6, label=c)
        axes[1].scatter(emb[m, 0], emb[m, 1], s=5, color=cmap(j), alpha=0.6)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("First two principal components")
    axes[1].set_title(f"t-SNE of the first {n90} components (perplexity {cp['tsne_perplexity']})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[0].legend(markerscale=3, frameon=False, fontsize=7, ncol=2)
    fig.savefig(fig_dir / "fig04_clip_embedding.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. what do unsupervised partitions line up with?
    results = {}
    results["all classes vs word label"] = partition_agreement(x_all[idx], labels, seed, variance)

    sil_rows = np.flatnonzero(train & silence)
    results["silence clips vs noise source"] = partition_agreement(
        x_all[sil_rows], man["speaker_id"].to_numpy()[sil_rows], seed, variance
    )

    kw = train & (man["source"] == "keyword").to_numpy()
    counts = man.loc[kw, "speaker_id"].value_counts()
    eligible = np.sort(counts.index[counts >= cp["prolific_min_clips"]].to_numpy())
    chosen = rng.choice(eligible, size=min(cp["prolific_speakers"], len(eligible)), replace=False)
    rows = np.flatnonzero(kw & man["speaker_id"].isin(chosen).to_numpy())
    words, speakers = man["word"].to_numpy()[rows], man["speaker_id"].to_numpy()[rows]
    resid = word_residuals(x_all, man["word"], train & ~silence)
    for rep_name, rep in (("raw features", x_all[rows]), ("word mean removed", resid[rows])):
        results[f"{rep_name} vs word"] = partition_agreement(rep, words, seed, variance)
        results[f"{rep_name} vs speaker"] = partition_agreement(rep, speakers, seed, variance)

    out = {
        "features_used": feature_sets(names)["classifier"],
        "sample": {"clips": int(len(idx)), "per_class": cp["sample_per_class"]},
        "pca": {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "components_for_target_variance": n90,
            "target_variance": variance,
        },
        "prolific_subset": {"speakers": int(len(chosen)), "clips": int(len(rows))},
        "partition_agreement": results,
    }
    (rep_dir / "clip_structure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    keys = list(results)
    ax.barh(np.arange(len(keys)), [results[k]["nmi"] for k in keys], color="#4C72B0")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([f"{k}  (k={results[k]['k']})" for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized mutual information between k-means clusters and the target")
    ax.set_title("What k-means clusters of clip features line up with")
    fig.savefig(fig_dir / "fig05_clip_structure.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for k, v in results.items():
        print(f"{k:38s} k={v['k']:3d}  NMI={v['nmi']:.3f}  ARI={v['ari']:.3f}")
    print(f"PCA: {n90} components for {variance:.0%} variance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
