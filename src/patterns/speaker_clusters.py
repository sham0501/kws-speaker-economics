#!/usr/bin/env python3
"""
Do training speakers fall into groups, and can a few clips place a speaker
in the right one?

Steps
  1. Standardize the speaker profiles and keep the principal components
     that reach the configured share of variance.
  2. For every k in the configured range, cluster with k-means and with
     Ward linkage, and score the partitions (silhouette, Davies-Bouldin).
  3. Put the silhouette next to what the same procedure finds in Gaussian
     data with the same component variances. k-means always returns k
     clusters; the reference tells whether they are more than a slicing of
     one cloud.
  4. Measure how much each partition moves when the initialization changes,
     when a fifth of the speakers are left out, when the scaling changes,
     and when the clustering method changes.
  5. Choose k: among the values of k that are stable under
     re-initialization, the one whose silhouette exceeds the Gaussian
     reference by the widest margin. The raw silhouette is not used for the
     choice because it falls with k even when there is nothing to find.
  6. Describe the clusters in terms of the readable descriptors.
  7. Check how many clips a profile needs: build two profiles per speaker
     from disjoint clips and see whether they land in the same cluster.

Outputs
    reports/phase2/k_selection.csv
    reports/phase2/speaker_clusters.csv
    reports/phase2/clustering_summary.json
    figures/phase2/fig06_k_selection.png
    figures/phase2/fig07_speaker_clusters.png
    figures/phase2/fig08_cluster_profiles.png
    figures/phase2/fig09_profile_reliability.png

Usage
    python src/patterns/speaker_clusters.py
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
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.speaker_profiles import build_profiles, profile_matrix, sample_rows, word_residuals  # noqa: E402
from features.store import load_features  # noqa: E402
from utils.common import ensure_dir, load_config, resolve  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

DESCRIPTORS = [
    "f0_log2",
    "voiced_frac",
    "centroid_hz",
    "flatness",
    "zcr",
    "active_frac",
    "noise_floor_db",
    "dyn_range_db",
    "level_db",
    "mfcc_mean_0",
    "mfcc_mean_1",
    "mfcc_mean_2",
]


def kmeans(x, k, seed, n_init):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)


def evaluate_k(p: np.ndarray, k: int, cl: dict, seed: int, p_robust: np.ndarray) -> tuple[dict, np.ndarray]:
    """All scores for one value of k. Returns (row, reference labels)."""
    st = cl["stability"]
    rng = np.random.default_rng(seed + k)
    ref = kmeans(p, k, seed, cl["n_init"])
    labels = ref.labels_
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(p)

    null = []
    sd = p.std(axis=0)
    for _ in range(cl["null_draws"]):
        g = rng.normal(0.0, sd, size=p.shape)
        null.append(silhouette_score(g, kmeans(g, k, int(rng.integers(1 << 31)), 3).labels_))

    seed_ari = [adjusted_rand_score(labels, kmeans(p, k, s, 1).labels_) for s in range(st["n_seeds"])]

    sub_ari = []
    n_sub = int(round(st["subsample_frac"] * len(p)))
    for _ in range(st["n_subsamples"]):
        keep = rng.choice(len(p), size=n_sub, replace=False)
        fit = kmeans(p[keep], k, int(rng.integers(1 << 31)), cl["n_init"])
        sub_ari.append(adjusted_rand_score(labels, fit.predict(p)))

    robust = kmeans(p_robust, k, seed, cl["n_init"]).labels_

    row = {
        "k": k,
        "silhouette_kmeans": silhouette_score(p, labels),
        "silhouette_ward": silhouette_score(p, ward),
        "silhouette_null_mean": float(np.mean(null)),
        "silhouette_null_p95": float(np.quantile(null, 0.95)),
        "silhouette_excess": silhouette_score(p, labels) - float(np.mean(null)),
        "davies_bouldin": davies_bouldin_score(p, labels),
        "ari_seeds_mean": float(np.mean(seed_ari)),
        "ari_seeds_min": float(np.min(seed_ari)),
        "ari_subsamples_mean": float(np.mean(sub_ari)),
        "ari_subsamples_min": float(np.min(sub_ari)),
        "ari_robust_scaling": adjusted_rand_score(labels, robust),
        "ari_kmeans_vs_ward": adjusted_rand_score(labels, ward),
        "smallest_cluster": int(np.bincount(labels).min()),
    }
    return row, labels


def choose_k(table: pd.DataFrame, min_seed_ari: float) -> tuple[int, str]:
    stable = table[table["ari_seeds_mean"] >= min_seed_ari]
    if len(stable):
        k = int(stable.loc[stable["silhouette_excess"].idxmax(), "k"])
        return k, f"largest silhouette excess over the Gaussian reference among k with mean seed ARI >= {min_seed_ari}"
    k = int(table.loc[table["silhouette_excess"].idxmax(), "k"])
    return k, f"no k reached mean seed ARI >= {min_seed_ari}; largest silhouette excess overall"


def split_half(base, rows_by_speaker, m, scaler, pca, model, fill, rng) -> dict:
    """Two profiles per speaker from disjoint sets of m clips; do they agree?"""
    picked = sample_rows(rows_by_speaker, 2 * m, rng)
    halves = []
    for sl in (slice(0, m), slice(m, 2 * m)):
        # sample_rows returns sorted rows; shuffle so the halves are not ordered by row
        prof = build_profiles(base, {s: rng.permutation(r)[sl] for s, r in picked.items()})
        halves.append(prof.fillna(fill).to_numpy())
    za, zb = scaler.transform(halves[0]), scaler.transform(halves[1])
    ca, cb = model.predict(pca.transform(za)), model.predict(pca.transform(zb))
    share = np.bincount(np.concatenate([ca, cb]), minlength=model.n_clusters) / (2 * len(ca))
    feature_r = [np.corrcoef(za[:, j], zb[:, j])[0, 1] for j in range(za.shape[1])]
    return {
        "profile_clips": m,
        "same_cluster_rate": float(np.mean(ca == cb)),
        "chance_rate": float(np.sum(share**2)),
        "ari": float(adjusted_rand_score(ca, cb)),
        "mean_feature_correlation": float(np.nanmean(feature_r)),
    }


def main() -> int:
    cfg = load_config("configs/phase2.yaml")
    cl, sp, seed = cfg["clustering"], cfg["speaker_profiles"], cfg["seed"]
    rep_dir = ensure_dir(cfg["paths"]["reports_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

    prof = pd.read_csv(resolve(cfg["paths"]["reports_dir"]) / "speaker_profiles.csv", index_col=0, dtype={"speaker_id": str})
    n_clips = prof.pop("n_clips")
    cols = list(prof.columns)

    scaler = StandardScaler().fit(prof.to_numpy())
    z = scaler.transform(prof.to_numpy())
    pca = PCA(n_components=cl["pca_variance"], svd_solver="full", random_state=seed).fit(z)
    p = pca.transform(z)
    z_rob = RobustScaler().fit_transform(prof.to_numpy())
    p_rob = PCA(n_components=p.shape[1], svd_solver="full", random_state=seed).fit_transform(z_rob)
    print(f"{len(prof)} speakers, {len(cols)} profile features, {p.shape[1]} components kept")

    # ---- scores for every k
    rows, labels_by_k = [], {}
    for k in range(cl["k_min"], cl["k_max"] + 1):
        row, labels_by_k[k] = evaluate_k(p, k, cl, seed, p_rob)
        rows.append(row)
        print(
            f"  k={k:2d}  silhouette={row['silhouette_kmeans']:.3f} (reference {row['silhouette_null_mean']:.3f})"
            f"  seed ARI={row['ari_seeds_mean']:.2f}  subsample ARI={row['ari_subsamples_mean']:.2f}"
        )
    table = pd.DataFrame(rows)
    table.round(4).to_csv(rep_dir / "k_selection.csv", index=False)

    k, rule = choose_k(table, cl["stability"]["min_seed_ari"])
    chosen = table.set_index("k").loc[k]
    labels = labels_by_k[k]
    model = kmeans(p, k, seed, cl["n_init"])
    print(f"chosen k = {k} ({rule})")

    out = pd.DataFrame({"speaker_id": prof.index, "cluster": labels, "n_clips": n_clips.to_numpy()})
    for j in range(p.shape[1]):
        out[f"pc{j + 1}"] = p[:, j]
    out.round(5).to_csv(rep_dir / "speaker_clusters.csv", index=False)

    # ---- description of the clusters
    zdf = pd.DataFrame(z, columns=cols)
    centers = zdf.groupby(labels).mean()
    desc = [d for d in DESCRIPTORS if d in cols]
    clusters = []
    for c in range(k):
        top = centers.loc[c].abs().sort_values(ascending=False).index[:5]
        members = n_clips.to_numpy()[labels == c]
        clusters.append(
            {
                "cluster": c,
                "speakers": int(np.sum(labels == c)),
                "share": float(np.mean(labels == c)),
                "median_clips_per_speaker": float(np.median(members)),
                "share_with_40_or_more_clips": float(np.mean(members >= 40)),
                "most_distinctive_features": {f: round(float(centers.loc[c, f]), 3) for f in top},
                "descriptor_z": {f: round(float(centers.loc[c, f]), 3) for f in desc},
            }
        )

    # ---- how many clips does a profile need?
    man, feats, names = load_features(cfg)
    x, _ = profile_matrix(feats, names)
    human = ((man["split"] == sp["split"]) & (man["label"] != cfg["labels"]["silence"])).to_numpy()
    base = word_residuals(x, man["word"], human) if sp["residualize_word"] else x
    need = 2 * max(cl["reliability_clips"])
    eligible = [s for s in prof.index if n_clips[s] >= need]
    spk = man["speaker_id"].to_numpy()
    rows_by_speaker = {s: np.flatnonzero(human & (spk == s)) for s in eligible}
    rng = np.random.default_rng(seed)
    fill = pd.Series(prof.median().to_numpy())
    reliability = [split_half(base, rows_by_speaker, m, scaler, pca, model, fill, rng) for m in cl["reliability_clips"]]
    for r in reliability:
        print(
            f"  profile from {r['profile_clips']} clip(s): same cluster {r['same_cluster_rate']:.2f}"
            f" (chance {r['chance_rate']:.2f}), feature r = {r['mean_feature_correlation']:.2f}"
        )

    summary = {
        "speakers": int(len(prof)),
        "profile_features": int(len(cols)),
        "pca_components": int(p.shape[1]),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "k_range": [cl["k_min"], cl["k_max"]],
        "chosen_k": k,
        "selection_rule": rule,
        "chosen_k_scores": {c: float(chosen[c]) for c in table.columns if c != "k"},
        "silhouette_exceeds_null_p95": bool(chosen["silhouette_kmeans"] > chosen["silhouette_null_p95"]),
        "clusters": clusters,
        "profile_reliability": {"speakers": int(len(eligible)), "clips_required": need, "by_profile_size": reliability},
    }
    (rep_dir / "clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figures
    ks = table["k"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ks, table["silhouette_kmeans"], marker="o", label="k-means")
    axes[0].plot(ks, table["silhouette_ward"], marker="s", label="Ward linkage")
    axes[0].plot(ks, table["silhouette_null_mean"], color="grey", ls="--", label="Gaussian reference (mean)")
    axes[0].fill_between(ks, table["silhouette_null_mean"], table["silhouette_null_p95"], color="grey", alpha=0.2)
    axes[0].axvline(k, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("number of clusters k")
    axes[0].set_ylabel("mean silhouette")
    axes[0].set_title("Cluster quality against a no-structure reference")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(ks, table["ari_seeds_mean"], marker="o", label="re-initialization")
    axes[1].plot(ks, table["ari_subsamples_mean"], marker="s", label="80% subsamples")
    axes[1].plot(ks, table["ari_robust_scaling"], marker="^", label="robust scaling")
    axes[1].plot(ks, table["ari_kmeans_vs_ward"], marker="v", label="k-means vs Ward")
    axes[1].axhline(cl["stability"]["min_seed_ari"], color="grey", lw=0.8, ls="--")
    axes[1].axvline(k, color="k", lw=0.8, ls=":")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("adjusted Rand index vs reference partition")
    axes[1].set_title("Stability of the partition")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "fig06_k_selection.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    evr = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.get_cmap("tab10")
    for c in range(k):
        m = labels == c
        axes[0].scatter(p[m, 0], p[m, 1], s=7, alpha=0.6, color=cmap(c), label=f"cluster {c} (n={m.sum()})")
    axes[0].set_xlabel(f"PC1 ({evr[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%})")
    axes[0].set_title(f"Speaker profiles, k-means with k={k}")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)
    sc = axes[1].scatter(p[:, 0], p[:, 1], s=7, alpha=0.6, c=np.log10(n_clips.to_numpy()), cmap="viridis")
    axes[1].set_xlabel(f"PC1 ({evr[0]:.1%})")
    axes[1].set_title("Same plane, colored by clips recorded per speaker")
    fig.colorbar(sc, ax=axes[1], label="log10(clips per speaker)")
    fig.savefig(fig_dir / "fig07_speaker_clusters.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1 + 0.75 * len(desc), 1.4 + 0.45 * k))
    data = centers[desc].to_numpy()
    lim = max(1.0, float(np.abs(data).max()))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(desc)))
    ax.set_xticklabels(desc, rotation=45, ha="right")
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([f"cluster {c} (n={int(np.sum(labels == c))})" for c in range(k)])
    for i in range(k):
        for j in range(len(desc)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cluster means of the readable descriptors (z-scores over the pool)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(fig_dir / "fig08_cluster_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    rel = pd.DataFrame(reliability)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(rel["profile_clips"], rel["same_cluster_rate"], marker="o", label="two profiles land in the same cluster")
    ax.plot(rel["profile_clips"], rel["mean_feature_correlation"], marker="s", label="mean split-half correlation of features")
    ax.plot(rel["profile_clips"], rel["chance_rate"], color="grey", ls="--", label="same cluster by chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rel["profile_clips"])
    ax.set_xticklabels(rel["profile_clips"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("clips per profile")
    ax.set_title(f"How many clips a profile needs ({len(eligible)} speakers)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "fig09_profile_reliability.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
