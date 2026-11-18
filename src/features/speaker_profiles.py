#!/usr/bin/env python3
"""
One feature vector per speaker, built from a handful of that speaker's clips.

The profile answers a practical question: if a collecting team could listen
to a few "screening" recordings of a candidate before deciding whether to
record them in full, what could it know about that speaker?

Two things make a naive average of clip features a poor profile:

* It mixes what was said with who said it. A speaker whose four screening
  clips are "yes, yes, stop, go" would look different from the same speaker
  saying "no, left, off, up". So each clip is first expressed as its
  deviation from the training-set mean of the word it contains.
* It is only as reliable as the number of clips behind it, which is why the
  profile size is a config value and its reliability is measured in
  patterns/speaker_clusters.py rather than assumed.

Only training speakers are profiled, and the word means are computed on
training clips alone, so nothing downstream of the profiles has seen a
validation or test recording.

Outputs (reports/phase2/)
    speaker_profiles.csv            one row per pooled speaker
    screening_clips.csv             which clips each profile was built from
    feature_variance_shares.csv     per feature: share of variance between words / between speakers
    speaker_profiles_summary.json
    figures/phase2/fig02_variance_shares.png

Usage
    python src/features/speaker_profiles.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.store import load_features  # noqa: E402
from utils.common import ensure_dir, load_config  # noqa: E402


def profile_matrix(feats: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Clip features in the form used for profiles: pitch on a log scale."""
    out = feats.astype(np.float64).copy()
    j = names.index("f0_hz")
    out[:, j] = np.log2(out[:, j])
    cols = list(names)
    cols[j] = "f0_log2"
    return out, cols


def word_residuals(x: np.ndarray, words: pd.Series, fit_mask: np.ndarray) -> np.ndarray:
    """Subtract, from every clip, the mean of its word computed on `fit_mask` rows."""
    means = pd.DataFrame(x[fit_mask]).groupby(words[fit_mask].to_numpy()).mean()
    overall = np.nanmean(x[fit_mask], axis=0)
    lookup = means.reindex(words.to_numpy()).to_numpy()
    lookup = np.where(np.isnan(lookup), overall, lookup)
    return x - lookup


def variance_share(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per column: between-group sum of squares over total sum of squares (NaN-aware)."""
    df = pd.DataFrame(x)
    grand = df.mean()
    g = df.groupby(groups)
    n, mu = g.count(), g.mean()
    between = (n * (mu - grand) ** 2).sum()
    total = ((df - grand) ** 2).sum()
    return (between / total).to_numpy()


def sample_rows(rows_by_speaker: dict, k: int, rng: np.random.Generator) -> dict:
    return {s: np.sort(rng.choice(r, size=k, replace=False)) for s, r in rows_by_speaker.items()}


def build_profiles(x: np.ndarray, picked: dict) -> pd.DataFrame:
    """Mean of the selected rows per speaker; NaN where no selected clip defines a value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column -> NaN, by intent
        data = {s: np.nanmean(x[r], axis=0) for s, r in picked.items()}
    return pd.DataFrame.from_dict(data, orient="index")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config("configs/phase2.yaml")
    sp = cfg["speaker_profiles"]
    rng = np.random.default_rng(cfg["seed"])
    rep_dir = ensure_dir(cfg["paths"]["reports_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

    man, feats, names = load_features(cfg)
    x, cols = profile_matrix(feats, names)

    human = (man["split"] == sp["split"]) & (man["label"] != cfg["labels"]["silence"])
    human = human.to_numpy()
    counts = man.loc[human, "speaker_id"].value_counts()
    pool = np.sort(counts.index[counts >= sp["min_clips"]].to_numpy())
    in_pool = human & man["speaker_id"].isin(pool).to_numpy()

    # ---- how much of each feature is about the word, how much about the speaker
    resid = word_residuals(x, man["word"], human)
    spk, wrd = man.loc[in_pool, "speaker_id"].to_numpy(), man.loc[in_pool, "word"].to_numpy()
    shares = pd.DataFrame(
        {
            "feature": cols,
            "between_word": variance_share(x[in_pool], wrd),
            "between_speaker": variance_share(x[in_pool], spk),
            "between_speaker_after_word_removal": variance_share(resid[in_pool], spk),
        }
    ).round(4)
    shares.to_csv(rep_dir / "feature_variance_shares.csv", index=False)

    # ---- profiles from a fixed number of screening clips
    base = resid if sp["residualize_word"] else x
    rows_by_speaker = {s: np.flatnonzero(in_pool & (man["speaker_id"] == s).to_numpy()) for s in pool}
    picked = sample_rows(rows_by_speaker, sp["profile_clips"], rng)
    prof = build_profiles(base, picked)
    prof.columns = cols
    undefined = prof.isna().sum()
    # A profile value is undefined only when none of the screening clips has
    # it (pitch, for a speaker whose screening clips are all unvoiced). The
    # pool median is the neutral choice: after word removal it means "typical
    # for the words spoken".
    prof = prof.fillna(prof.median())
    prof.insert(0, "n_clips", counts.reindex(prof.index).to_numpy())
    prof.index.name = "speaker_id"
    prof.round(5).to_csv(rep_dir / "speaker_profiles.csv")

    screening = pd.DataFrame(
        [(s, int(r), man["filepath"].iat[r]) for s, rows in picked.items() for r in rows],
        columns=["speaker_id", "manifest_row", "filepath"],
    )
    screening.to_csv(rep_dir / "screening_clips.csv", index=False)

    summary = {
        "split": sp["split"],
        "speakers_in_split": int(len(counts)),
        "min_clips": sp["min_clips"],
        "speakers_in_pool": int(len(pool)),
        "speakers_excluded": int(len(counts) - len(pool)),
        "clips_of_pooled_speakers": int(in_pool.sum()),
        "profile_clips": sp["profile_clips"],
        "screening_clips_total": int(len(screening)),
        "residualize_word": sp["residualize_word"],
        "profile_features": cols,
        "profiles_with_undefined_value": {k: int(v) for k, v in undefined.items() if v},
        # With g groups and n clips, a feature unrelated to the grouping
        # still shows a between-group share of about (g - 1) / (n - 1).
        "variance_share_expected_by_chance": {
            "between_word": float((len(np.unique(wrd)) - 1) / (in_pool.sum() - 1)),
            "between_speaker": float((len(pool) - 1) / (in_pool.sum() - 1)),
        },
        "mean_variance_share": {
            "between_word": float(shares["between_word"].mean()),
            "between_speaker": float(shares["between_speaker"].mean()),
            "between_speaker_after_word_removal": float(shares["between_speaker_after_word_removal"].mean()),
        },
    }
    (rep_dir / "speaker_profiles_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figure
    fig, ax = plt.subplots(figsize=(11, 3.8))
    pos = np.arange(len(cols))
    ax.bar(pos - 0.27, shares["between_word"], 0.27, label="between words")
    ax.bar(pos, shares["between_speaker"], 0.27, label="between speakers")
    ax.bar(pos + 0.27, shares["between_speaker_after_word_removal"], 0.27, label="between speakers, word mean removed")
    ax.set_xticks(pos)
    ax.set_xticklabels(cols, rotation=90, fontsize=6.5)
    ax.set_ylabel("share of variance")
    ax.set_title("What each clip feature varies with (training clips of pooled speakers)")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(fig_dir / "fig02_variance_shares.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "profile_features"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
