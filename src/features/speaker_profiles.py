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
  deviation from the mean of the word it contains. By default that mean is
  computed from the screening clips of all candidates and nothing else,
  because the scenario is a decision taken before anyone is recorded in
  full; config `word_reference: train` switches to the whole training
  split, which is the right reference for describing an existing archive.
* It is only as reliable as the number of clips behind it, which is why the
  profile size is a config value and its reliability is measured in
  patterns/speaker_clusters.py rather than assumed.

Only training speakers are profiled and the word means never use a
validation or test clip, so nothing downstream of the profiles has seen one.

What the screening costs is written to screening_budget.json: candidates,
clips per candidate, how a screening clip that is later used for training
is counted, and what the screening of candidates who are not selected
costs. Phase 3 uses that accounting to compare strategies fairly.

Outputs (reports/phase2/)
    speaker_profiles.csv            one row per pooled speaker
    screening_clips.csv             which clips each profile was built from
    screening_budget.json           what the screening costs and how it is counted
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
from utils.common import ensure_dir, load_config, resolve  # noqa: E402


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


def word_reference_mask(man: pd.DataFrame, cfg: dict, screening_rows=None) -> np.ndarray:
    """Rows whose per-word means are subtracted from every clip (see config `word_reference`)."""
    sp = cfg["speaker_profiles"]
    if sp["word_reference"] == "train":
        return ((man["split"] == sp["split"]) & (man["label"] != cfg["labels"]["silence"])).to_numpy()
    if sp["word_reference"] != "screening":
        raise ValueError(f"unknown word_reference: {sp['word_reference']}")
    if screening_rows is None:
        path = resolve(cfg["paths"]["reports_dir"]) / "screening_clips.csv"
        screening_rows = pd.read_csv(path)["manifest_row"].to_numpy()
    mask = np.zeros(len(man), dtype=bool)
    mask[np.asarray(screening_rows, dtype=int)] = True
    return mask


def profile_base(x: np.ndarray, man: pd.DataFrame, cfg: dict, screening_rows=None) -> np.ndarray:
    """Clip features in the form profiles are averaged from."""
    if not cfg["speaker_profiles"]["residualize_word"]:
        return x
    return word_residuals(x, man["word"], word_reference_mask(man, cfg, screening_rows))


def screening_budget(cfg: dict, n_candidates: int, words: pd.Series) -> dict:
    """Cost of screening, the rule for counting reused clips, and an example table."""
    s = int(cfg["speaker_profiles"]["profile_clips"])
    sb = cfg["screening_budget"]
    reuse = bool(sb["reuse_in_training"])
    table = []
    for n in sb["example_selected_speakers"]:
        if n > n_candidates:
            continue
        for k in sb["example_clips_per_speaker"]:
            reused = n * min(s, k) if reuse else 0
            table.append(
                {
                    "selected_speakers": n,
                    "clips_per_speaker": k,
                    "training_clips": n * k,
                    "screening_clips": n_candidates * s,
                    "screening_clips_reused_for_training": reused,
                    "screening_clips_without_training_use": n_candidates * s - reused,
                    "total_clips_recorded": n_candidates * s + n * k - reused,
                }
            )
    return {
        "candidates": n_candidates,
        "screening_clips_per_candidate": s,
        "screening_clips_total": n_candidates * s,
        "word_reference": cfg["speaker_profiles"]["word_reference"],
        "words_in_reference": int(words.nunique()),
        "reference_clips_per_word": {"min": int(words.value_counts().min()), "median": float(words.value_counts().median())},
        "reuse_in_training": reuse,
        "accounting": {
            "training_budget_view": (
                "every strategy trains on the same number of clips; screening is free. "
                "Measures the value of choosing speakers, as if profiles came from an existing archive."
            ),
            "recording_budget_view": (
                "every recorded clip counts. Random selection needs no screening; a profile-based "
                "strategy pays candidates x screening clips first, and a selected speaker's screening "
                "clips count toward their training clips. Strategies are compared at equal total recording cost."
            ),
        },
        "cost_table": table,
    }


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
    rows_by_speaker = {s: np.flatnonzero(in_pool & (man["speaker_id"] == s).to_numpy()) for s in pool}
    picked = sample_rows(rows_by_speaker, sp["profile_clips"], rng)
    screening_rows = np.concatenate(list(picked.values()))
    base = profile_base(x, man, cfg, screening_rows)
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
    budget = screening_budget(cfg, len(pool), man["word"].iloc[screening_rows])
    (rep_dir / "screening_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

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
        "word_reference": sp["word_reference"],
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
