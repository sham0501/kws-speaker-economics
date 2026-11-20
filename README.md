# Data Economics of Keyword Spotting

**Smarter Speaker Selection vs. Random Sampling under a Fixed Recording Budget**

Data Analytics course project, Fall 2026 — **Route D:** Google Speech Commands, audio analysis and KWS.

---

## Research question

Given a fixed recording budget (total number of clips), which data-collection strategy produces a better keyword-spotting model?

- **More speakers**, with fewer samples per person, or
- **Fewer speakers**, with more samples per person?

And can **clustering of speakers' audio representations** pick a smarter subset than random sampling?

Evaluation metric: macro-F1 on a test set that is **fully speaker-independent**.

The complete problem statement is recorded in [`PROJECT_CARD.md`](PROJECT_CARD.md).

---

## Phase status

| Phase | Topic | Status | Tag |
|---|---|---|---|
| 1 | Problem framing, data understanding, quality, EDA | done | `phase-1` |
| 2 | Data prep, feature engineering, speaker clustering | done | `phase-2` |
| 3 | Predictive modeling and the budget experiment | | `phase-3` |
| 4 | Interpretability, speaker generalization, noise robustness | | `phase-4` |
| 5 | Robustness, case study, final version | | `phase-5-final` |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Reference hardware:** Lenovo ThinkPad T14s, 11th-gen Intel Core i5, integrated Iris Xe graphics, no discrete GPU. Everything runs on CPU with no CUDA dependency.

## Running the pipeline

Run from the repository root, in this order. Each step reads only what the steps above it wrote.

### Phase 1

| # | Command | Reads | Writes |
|---|---|---|---|
| 1 | `python src/data/download_dataset.py` | official URL | `data/raw/` (not versioned), `reports/phase1/download_manifest.json` |
| 2 | `python src/data/build_subset.py` | `data/raw/` | `data/subset_manifest.csv`, `data/raw/silence_clips/`, `reports/phase1/subset_summary.json`, `reports/phase1/silence_provenance.csv` |
| 3 | `python src/data/silence_overlap.py` | manifest, silence clips, background recordings | `reports/phase1/silence_provenance.csv`, `reports/phase1/silence_overlap.json`, `figures/phase1/fig10_silence_coverage.png` |
| 4 | `python src/data/audit_quality.py` | manifest, audio | `reports/phase1/audit_*.csv`, `reports/phase1/audit_summary.json` |
| 5 | `python src/eda/run_eda.py` | manifest, audio, audit | `figures/phase1/fig01..fig09`, `reports/phase1/eda_manifest.json`, `reports/phase1/budget_speaker_range.csv`, `reports/phase1/speaker_profiles_phase1.npz` |
| 6 | `python src/reports/build_pdf.py reports/phase1/phase1_report.md reports/phase1.pdf` | report and figures | `reports/phase1.pdf` |

Step 2 is deterministic for a given seed and only needs to run once; step 3 works on the subset as it is on disk and does not rebuild it.

### Phase 2

| # | Command | Reads | Writes |
|---|---|---|---|
| 1 | `python src/features/build_features.py` | manifest, audio | `data/processed/phase2/` (not versioned, ~1.5 GB), `reports/phase2/features_summary.json`, `reports/phase2/feature_catalog.csv` |
| 2 | `python src/features/augment.py` | one train clip, train background recordings | `reports/phase2/augmentation_settings.json`, `figures/phase2/fig01_augmentation_examples.png` |
| 3 | `python src/features/speaker_profiles.py` | clip features | `reports/phase2/speaker_profiles.csv`, `screening_clips.csv`, `screening_budget.json`, `feature_variance_shares.csv`, `speaker_profiles_summary.json`, `figures/phase2/fig02_variance_shares.png` |
| 4 | `python src/patterns/clip_patterns.py` | clip features | `reports/phase2/clip_structure.json`, `figures/phase2/fig03..fig05` |
| 5 | `python src/patterns/speaker_clusters.py` | speaker profiles, screening clips, clip features | `reports/phase2/k_selection.csv`, `speaker_clusters.csv`, `clustering_summary.json`, `figures/phase2/fig06..fig09` |
| 6 | `python src/reports/build_pdf.py reports/phase2/phase2_report.md reports/phase2.pdf` | report and figures | `reports/phase2.pdf` |

Phase-2 stages read the clip features through `src/features/store.py`, which refuses a feature build made from another manifest, with other signal or feature settings, or by an older version of the extraction code.

### Smoke tests

The `--limit` options run a stage on a small sample for a quick check. Their outputs go to `limited/` subfolders, which are not versioned and are never read by a later stage:

```bash
python src/data/audit_quality.py --limit 5000                       # reports/phase1/limited/
python src/eda/run_eda.py --max-files-per-class 300 --skip-speaker-pca   # figures/phase1/limited/, reports/phase1/limited/
python src/features/build_features.py --limit 600                   # data/processed/phase2/limited/, reports/phase2/limited/
```

### Checks

```bash
python tests/test_budget_range.py
python tests/test_padding_invariance.py
python tests/test_split_half.py
python tests/test_feature_builds.py      # needs a full feature build first
python tests/test_interrupted_build.py   # an interrupted build is refused (runs in a temporary folder)
```

Each file also runs under `python -m pytest tests`.

---

## Data

| | |
|---|---|
| Dataset | Google Speech Commands **v0.02** |
| Official source | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| License | CC BY 4.0 |
| Full dataset | 105,829 clips, 35 words, 2,618 speakers, 16 kHz mono |
| This project's subset | 12 classes: 10 keywords + `unknown` + `silence` |

**Raw data is not stored in this repository** — the archive is ~2.3 GB, which does not belong in version control. It's rebuilt with `download_dataset.py`, and its URL, size and SHA256 are recorded in `reports/phase1/download_manifest.json`. Details in [`data/README.md`](data/README.md).

### Speaker-independent split

The train/val/test split is read **directly** from Google's official files (`validation_list.txt` and `testing_list.txt`). These are built from a hash of `speaker_id`, so each speaker appears in exactly one set. `build_subset.py` **automatically verifies** this property after building the manifest and exits with a non-zero status if leakage is found.

---

## Repository layout

```
README.md                  this file
PROJECT_CARD.md             problem statement, main question, split design, leakage risks
AI_USAGE.md                 AI/LLM usage log
requirements.txt
configs/
    phase1.yaml             subset, audit and EDA parameters
    phase2.yaml             signal, features, augmentation, profiles, clustering
data/
    README.md               data provenance, license, subset rationale
    raw/                    (not versioned) downloaded dataset and silence clips
    processed/              (not versioned) standardized waveforms and clip features
    subset_manifest.csv     output of build_subset.py
src/
    utils/common.py         config, seeding, speaker_id extraction, wav I/O, manifest paths
    data/                   download, subset, silence provenance, quality audit
    eda/run_eda.py
    features/
        waveform.py         length and level standardization
        audio_features.py   log-mel, MFCC, descriptors, the feature catalog
        build_features.py   runs the pipeline over the manifest
        store.py            checked access to the prepared arrays
        augment.py          time shift, gain, added noise (training clips only)
        speaker_profiles.py screening profiles and the screening budget
    patterns/
        clip_patterns.py    structure of the clip features
        speaker_clusters.py clustering, k selection, stability, reliability
    reports/build_pdf.py    markdown report -> PDF with embedded figures
tests/                      checks for the calculations that are easy to get wrong
figures/phase1/ phase2/     figures by phase
models/                    (not versioned) model weights
reports/
    phase1/ phase2/ ...     generated tables and the report of each phase
```

---

## Phase-1 outputs

| File | Content |
|---|---|
| `reports/phase1/phase1_report.md` | phase-1 report, revised in phase 2 (PDF: `reports/phase1.pdf`) |
| `reports/phase1/download_manifest.json` | data-acquisition record |
| `reports/phase1/subset_summary.json` | subset stats and leakage verification |
| `reports/phase1/audit_summary.json` | quality-audit summary |
| `reports/phase1/audit_per_file.csv` | per-file signal metrics |
| `reports/phase1/audit_flags.csv` | flagged files with reasons |
| `reports/phase1/audit_per_speaker.csv` | per-speaker sample counts by class |
| `reports/phase1/silence_provenance.csv` | source recording and start sample of every silence clip |
| `reports/phase1/silence_overlap.json` | how much the silence clips of each recording overlap |
| `reports/phase1/budget_speaker_range.csv` | fewest and most speakers per budget and k, and where exactly-k is feasible |
| `reports/phase1/eda_manifest.json` | what each figure was drawn from (seeded samples, sources, words per profile) |
| `figures/phase1/*.png` | 10 figures |

### Figures

| Figure | Content |
|---|---|
| `fig01_class_distribution` | 12-class distribution by split |
| `fig02_samples_per_speaker` | sample imbalance across speakers + concentration curve |
| `fig03_speakers_vs_budget` | fewest / most speakers that can supply a budget, and where exactly k clips per speaker is feasible |
| `fig04_duration_distribution` | file-length distribution |
| `fig05_rms_by_class` | signal energy by class (silence detection) |
| `fig06_waveforms` | example waveforms per class |
| `fig07_logmel` | log-mel spectrograms of the same examples |
| `fig08_mean_logmel_per_class` | average spectral pattern per class (silence balanced over its train sources) |
| `fig09_speaker_mfcc_pca` | speaker audio-profile space, random speakers, word-diverse profiles |
| `fig10_silence_coverage` | reuse of each background recording in the silence class |

---

## Phase-2 outputs

| File | Content |
|---|---|
| `reports/phase2/phase2_report.md` | phase-2 report (PDF: `reports/phase2.pdf`) |
| `reports/phase2/features_summary.json` | preparation steps and their effect, settings, undefined values |
| `reports/phase2/feature_catalog.csv` | feature groups, definitions and the reason for each |
| `reports/phase2/augmentation_settings.json` | augmentation ranges and the gain-invariance check |
| `reports/phase2/feature_variance_shares.csv` | per feature: variance between words and between speakers |
| `reports/phase2/speaker_profiles.csv` | one screening profile per candidate speaker |
| `reports/phase2/screening_clips.csv` | the clips behind each profile |
| `reports/phase2/screening_budget.json` | screening cost and how it is counted |
| `reports/phase2/speaker_profiles_summary.json` | pool, profile settings, chance levels of the variance shares |
| `reports/phase2/clip_structure.json` | clip PCA and what k-means clusters of clips line up with |
| `reports/phase2/k_selection.csv` | silhouette against a Gaussian reference and stability for k = 2..10 |
| `reports/phase2/speaker_clusters.csv` | cluster of every candidate |
| `reports/phase2/clustering_summary.json` | chosen k, cluster descriptions, profile reliability |
| `figures/phase2/*.png` | 9 figures |

### Figures

| Figure | Content |
|---|---|
| `fig01_augmentation_examples` | time shift, gain and added noise on one training clip |
| `fig02_variance_shares` | what each clip feature varies with: word or speaker |
| `fig03_clip_pca_variance` | explained variance of the clip vector |
| `fig04_clip_embedding` | PCA plane and t-SNE of the clips |
| `fig05_clip_structure` | agreement of clip clusters with word, speaker and noise source |
| `fig06_k_selection` | silhouette against a no-structure reference, and stability, for each k |
| `fig07_speaker_clusters` | the two speaker clusters, and the same plane by clips recorded |
| `fig08_cluster_profiles` | cluster means of the readable descriptors |
| `fig09_profile_reliability` | split-half reliability of profiles by size, and of 4-clip profiles on the whole pool |

---

## Reproducibility

- All seeds live in the config files (`seed: 1405`); every sample drawn for a figure uses it, so no figure depends on the row order of the manifest.
- No number is typed into the report by hand; everything comes from the generated JSON/CSV files.
- Feature-extraction parameters (`n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`) live in config, not in code.
- The length-normalization policy (zero-pad at the end, crop from the start) is recorded in `src/utils/common.py::fix_length`; clip statistics and level normalization ignore the padding.
- Everything fitted in phase 2 (word means, scalers, PCA, clusters) uses training speakers only.
- A matching tag is created at the end of each phase.

---

## Ethics and privacy

`speaker_id` is an **anonymous hash** used only to design the speaker-independent split and to analyze generalization. No attempt is made to deanonymize speakers or infer their identity, age, gender or accent. Full discussion in `PROJECT_CARD.md`, item 14.

## AI usage

Logged in [`AI_USAGE.md`](AI_USAGE.md).
