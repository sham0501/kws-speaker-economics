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

The complete problem statement, including all 14 required items, is recorded in [`PROJECT_CARD.md`](PROJECT_CARD.md).

---

## Phase status

| Phase | Topic | Status | Tag |
|---|---|---|---|
| 1 | Problem framing, data understanding, quality, EDA | ✅ done | `phase-1` |
| 2 | Data prep, feature engineering, speaker clustering | ⏳ | `phase-2` |
| 3 | Predictive modeling and the budget experiment | ⏳ | `phase-3` |
| 4 | Interpretability, speaker generalization, noise robustness | ⏳ | `phase-4` |
| 5 | Robustness, case study, final version | ⏳ | `phase-5-final` |

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/data/download_dataset.py   # ~2.3 GB download, one-time
python src/data/build_subset.py       # build the 12-class subset + speaker-independent split
python src/data/audit_quality.py      # audio quality audit + speaker audit
python src/eda/run_eda.py             # generate the 9 EDA figures
```

Lighter run for a quick check on a laptop with no GPU:

```bash
python src/data/audit_quality.py --limit 5000
python src/eda/run_eda.py --max-files-per-class 300 --skip-speaker-pca
```

**Reference hardware:** Lenovo ThinkPad T14s, 11th-gen Intel Core i5, integrated Iris Xe graphics, no discrete GPU. Everything runs on CPU with no CUDA dependency.

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
    phase1.yaml             every seed and parameter
data/
    README.md               data provenance, license, subset rationale
    raw/                    (gitignored) downloaded dataset
    subset_manifest.csv     output of build_subset.py
src/
    utils/common.py         config, seeding, speaker_id extraction, wav I/O
    data/download_dataset.py
    data/build_subset.py
    data/audit_quality.py
    eda/run_eda.py
    features/               (phase 2)
    models/                 (phase 3)
notebooks/                  exploratory notebooks
figures/phase1/             the 9 EDA figures
models/                    (gitignored) model weights
reports/
    phase1/ ... phase4/ final/
```

---

## Phase-1 outputs

| File | Content |
|---|---|
| `PROJECT_CARD.md` | problem statement, 14 items |
| `reports/phase1/phase1_report.md` | phase-1 report (source for the final write-up) |
| `reports/phase1/download_manifest.json` | data-acquisition record |
| `reports/phase1/subset_summary.json` | subset stats and leakage verification |
| `reports/phase1/audit_summary.json` | quality-audit summary |
| `reports/phase1/audit_per_file.csv` | per-file signal metrics |
| `reports/phase1/audit_flags.csv` | flagged files with reasons |
| `reports/phase1/audit_per_speaker.csv` | per-speaker sample counts by class |
| `figures/phase1/*.png` | 9 figures |

### Figures

| Figure | Content |
|---|---|
| `fig01_class_distribution` | 12-class distribution by split |
| `fig02_samples_per_speaker` | sample imbalance across speakers + concentration curve |
| `fig03_speakers_vs_budget` | **the core trade-off:** speakers vs. samples under a fixed budget |
| `fig04_duration_distribution` | file-length distribution |
| `fig05_rms_by_class` | signal energy by class (silence detection) |
| `fig06_waveforms` | example waveforms per class |
| `fig07_logmel` | log-mel spectrograms of the same examples |
| `fig08_mean_logmel_per_class` | average spectral pattern per class |
| `fig09_speaker_mfcc_pca` | speaker audio-profile space (groundwork for phase 2) |

---

## Reproducibility

- All seeds live in `configs/phase1.yaml` (`seed: 1405`).
- No number is typed into the report by hand; everything comes from the generated JSON/CSV files.
- Feature-extraction parameters (`n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`) live in config, not in code.
- The length-normalization policy (zero-pad at the end, crop from the start) is recorded in `src/utils/common.py::fix_length`.
- A matching tag is created at the end of each phase.

---

## Ethics and privacy

`speaker_id` is an **anonymous hash** used only to design the speaker-independent split and to analyze generalization. No attempt is made to deanonymize speakers or infer their identity, age, gender or accent. Full discussion in `PROJECT_CARD.md`, item 14.

## AI usage

Logged in [`AI_USAGE.md`](AI_USAGE.md).
