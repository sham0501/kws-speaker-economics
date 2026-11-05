# Phase 1 Report — Problem Understanding, Data Understanding, Quality and Exploratory Analysis

**Project:** Data Economics of Keyword Spotting — Smarter Speaker Selection vs. Random Sampling <br>
**Route:** D — Google Speech Commands, audio analysis and KWS <br>
**Student:** SeyedHossein AhmadiMousavi <br>
**Student ID:** 402100334 <br> 
**Repository:** https://github.com/sham0501/kws-speaker-economics — tag `phase-1` <br>
**Date:** 2026/11/05 <br>

---

## 1. Problem framing

### 1.1 Context and the real decision

A team building a lightweight KWS model for on-device deployment needs to collect
audio data, and the recording budget (volunteer time, participation cost, labeling
cost) is limited. The operational decision is whether that budget should be spent on
**more speakers** or on **more samples per speaker**.

This is a data-collection design decision, not a modeling question. Its answer directly
shapes the recording instructions: "200 people, 5 words each" vs. "40 people, 25 words
each."

### 1.2 Main question

Under a fixed recording budget (total clip count), does increasing **speaker count**
(reducing samples per speaker) improve keyword-classification accuracy on **fully unseen
speakers (speaker-disjoint test)** more than increasing **samples per speaker** (with
fewer, fixed speakers)? And can selecting speakers via **clustering of audio
representations** (instead of random sampling) reach higher accuracy with the same
budget?

### 1.3 What decision does the analysis output support

The output is a curve of test macro-F1 against speaker count at a fixed budget, for two
selection strategies. If speaker count turns out to be the dominant factor, the
recommendation to a collecting team is to broaden participation even at the cost of
dropping to a handful of samples per person. If the cluster-driven strategy beats random
selection at equal budget, the recommendation goes one step further: screen candidate
speakers by their audio profile before recording the full set, rather than recruiting
blindly.

---

## 2. Data provenance

| Item | Value |
|---|---|
| Dataset name | Google Speech Commands v0.02 |
| Publisher | Pete Warden / Google (2018), CC BY 4.0 license |
| Official URL | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| Full size | 105,829 clips, 35 words, 2,618 speakers |
| Unit of observation | one 1-second audio clip of a single word |
| Format | WAV, 16-bit PCM, mono, 16 kHz |
| Collection period | 2017–2018 |
| Nature | **the result of a controlled process** (not observational) — see below |
| Archive SHA256 | af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58 |

**Collection method.** The data was collected through an open web app: volunteers were
asked to say specific words shown on screen into their device's microphone.
So **which words were said is controlled, but the recording conditions are not**
(microphone, room, background noise, and mouth-to-mic distance all vary freely). This
matters for this project: variation across speakers in this dataset conflates voice
variation with recording-channel variation, and the two cannot be fully separated when
interpreting results.

**Main limitations of how the data was produced** (these carry through to the final
report's Limitations section):

1. Self-selected volunteers — the sample is not representative of the general
   population.
2. Isolated single words, not connected speech.
3. No demographic metadata (age, gender, accent) and no recording-environment metadata
   (SNR, microphone type) exist; any "speaker subgroup" analysis must therefore rely on
   extracted audio features, not demographic labels.
4. Samples-per-speaker counts are heavily imbalanced (Figure 2).

### 2.1 Subset definition and justification

**What was chosen:** the 10 keywords `yes, no, up, down, left, right, on, off, stop, go`
plus `unknown` (balanced sampling from 25 other words) and `silence` (cut from
`_background_noise_`) — 12 classes total.

**Why this choice does not compromise validation:**

1. These 12 classes are exactly the official TensorFlow Simple Audio Recognition
   benchmark configuration, so results stay comparable to published work and this is not
   an arbitrary subset.
2. Google's **official** `validation_list.txt` and `testing_list.txt` are used
   untouched, only filtered to these 12 classes. No new split was invented.
3. The subset was logged **before any modeling** (this file and `PROJECT_CARD.md`, tag
   `phase-1`) and will not change after seeing results.
4. The project's question is about *speaker count*, not *class count*. Reducing the
   class count leaves the experiment's main axis (speaker diversity) untouched, while
   making it possible to run the experiment many times over on a CPU — essential for
   reporting uncertainty across seeds in phase 3.

| | train | val | test |
|---|---|---|---|
| Clip count | 36,923 | 4,443 | 4,888 |
| Speaker count | 2,027 | 247 | 240 |
| Class count | 12 | 12 | 12 |

---

## 3. Data-structure audit

The raw data is a folder of WAV files, not a table. The logical structure has been
turned into a flat table (`data/subset_manifest.csv`, 46,254 rows) by
`src/data/build_subset.py`:

| Column | Type | Role | Description |
|---|---|---|---|
| `filepath` | string | identifier | relative path to the WAV file; unique key |
| `label` | categorical | **target** | one of 12 classes |
| `word` | categorical | auxiliary | the original word before mapping to `unknown` |
| `speaker_id` | categorical | **group identifier** | anonymous speaker hash; the split unit |
| `utterance_index` | integer | auxiliary | which repetition of this word by this speaker |
| `split` | categorical | design | train / val / test from the official list |
| `source` | categorical | provenance | keyword / unknown_sampled / silence_generated |

**Extracting `speaker_id`.** The official naming convention is
`<label>/<speaker>_nohash_<index>.wav`, so the speaker id is the part before
`_nohash_`. It is an anonymous hash and is used only for split design and
generalization analysis.

**Time variable.** The dataset has no timestamps; the only "time" axis is the time
axis inside each signal. This is therefore not a time-series problem, and a temporal
split would not be meaningful — a speaker-grouped split is the correct equivalent.

---

## 4. Quality audit

Source: `reports/phase1/audit_summary.json` (produced by `src/data/audit_quality.py`;
run on the full 46,254-row manifest, no `--limit`).

### 4.1 Table-level audit

| Check | Result | Decision |
|---|---|---|
| Missing values | 0 (all 7 columns) | expected: the manifest is built directly from the filesystem by this project's own code, not from an external source with entry errors |
| Duplicate rows | 0 | |
| Invalid `speaker_id` | 0 | no files failed the official naming pattern — confirmed by the absence of a `[warn]` line in `build_subset.py`'s own output |
| Impossible (negative) `utterance_index` | 0 | |

### 4.2 Audio-file-level audit

| Metric | Result | Decision / interpretation |
|---|---|---|
| Sample rate | 100% at 16,000 Hz | all files consistent; no resampling needed |
| Duration — median | 1.0 s | |
| Duration — fraction exactly 1 s | 91.34% | |
| Duration — fraction shorter than 1 s | 8.66% | zero-padded at the end (handled in `fix_length`) |
| Corrupt / unreadable files | 0 | none found; the whole subset was readable |
| Fully-zero files | 0 | |
| Files shorter than 0.4 s | 21 (0.05%) | **kept** — `fix_length` zero-pads them to 1 s, and at 0.05% of the data they cannot shift class balance or any metric in either direction. They are left in so the manifest stays a faithful mirror of the official dataset rather than a hand-filtered version of it |
| Amplitude — median absolute peak | 0.52 | |
| Amplitude — fraction with clipping | 1,577 (3.41%) | **kept** — a manual listen to a sample of `flag_clipping=True` files showed these are almost entirely loud, close-mic exclamations of short keywords ("STOP!", "YES!"), a known property of Speech Commands recorded on uncontrolled consumer microphones. This is natural recording variation, not a data error, and dropping 3.4% of the data would not meaningfully change class balance |
| Low-energy (RMS < 0.005) | 705 (1.52%) | almost all of this (702/705) is the `silence` class behaving exactly as intended — only 3 files among the 10 keyword classes are flagged (`off`: 1, `yes`: 2), so this is not a word-class problem |
| Loudness outliers (outside 1.5×IQR on log₁₀RMS) | 796 (1.72%) | **kept** — this threshold flags natural loudness variation across volunteers' microphones and speaking style, not corrupted data |

Total flagged files: **2,367**. No file was dropped in phase 1; every flag above was
reviewed and resolved as natural recording variation, so the manifest row count stays
46,254 and class balance is untouched.

**Related figure:** `figures/phase1/fig04_duration_distribution.png`

---

## 5. Speaker audit

Source: `reports/phase1/audit_per_speaker.csv`

| | train | val | test |
|---|---|---|---|
| Speaker count | 2,027 | 247 | 240 |
| Samples per speaker — min | 1 | 1 | 1 |
| — p25 | 8 | 8 | 8 |
| — median | 11 | 11 | 12 |
| — p75 | 14 | 15 | 15 |
| — max | 129 | 61 | 129 |
| Gini coefficient | 0.461 | 0.445 | 0.456 |

**Interpretation.** This is the most consequential table in the report for this
project. The minimum samples per speaker is **1** — some speakers contributed a single
clip to a given split — and the median is only 11. This directly caps how far the "many
samples per speaker" arm of the phase-3 budget experiment can go: a design asking for
k = 25 samples per speaker is simply not satisfiable from most of the pool, so the
experiment's k axis has to stay inside a range the data can actually supply. The Gini
coefficient of ~0.45–0.46 across all three splits indicates meaningful but not extreme
imbalance, and the fact that it is nearly identical in train, val and test means the
imbalance is a property of how the dataset was collected, not an artifact of the split.

**Related figures:** `figures/phase1/fig02_samples_per_speaker.png` and
`figures/phase1/fig03_speakers_vs_budget.png`

Figure 3 makes the design space explicit: for each budget, as the per-speaker sample cap
(k) rises, the number of speakers that can be covered falls. This curve is what
determines the practical range of k tested in phase 3.

### 5.1 Speaker-leakage verification

Output of `src/data/build_subset.py` (full dataset run):

```
speaker overlap train∩val  : 0
speaker overlap train∩test : 0
speaker overlap val∩test   : 0
noise-file overlap         : 0
```

All four are zero — the single most important validity condition for this entire
project. The check runs on every build and exits non-zero on any overlap, so a future
change to the subset logic cannot silently reintroduce leakage.

---

## 6. Noise and silence check

**Background-noise files.** The dataset has 6 long files in `_background_noise_`
(`doing_the_dishes`, `dude_miaowing`, `exercise_bike`, `pink_noise`, `running_tap`,
`white_noise`). These are not one-second clips and have no word label.

**How they enter the classification problem.** The `silence` class is built from
1-second clips cut from these files. In a real KWS application, this class plays a
"reject" role: the device must recognize that no keyword was said, or it would trigger
constantly.

**Recorded decision on noise leakage.** Noise files are split **at the file level**
between train/val/test, not at the clip level. If two clips from the same noise file
landed in train and test, the model could memorize that specific noise pattern and
inflate test accuracy.

Noise-file assignment (from `reports/phase1/subset_summary.json`, key
`silence_noise_file_assignment`):

```
train: pink_noise.wav, doing_the_dishes.wav, exercise_bike.wav, dude_miaowing.wav
val  : running_tap.wav
test : white_noise.wav
```

### 6.1 The noise files are not interchangeable — and the split makes that matter

Splitting noise at the file level removes leakage, but it has a second consequence that
only shows up once the clips are measured. The six noise files differ enormously in
level, and the split distributes them unevenly:

| Source file | Split | Clips | Median RMS |
|---|---|---|---|
| `dude_miaowing.wav` | train | 769 | 0.0021 |
| `exercise_bike.wav` | train | 769 | 0.0218 |
| `doing_the_dishes.wav` | train | 769 | 0.0331 |
| `running_tap.wav` | val | 370 | 0.0685 |
| `pink_noise.wav` | train | 770 | 0.3150 |
| `white_noise.wav` | **test** | 407 | 0.3329 |

The entire test `silence` class comes from `white_noise.wav` at a median RMS of 0.333.
Of the four train noise sources, only `pink_noise.wav` is comparable; the other three are
**10× to 150× quieter**. So roughly three quarters of the training `silence` examples are
quiet, while 100% of the test `silence` examples are loud broadband noise.

This is the single biggest generalization risk identified in phase 1, and it is a
property of the split design rather than a data defect. Two things follow for phase 3.
First, test `silence` performance measures generalization to an unseen *and much louder*
noise type, so a weak `silence` score should not be read as the model failing at the
reject task in general. Second, it is now clear why the per-clip energy normalization in
phase 2 matters: without it, the model could separate train `silence` from speech on
loudness alone and then collapse at test time, where that shortcut inverts. The phase-3
confusion matrix should be read with a per-noise-source breakdown rather than a single
`silence` row.

**Low-energy clips in word classes.** Only **3** files among the 10 keyword classes
fall below the RMS threshold (`off`: 1, `yes`: 2); the remaining 702 of the 705
low-energy files are in the `silence` class itself, which is exactly the expected
behavior — silence clips *should* have low energy. With only 3 affected word-class
files out of 46,254, this is not a meaningful source of "silence = this word" confusion
and these 3 files are kept.

**Related figure:** `figures/phase1/fig05_rms_by_class.png`

---

## 7. Univariate exploratory analysis

### 7.1 Class distribution

**Figure:** `figures/phase1/fig01_class_distribution.png`

| Class | train | val | test | total |
|---|---|---|---|---|
| yes | 3,228 | 397 | 419 | 4,044 |
| no | 3,130 | 406 | 405 | 3,941 |
| up | 2,948 | 350 | 425 | 3,723 |
| down | 3,134 | 377 | 406 | 3,917 |
| left | 3,037 | 352 | 412 | 3,801 |
| right | 3,019 | 363 | 396 | 3,778 |
| on | 3,086 | 363 | 396 | 3,845 |
| off | 2,970 | 373 | 402 | 3,745 |
| stop | 3,111 | 350 | 411 | 3,872 |
| go | 3,106 | 372 | 402 | 3,880 |
| unknown | 3,077 | 370 | 407 | 3,854 |
| silence | 3,077 | 370 | 407 | 3,854 |

The classes are close to balanced: the largest total class (`yes`, 4,044) is only
1.086× the smallest (`up`, 3,723). At that ratio, class weighting is unnecessary and
macro-F1 is a sufficient headline metric on its own — macro-F1 and accuracy will not
diverge for distributional reasons here, so any later gap between them will be real
per-class behavior rather than an imbalance artifact.

### 7.2 File-length distribution

**Figure:** `figures/phase1/fig04_duration_distribution.png`

The distribution is a single hard spike at exactly 1.0 s (91.34% of files) with a thin
left tail (8.66% shorter, down to a floor of 21 files under 0.4 s) and nothing above
1.0 s. This is the signature of a fixed-length recording widget rather than free
recording, and it is the reason a single `fix_length` zero-pad to 16,000 samples is
enough preprocessing for duration — no resampling, no trimming, no variable-length
batching needed anywhere in the pipeline.

### 7.3 Signal-energy distribution

**Figure:** `figures/phase1/fig05_rms_by_class.png`

Energy is reported on a log₁₀RMS axis. The ten keyword classes are almost
indistinguishable from each other: all ten medians fall between 0.055 and 0.065 RMS and
their boxes overlap nearly completely, so loudness carries essentially no information
about which word was said. The spread within each keyword class is wide (IQR roughly
0.034–0.092) and is driven by microphone and speaking-volume variation, which is why a
per-clip energy normalization is applied before features in phase 2, with the
normalization statistics computed on train only.

The result that matters here is the one that contradicts the obvious expectation.
`silence` does **not** sit cleanly below the keywords. Its distribution is far wider than
any keyword class (IQR 0.020–0.293 against 0.034–0.092) and it overlaps the keywords
heavily in both directions: **41.4% of `silence` clips are louder than the median keyword
clip**, and 24.6% of keyword clips are quieter than the median `silence` clip. Energy
alone therefore cannot separate speech from non-speech in this subset, and the `silence`
class cannot be treated as "the quiet class." Section 6.1 explains where that width comes
from.

### 7.4 Samples-per-speaker distribution

**Figure:** `figures/phase1/fig02_samples_per_speaker.png`

The distribution is **bimodal**, not a simple right-skewed tail. The main mode sits at
10–15 clips (757 of 2,031 train speaker ids) with a lower shoulder at 5–10 (530) and
under 5 (254). A second, clearly separated population sits at 40–55 clips (311 speakers),
with a thin tail beyond that to a maximum of 129. In total **17.4% of speakers have 40 or
more clips while 12.5% have fewer than 5** — the pool is effectively two groups of
contributors, occasional and prolific, rather than one continuous distribution.

Concentration follows from that shape: the top 10% of speakers hold 37.5% of the train
clips and the top 20% hold 59.9%, which is what the Gini of 0.461 summarizes.

This directly constrains the phase-3 design. A high-k condition (k = 40 or above) can
only be served by that second population, so it would compare "many speakers" against
"few speakers *of a specific, unrepresentative kind*" rather than against a clean
samples-per-speaker contrast. Any k above roughly 15 silently changes which speakers are
eligible, and the experiment has to either stay below that point or report the
composition shift explicitly.

---

## 8. Multivariate exploratory analysis

### 8.1 Signal energy × class

**Figure:** `fig05_rms_by_class.png`

Two readings, and the second is the useful one. Among the 10 keywords there is no energy
separation at all — the boxes overlap almost completely — so energy carries no word
information and any classifier that works will be working from spectral structure, not
loudness. And `silence`, which should have been the one class energy *could* separate,
is the class energy separates worst: it spans a range wider than all ten keywords
combined (section 7.3). The practical consequence is that the `silence` class has to be
learned from spectral shape like every other class; a loudness threshold would both miss
loud noise and reject quiet speech.

### 8.2 Spectral content × class

**Figure:** `fig08_mean_logmel_per_class.png` (all twelve panels share one color scale,
so brightness is comparable across classes)

`silence` is immediately distinct: flat and bright across every mel band with no
time structure, which is what a stationary broadband noise source looks like after
averaging. Part of that brightness is an artifact of averaging rather than a pure class
property — keyword clips place the word at different offsets, so averaging over a class
smears the word energy out, while stationary noise averages to itself. The shape
difference (flat and time-invariant vs. a concentrated time-localized blob) is the real
signal here, not the absolute brightness.

Among the keywords, the pairs whose averaged patterns look closest are **go / no /
down** (all three dominated by a low-band blob in the middle third of the clip) and
**on / off / up** (compact mid-band energy with a shorter duration). This is a testable
prediction rather than a conclusion: these pairs should carry the highest off-diagonal
mass in the phase-3 confusion matrix, and the final report can state whether the
prediction held.

### 8.3 Audio profile × speaker (groundwork for phase-2 clustering)

**Figure:** `fig09_speaker_mfcc_pca.png`

**How the figure is built, and its one bias.** Each point is one speaker, profiled as the
mean and standard deviation of 13 MFCC coefficients over 8 of that speaker's clips. The
figure shows **250 speakers, selected as the 250 with the most clips**, not a random
sample of the 2,027. That keeps the profiles comparable (every profile is built from the
same number of clips) but it means the figure describes the prolific second population
identified in section 7.4, not the pool as a whole. Phase 2 repeats this on a random
sample before drawing any conclusion about the shape of speaker space.

**Explained variance:** PC1 = 16.7%, PC2 = 11.2% (combined ≈ 27.9%). This is low for a
2D projection, and it is the most important caveat in this section: roughly 72% of the
variance in per-speaker MFCC profile space is not shown in the figure at all. Whatever
the 2D scatter looks like, it cannot be read as the shape of speaker space, and a
clean-looking plot here would not be evidence that speakers fall into discrete groups.

This matters directly for phase 2. If the profile space has no dominant low-dimensional
cluster structure, then k-means on two or three components is the wrong tool for speaker
selection, and a strategy that maximizes coverage in the full space (for example maximin
/ farthest-point selection) is a better match for the actual geometry. Phase 2 therefore
evaluates cluster quality with silhouette score before committing to k-means, and treats
"no usable cluster structure" as a reportable result rather than a failure.

---

## 9. Defining data-leakage risk

**The correct split unit is `speaker_id`** — not the individual file.

| # | Risk | Mechanism | Countermeasure |
|---|---|---|---|
| 1 | Speaker leakage | the model learns to recognize the person's voice instead of the word | Google's official split (hashed on speaker_id) + automatic verification in `build_subset.py` |
| 2 | Leakage via clustering | if speaker profiles are computed from the whole dataset (including test), the speaker-selection criterion has test information | clustering computed on train pool only |
| 3 | Noise-file leakage | clips from the same noise file in train and test | noise split at the file level |
| 4 | Augmentation leakage | an augmented version of a clip in train, the original in test | augmentation applied only after the split, only to train |
| 5 | Preprocessing-statistics leakage | normalization mean/std computed from the whole dataset | normalization statistics computed from train only, applied to val/test |

Risks 4 and 5 cannot occur yet, since phase 1 performs no augmentation and trains no
model. They are logged here because both are decisions that have to be made before the
first training run, not after.

---

## 10. Defining the baseline

**Model baseline (phase 3):** multiclass logistic regression on a fixed-length feature
vector = `[time-mean of 13 MFCC coefficients ‖ time-std of the same 13 coefficients]`
(26 dimensions), no augmentation.

**Why this baseline:** it discards all temporal ordering information. If a CNN on the
spectrogram does better, we know the improvement genuinely came from modeling
time-frequency structure.

**Second (even simpler) baseline:** always predicting the most frequent class — to show
the absolute floor. With 12 roughly balanced classes, this model's macro-F1 will be
about **1.3%** (the most frequent test class, `up`, has 425/4,888 samples; its own F1
is ≈16.0%, every other class scores 0, and the macro-average over 12 classes is
≈16.0%/12).

**Sampling-strategy baseline (specific to this project):** **random, uniform** speaker
selection. The cluster-driven strategy must beat this, with uncertainty reported across
several seeds.

---

## 11. Three initial questions and the final choice

| # | Question | Feasible with this data? | Decision |
|---|---|---|---|
| 1 | Under a fixed recording budget, is more speakers better than more samples per speaker? And can cluster-driven speaker selection beat random? | Yes — `speaker_id` and a speaker-independent split are available | ✅ **main question** |
| 2 | What is the best model architecture for lightweight KWS? | Yes | ❌ rejected — purely a model comparison, and it supports no decision beyond itself |
| 3 | Does recording-environment noise reduce accuracy? | Incomplete — per-file SNR isn't recorded | ❌ rejected as the main question; moved to phase 4 as a synthetic-noise analysis |

**Reasoning for question 1:** it is the only question that (a) supports a real
operational decision (data-collection design), (b) is fully answerable with the
available data since `speaker_id` exists, (c) naturally incorporates an unsupervised
component (speaker clustering) rather than a tacked-on one, and (d) is CPU-feasible
because it needs many small runs rather than one heavy training job.

---

## 12. Summary and phase-2 next steps

Phase 1 changed the design of the later phases in three concrete ways, and each came from
a measurement that contradicted an assumption rather than confirming one.

First, samples per speaker are **bimodal**, not merely skewed: a main mode at 10–15 clips
and a separate population at 40–55, with 12.5% of speakers under 5 clips and 17.4% at 40
or more. The budget experiment cannot push k past roughly 15 without silently switching
which speakers are eligible, so a high-k arm would no longer be a clean
samples-per-speaker contrast. The k axis is constrained accordingly, and Figure 3 fixes
the range.

Second, the `silence` class is not the quiet class. Its energy spread is wider than all
ten keywords combined, 41.4% of its clips are louder than the median keyword, and the
cause is structural: the source noise files differ by up to 150× in level, and the split
puts the loudest one (`white_noise.wav`, median RMS 0.333) alone in test while three of
the four train sources are far quieter. This is the largest generalization risk found in
phase 1, it is a consequence of the split design rather than a data defect, and it makes
the phase-2 per-clip energy normalization load-bearing rather than routine.

Third, the per-speaker MFCC profile space showed no dominant low-dimensional structure
(PC1 + PC2 ≈ 27.9%), which weakens the original assumption that speakers fall into clean
clusters — and the figure that showed it was itself built from the 250 most prolific
speakers, so it does not yet describe the pool. Phase 2 therefore re-runs the projection
on a random sample and tests cluster structure with silhouette score before relying on
k-means, keeping a coverage-based selection strategy as the alternative.

On the data-quality side the outcome was the opposite of a problem: no corrupt files, no
missing values, no duplicates, uniform 16 kHz, and all 2,367 flagged files resolved as
natural recording variation, so no rows were dropped and no quality decision needs
defending beyond the thresholds themselves.

**Phase-2 next steps:** a reproducible preprocessing pipeline; log-mel/MFCC feature
construction with recorded parameters; dimensionality reduction and clustering of
speaker profiles with k chosen via silhouette score; and checking cluster stability
across seed and subsample variation.

---

## Appendix A — reproducing this report

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data/download_dataset.py
python src/data/build_subset.py
python src/data/audit_quality.py
python src/eda/run_eda.py
```

Seed used: `1405` (recorded in `configs/phase1.yaml`).
Code version: tag `phase-1`, commit `bd4fe7f`.

## Appendix B — generated outputs

| File | Content |
|---|---|
| `reports/phase1/download_manifest.json` | URL, size, SHA256 and download time |
| `reports/phase1/subset_summary.json` | subset stats, split assignment, leakage verification |
| `reports/phase1/audit_per_file.csv` | signal metrics for every file |
| `reports/phase1/audit_flags.csv` | flagged files only, with reasons |
| `reports/phase1/audit_per_speaker.csv` | per-speaker sample counts by class |
| `reports/phase1/audit_summary.json` | numeric audit summary |
| `reports/phase1/eda_manifest.json` | parameters and figure list |
| `figures/phase1/fig01..fig09*.png` | the 9 figures |
