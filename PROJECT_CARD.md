# PROJECT_CARD.md

**Course:** Data Analytics — Fall 2026
**Student:** _____________
**Route:** D — Google Speech Commands; audio analysis and keyword spotting (KWS)
**Logged:** 1405/08/13 (phase 1)

---

## 1. Chosen route and dataset subset

Route **D (audio and KWS)** was chosen. The reference dataset is the official
**`speech_commands_v0.02`** release (Google / TensorFlow Datasets). Given the hardware
constraints (Lenovo ThinkPad T14s, 11th-gen Intel Core i5 CPU, integrated Iris Xe
graphics, no discrete GPU), an **explainable subset** is used rather than the full
dataset:

- The **10 classic keywords** from TensorFlow's official benchmark: `yes, no, up, down, left, right, on, off, stop, go`
- An **`unknown`** class: a random, balanced sample of the dataset's other keywords
  (`bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow, zero..nine`, etc.),
  sized close to one normal class.
- A **`silence`** class: 1-second clips cut from the dataset's official
  `_background_noise_` files.

**Why this subset choice does not compromise validation:**
1) These 10 words are exactly the official TensorFlow Simple Audio Recognition
   benchmark, so results remain comparable to published work; 2) Google's official
   `testing_list.txt` and `validation_list.txt` (already designed, by filename hash, to be
   **speaker-independent**) are kept fully intact and only filtered down to these 12
   classes — so the official split is untouched and validation is not compromised;
   3) because the project's core axis is "speaker-count economics" (not finalizing a
   SOTA model), a smaller and CPU-runnable volume is essential — the experiment needs
   many repeated runs rather than one heavy training job.

---

## 2. Working title

**"Data Economics of Keyword Spotting: Smarter Speaker Selection vs. Random Sampling under a Fixed Recording Budget"**

---

## 3. Real-world problem or decision this analysis supports

A team building a lightweight KWS model (for embedded/CPU deployment) faces a
**limited recording budget** (e.g. a fixed capacity of N short audio samples). That
budget can be spent in two ways: **(a)** more speakers with fewer samples from each
(covering more voice diversity), or **(b)** fewer speakers with more samples from each
(consolidating quality/repetition). This project answers that operational question:
**under a fixed recording budget, how should a data-collection strategy be designed to
maximize generalization accuracy, especially on unseen speakers?** It further
investigates whether **clustering of speakers' audio representations** can select
speakers that are "more diverse" than random selection.

---

## 4. Unit of analysis and unit of prediction

- **Unit of analysis (for the data-economics experiment):** a *budget configuration*
  consisting of (speaker count × average samples per speaker), whose product is
  approximately constant and equals the total budget; each configuration is one point on
  a learning curve.
- **Unit of prediction (for the core KWS model):** a **one-second audio clip** →
  predicted as one of 12 labels (10 keywords + `unknown` + `silence`).

---

## 5. Target domain or population

Adult speakers (mostly English-speaking, US-based) who volunteered audio samples for
the public Speech Commands project; recording environments are mostly browser/mobile,
with variable quality and noise. Findings are generalized within the frame of "data
collection for lightweight, single-language (English) KWS under a limited budget," not
to any language or recording environment in general.

---

## 6. Main analytical or predictive question

**Chosen main question:** under a fixed recording budget (total clip count), does
increasing **speaker count** (reducing samples per speaker) improve keyword-classification
accuracy on **fully unseen speakers (speaker-disjoint test)** more than increasing
**samples per speaker** (with fewer, fixed speakers)? And can selecting speakers via
**clustering of audio representations** (instead of random sampling) reach higher accuracy
with the same budget?

**Two alternative questions set aside in phase 1:**
1. "What is the best model architecture for KWS?" — rejected because it's purely a
   model comparison with limited novelty relative to the data-economics angle.
2. "Does adding background noise during recording reduce accuracy?" — rejected because
   the current dataset has no metadata on the actual recording environment (per-file
   SNR), and answering it would require a different dataset. It remains a supplementary
   analysis in phase 4 (robustness to synthetic noise).

---

## 7. Target variable or output

- **Classifier output:** a class label among 12 (`yes/no/.../stop/go/unknown/silence`) —
  a nominal categorical variable.
- **Main project analysis output (not a label in the data, but the result of a
  simulated experiment):** a curve of *macro-F1 on the speaker-disjoint test set* as a
  function of (speaker count, samples per speaker), for each speaker-selection strategy
  (random vs. cluster-driven).

---

## 8. Initial baseline

- **Model baseline:** linear `LogisticRegression` or `SVM` on time-averaged MFCC
  (13 coefficients, mean + standard deviation over time → a fixed-length feature
  vector), with no augmentation.
- **Sampling-strategy baseline:** **random, uniform** speaker selection from the
  training pool, with no smart criterion — this is the baseline the cluster-driven
  strategy must beat.

---

## 9. Proposed evaluation metrics

- **Macro-F1** and **per-keyword recall** — accuracy alone is insufficient here, because
  in KWS the false-accept rate on `unknown`/`silence` carries most of the practical cost.
- **Confusion matrix** to identify acoustically similar word pairs (e.g. `go`/`no`).
- For the data-economics question: **area under the learning curve (AULC)** over the
  "speaker count" axis at a fixed budget, plus **variance across several seeds**
  (because with few speakers, sampling variance is higher).

---

## 10. Initial train / validation / test split design

The split is built **directly from the official Google files**
(`testing_list.txt`, `validation_list.txt`, distributed with the dataset), filtered down
to the 12 subset classes only. Google's official filename-hashing function is designed
so that **a given speaker is never split across train and val/test**. This split is
verified in phase 1 by an automatic overlap check in `build_subset.py`. For the
data-economics experiments in later phases, controlled subsamples with varying
speaker/sample counts are drawn **from train**, while val/test stay **fixed** across all
conditions so comparisons remain fair.

---

## 11. Most important data-leakage risks

1. **Speaker leakage (the most important risk):** if a speaker appears in both train
   and test, the model can learn to recognize the person's voice instead of the word,
   artificially inflating test accuracy. Mitigation: mandatory use of `speaker_id`
   extracted from the filename (the hash prefix before `_nohash_`) plus Google's
   official split, and an audit script that guarantees zero speaker overlap between
   sets.
2. **Leakage via clustering:** if speaker clustering uses features computed from the
   whole dataset (including val/test), the speaker-selection criterion indirectly draws
   information from test. Mitigation: clustering is computed only on the **train pool**.
3. **Background-noise leakage:** the same background-noise file must not contribute
   clips to both training `silence` and the phase-4 noise-robustness evaluation; noise
   files are split between train/test.
4. **Augmentation-version leakage:** augmentation (time-shift, gain, noise) is applied
   only to train, never to val/test — otherwise an augmented copy of a clip could sit in
   train while its original sits in test.

---

## 12. Proposed unsupervised / pattern-discovery component (phase 2)

**Clustering of speaker-level audio representations** is the core of this project: for
each speaker in the train pool, a profile vector is built (a centroid or statistical
summary of MFCC/log-mel over all of that speaker's samples), then speakers are
clustered with **k-means** or **hierarchical clustering** (number of clusters chosen via
silhouette score). Hypothesis: sampling **one speaker per cluster** (instead of pure
random sampling) achieves better coverage of voice diversity (pitch, accent, speaking
rate) under an equal budget. Cluster stability is checked across seed and subsample
variation, since a clustering that reshuffles on a new seed cannot support a sampling
strategy.

---

## 13. Route-specific advanced component (audio, noise, speaker)

Two advanced components are planned (subject to revision in phase 4):
- **Speaker generalization:** compare performance between seen speakers (if a
  controlled-overlap scenario is defined) and fully unseen speakers — a direct extension
  of the project's main question.
- **Noise robustness:** add several levels of background noise (from
  `_background_noise_` files held out for test) and report the macro-F1 drop at each
  level, separately for the speaker-driven vs. random strategy — to see whether smart
  speaker selection also improves noise robustness, not just clean accuracy.

---

## 14. Ethical, privacy, or bias risks

- **Speaker identity:** the `speaker_id` present in filenames is purely an anonymous
  hashed identifier and is used **only for split design and generalization analysis**;
  no attempt is made at de-anonymization or at inferring real identity, age, gender, or
  any other personal attribute of speakers.
- **Demographic bias:** the dataset mostly contains English-speaking speakers (likely
  with an imbalanced share by dialect/accent/gender); phase 1 examines the per-speaker
  sample distribution but infers no demographic attribute — only the imbalance in
  sample count is noted, not speaker identity.
- **Misuse risk:** a KWS model with a high false-accept rate on `unknown` could cause
  unwanted activation in a real application (a voice assistant); this limitation must be
  explicitly noted in the final report and in the application-discussion section
  (a privacy-sensitive device, since it is always listening).
- **Causal claim:** the finding "more speakers beats more samples" is a
  **correlational/controlled-experiment finding on one specific dataset**, not a
  universal law for every KWS system or language; this limitation is stated in the final
  report's Discussion/Limitations section.
