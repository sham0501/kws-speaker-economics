# Data

This folder is **deliberately empty of raw data**. The dataset archive is ~2.3 GB and does not belong in version control, so what is documented here instead is the exact route by which the data is obtained and the subset is built.

## Source

| | |
|---|---|
| Name | Google Speech Commands v0.02 |
| Author | Pete Warden (Google) |
| Released | April 2018 |
| License | Creative Commons BY 4.0 |
| Archive URL | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| Official page | https://www.tensorflow.org/datasets/catalog/speech_commands |
| Paper | Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 |
| Archive size | ≈ 2.3 GB |
| Extracted size | ≈ 3.8 GB |

## Rebuilding it

```bash
python src/data/download_dataset.py
python src/data/build_subset.py
```

The first script verifies the archive's SHA256 and records the URL, size and download time in `reports/phase1/download_manifest.json`.

## Layout after running

```
data/
    README.md                        (tracked)
    subset_manifest.csv              (tracked — small, a few MB)
    raw/                             (gitignored)
        speech_commands_v0.02/
            yes/ no/ up/ down/ ...   35 word folders
            _background_noise_/      6 long noise files
            validation_list.txt      Google's official split
            testing_list.txt         Google's official split
            LICENSE
        silence_clips/               generated silence clips
```

## Raw dataset structure

- 105,829 WAV files, 16 kHz, mono, 16-bit PCM, mostly 1 second long
- 35 words, 2,618 speakers
- Naming convention: `<word>/<speaker_id>_nohash_<utterance_index>.wav`
  - `speaker_id` is an anonymous hash
  - `utterance_index` is how many times this speaker has already said this word

## This project's subset

12 classes:

| Group | Content |
|---|---|
| 10 keywords | `yes, no, up, down, left, right, on, off, stop, go` |
| `unknown` | balanced sampling from the remaining 25 words |
| `silence` | 1-second clips cut from `_background_noise_` |

The full rationale for this choice is in `PROJECT_CARD.md` item 1 and `reports/phase1/phase1_report.md` section 2.1.

## Split

The split is **read, not built**: from the official `validation_list.txt` and `testing_list.txt`, which are hashed on `speaker_id` and guarantee each speaker appears in exactly one set. Anything not in either list is train.

The `silence` class has no ready-made files; its split is done **at the level of the source noise file** so that clips from the same noise file never end up split across sets.

`build_subset.py` automatically checks for speaker overlap and noise-file overlap after building the manifest, and exits with a non-zero status if leakage is found.

## `subset_manifest.csv` columns

| Column | Type | Description |
|---|---|---|
| `filepath` | string | path relative to the repo root |
| `label` | categorical | one of the 12 classes — the target variable |
| `word` | categorical | the original word before being mapped to `unknown` |
| `speaker_id` | categorical | anonymous speaker hash; `noise::<filename>` for silence |
| `utterance_index` | integer | repeat count |
| `split` | categorical | `train` / `val` / `test` |
| `source` | categorical | `keyword` / `unknown_sampled` / `silence_generated` |

## Privacy

`speaker_id` is an anonymous hash used only to design the split and analyze generalization. No attempt is made to deanonymize speakers.
