# AI_USAGE.md — AI and Large Language Model Usage Log

**Status:** living document, updated every phase.
**Last updated:** phase 1 — ____________

---

## 1. Personal policy summary

_One paragraph, in your own words: where is the line between what the assistant did and
what you did? Write this yourself — it is the paragraph a TA is most likely to ask you to
expand on._

---

## 2. Tools used

| Tool | Version / model | Period used | Main purpose |
|---|---|---|---|
| | | phase 1 | |
| | | | |

---

## 3. Per-phase usage log

### Phase 1

| # | Task | Tool | Main prompt / protocol | How I validated it |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

The "how I validated it" column is the one that carries weight in the oral defense.
Validation means an independent check — re-running a count against the filesystem,
re-deriving a split check with separate code, comparing feature parameters against
library documentation. "The output looked correct" is not validation.

### Phase 2
### Phase 3
### Phase 4
### Phase 5

---

## 4. Is an LLM/VLM part of the project's methodology?

**No.**

No LLM or VLM has any role in this project's analysis pipeline. The models used are
classical and lightweight (logistic regression/SVM on MFCC as the baseline, a small CNN
on log-mel as the main model), because the target hardware is CPU-only with no discrete
GPU, and the project's focus is data economics, not model scale.

If a pretrained audio model (for example an embedding encoder) enters a later phase, this
section has to be updated, and that model needs its own baseline and ablation like any
other.

---

## 5. Analytical limitations arising from AI use

_Write this yourself, concretely. Things worth considering: generated code can carry
assumptions that only surface on close reading, so note which files you read line by line
versus only tested behaviorally; suggestions about "standard practice" do not always match
the library versions actually installed (the `labels` argument to `boxplot`, removed in
matplotlib 3.11, only surfaced by running the code); and a first suggestion tends to be the
most common approach rather than the most suitable one, which matters in a course that
grades novelty._

---

## 6. Acknowledgment

I understand every analytical decision in this project and can explain it in the oral
defense.

**Name:** ____________  **Date:** ____________
