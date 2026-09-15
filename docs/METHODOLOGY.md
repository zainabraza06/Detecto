# Detecto: Real-Time Weapon and Violence Detection for CCTV Surveillance

## Problem

Closed-circuit television is nearly ubiquitous in public and private spaces, yet it
functions almost entirely as a passive, after-the-fact record. Footage is reviewed
once an incident is already reported — rarely watched live, rarely used to intervene
while something is still happening. The gap is not camera coverage; it's active
monitoring. A system that can watch a live feed, recognize the visual signatures of
violence or weapon presence as they occur, and surface that to a human fast enough
to act on it, would close a real and largely unaddressed gap between "recording" and
"prevention."

Building that system reliably is harder than it first appears:

- **CCTV footage is a distinct visual domain**, not a harder version of ordinary
  video. Fixed camera angles, compression artifacts, small and distant subjects,
  inconsistent lighting, and low frame rates all degrade the performance of models
  trained on cleaner data — a pattern confirmed repeatedly in this project's own
  experiments, not just assumed from the literature.
- **Weapons are small, often occluded, and easily confused with ordinary objects**
  (phones, bottles, tools), making them one of computer vision's persistently hard
  small-object detection problems.
- **Any system meant to alert on danger must be conservative about false alarms
  while still catching true threats** — a tension that shapes almost every design
  decision in this project, from architecture choice to threshold tuning.
- **The system needs to run on inexpensive, everyday hardware** — a single home or
  shop camera, not a data-center GPU cluster — if it's ever going to be deployable
  beyond a research demo.

## Research gap and novelty

A systematic review of 60+ papers (2020–2025) across weapon detection, violence
detection, and video anomaly detection revealed a specific, recurring pattern: no
existing system learns a shared representation that jointly reasons about weapon
presence and violent motion. Every system found in the literature — including the
creators' own follow-up work on the SCVD dataset, which was explicitly built to
enable this — treats weapon detection and violence detection as separate problems,
either run independently or combined only through late, rule-based fusion after
the fact.

This project doesn't just identify that gap from reading papers — it demonstrates
it empirically, three separate times, with three different weapon detectors (a
generic COCO-pretrained model, an off-the-shelf surveillance-trained model, and a
model fine-tuned specifically on CCTV-domain weapon imagery). In every case, a
rule-based fusion cascade ("if the violence classifier says Violence AND a weapon
is detected, escalate to Weaponized") reduced overall system accuracy rather than
improving it, by converting correctly-classified violent clips into false weapon
alerts. This is a genuinely novel, reproducible finding: the obvious architecture
for combining these two signals actively hurts performance, which motivated the
redesign described below.

A second, independent contribution emerged during dataset validation: the primary
training dataset (SCVD) contains an undocumented duration confound — clip length
alone (with zero visual information) predicts the correct class label with 81.7%
accuracy, because Normal clips run roughly 4 seconds, Weaponized clips roughly 7
seconds, and Violence clips 9-11 seconds. This confound is absent from the
dataset's original publication and from every subsequent paper using it that this
review could find. Left uncorrected, it would let a model partially "cheat" by
learning clip duration instead of visual content — undermining any accuracy claim
built on this dataset without controlling for it.

## Dataset

**Primary training and evaluation dataset:** Smart-City CCTV Violence Detection
(SCVD) — 481 real CCTV clips across Normal (246), Violence (111), and Weaponized
(124) categories, the first public dataset combining generalized and
weapon-involved violence in a CCTV-specific setting.

**Weapon-localization training data:** merged from five independent sources to
address a demonstrated class-imbalance weakness (knife detection recall as low as
15% in early fine-tuning) — US-Seville Mock Attack (5,149 real CCTV images, 3
cameras), US-Seville Unity Synthetic (2,500 images), and four Roboflow Universe
datasets (CCTV Gun Detector, Gun and Knife Detection System, Gun and Knife
Detection, Weapon Detection). Class taxonomies were reconciled across sources into
a unified 3-class scheme (Handgun, Knife, Rifle), with out-of-scope classes
(Person, Bazooka, Grenade Launcher, Sword) dropped. Final merged set: 11,000+
annotated images.

**Additional violence datasets (optional, gated access):**
- RWF-2000: requires a signed data usage agreement. Download `Agreement Sheet.pdf`
  from the official repository, sign it, and email the scanned copy to
  ming.cheng@dukekunshan.edu.cn. They will respond with a download link.
- NTU CCTV-Fights: requires registering an account at rose1.ntu.edu.sg, submitting
  a request form, and accepting a Release Agreement.
- RLVS (Real Life Violence Situations): freely downloadable from Kaggle
  (`mohamedmustafa/real-life-violence-situations-dataset`), no agreement required.
  Note: YouTube-sourced, not pure fixed-camera CCTV — validate any accuracy gain
  from including it specifically against a real-CCTV held-out set before trusting
  it as a straightforward improvement.

## Methodology

**Preprocessing.** Every clip is reduced to a fixed 12 uniformly-sampled frames
regardless of source duration — the direct fix for the duration confound described
above — with aspect-ratio-preserving resize and center-cropping (rather than a
square squash, which prior work on this dataset found degrades performance on wide
CCTV frames).

**Architecture.** A shared R3D-18 (3D convolutional network, Kinetics-400
pretrained) backbone feeds two independent sigmoid output heads — Violence-present
and Weapon-present — each with its own compact MLP and class-weighted binary
cross-entropy loss, trained on binary labels derived from SCVD's original 3-way
annotation. This multi-head design replaces the single-softmax-plus-fusion-rule
approach shown above to fail, and produces two genuinely independent,
simultaneously-reportable confidence scores rather than one forced,
mutually-exclusive label — matching real-world cases where an event can
legitimately be violent, weapon-involved, both, or neither. A separate YOLOv8
model performs visual weapon localization (bounding boxes) for interface display
only; it is deliberately excluded from the alert decision itself, consistent with
the project's human-verification design principle.

**Validation rigor.** All results are computed on a permanently pinned,
deterministic train/validation split with an explicit zero-overlap assertion
enforced before every training run — implemented after this project discovered
and corrected a split-reproducibility bug that had silently permitted
train/validation leakage, inflating one intermediate result to a spurious 96%
accuracy. This is documented rather than hidden, since catching and correcting it
is itself part of the methodological record.

## Results

- Violence-presence head: AUC 0.999, F1 0.98
- Weapon-presence head: AUC 0.911, F1 0.76 (precision 0.70, recall 0.84 —
  deliberately recall-favoring, appropriate for a system where a missed threat is
  costlier than a dismissible false alarm)
- Head-independence diagnostic: correlation between the two heads' outputs,
  restricted to cases where the true labels disagree, is 0.137 — confirming the
  heads learn genuinely separate signals rather than one duplicating the other
- Duration-only baseline (sanity floor): 81.7% — used throughout to confirm the
  trained model is learning visual/motion content, not exploiting a dataset
  artifact
- 5-fold cross-validation on the earlier single-head baseline: 78.2% ± 2.7%
  (superseded by the multi-head result above, retained here as a documented,
  independently-verified reference point)

## Design principle

Every model output — violence confidence, weapon confidence, localized bounding
boxes — is treated as a candidate for human review, never an autonomous trigger.
This is not a hedge; it is a direct consequence of the project's own experimental
findings: automated fusion of imperfect signals was shown, repeatedly, to
introduce more error than it removes.

## Limitations

- The weapon-presence head is a clip-level contextual classifier (motion and
  scene context), not an object detector; it does not localize the weapon
  in-frame. Localization is handled by a separately-trained model with its own,
  independently measured accuracy.
- Trained primarily on a single dataset (SCVD, 481 clips); genuine cross-camera,
  cross-geography generalization has not yet been rigorously validated against a
  held-out, unrelated CCTV source.
- Real-time performance has been validated in a simulated sliding-window
  pipeline, not yet benchmarked on target edge hardware (Jetson Nano / Raspberry
  Pi class devices).
- Weapon-localization data is class-imbalanced even after merging (Handgun
  heavily overrepresented relative to Knife and Rifle); knife detection remains
  the weakest category.
