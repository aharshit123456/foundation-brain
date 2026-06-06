# Foundation Brain — Design Spec
**Date:** 2026-06-06  
**Status:** Draft

---

## Goal

Build a task-agnostic contrastive pretraining backbone for simultaneous EEG + fNIRS signals, designed to be frozen and fine-tuned on arbitrary downstream tasks. This is the first EEG+fNIRS framework explicitly framed as a reusable embedding model rather than a task-specific classifier.

**Novelty claim:** All prior work (DC-AGIN, EFRM, ASAC-Net) produces task-specific models. None produce a general-purpose pretrained backbone evaluated across heterogeneous downstream tasks via linear probing and few-shot fine-tuning.

---

## Project Phases

```
Phase 1 — Baseline Reproduction
  Find open-source implementations of DC-AGIN, EFRM, ASAC-Net
  Run on Shin 2017 dataset
  Read papers in parallel to understand internals
  Record numbers in a comparison notebook

Phase 2 — Novel Backbone Design
  Informed by Phase 1 findings
  CLIP-style symmetric InfoNCE pretraining
  Evaluated on two downstream tasks via linear probe + few-shot fine-tuning
```

Phase 2 architecture decisions are deferred until Phase 1 produces empirical evidence.

---

## Dataset

**Shin et al. 2018 — Simultaneous EEG and NIRS during Cognitive Tasks**  
Published in *Scientific Data*, Nature. DOI: 10.1038/sdata.2018.3  
Data repository: http://dx.doi.org/10.14279/depositonce-5830  
MATLAB analysis code: https://github.com/JaeyoungShin/simultaneous_EEG-NIRS

### Recording specs

| Modality | Device | Channels | Sampling Rate |
|---|---|---|---|
| EEG | BrainAmp (Brain Products) | 30 active electrodes (10-5 system) + 4 EOG | 1000 Hz |
| NIRS | NIRScout (NIRx) | 36 channels (16 sources × 16 detectors) | 10.4 Hz |

- **Subjects:** 26 healthy right-handed participants (9M / 17F, age 26.1 ± 3.5)
- **NIRS signals:** HbO and HbR (deoxy/oxy-hemoglobin) via Beer-Lambert law
- **Total data size:** ~6.41 GB (MATLAB-compatible format)

### Tasks (three datasets)

| Dataset | Task | Trials | Trial structure |
|---|---|---|---|
| A | n-back (0-, 2-, 3-back) | 180 per n-back level | 2s instruction + 40s task + 20s rest |
| B | DSR — discrimination/selection response | 180 | 2s instruction + 40s task + 20s rest |
| C | WG — word generation vs. baseline | 60 | 2s instruction + 10s task + 13–15s rest |

### Downstream task mapping

- **Task 1 (cognitive load):** n-back difficulty classification (0-back vs 2-back vs 3-back) from Dataset A
- **Task 2 (cognitive state):** WG vs baseline classification from Dataset C

Both tasks are available in a single dataset from the same subjects with simultaneous EEG+NIRS recordings — no second dataset required.

### Preprocessing (following paper's protocol)

**EEG:**
- Downsample to 200 Hz
- Bandpass filter 1–40 Hz (6th order zero-phase Butterworth)
- Ocular artifact removal via AAR toolbox (iWASOBI)
- Epoch: −5s to end of task period
- Baseline correction: −5s to −2s

**NIRS:**
- Raw optical intensity → deoxy/oxy-hemoglobin via Beer-Lambert
- Downsample to 10 Hz
- Low-pass filter 0.2 Hz (6th order zero-phase Butterworth)
- Epoch: −5s to end of task period
- Baseline correction: −5s to −2s

---

## Phase 1 — Baseline Reproduction

### Target baselines

| Model | Paper | Year | Task | Open source? |
|---|---|---|---|---|
| EFRM | Computers in Biology and Medicine | 2025 | EEG+fNIRS classification | To be found |
| DC-AGIN | Brain Sciences (MDPI) | 2026 | Cross-subject emotion | To be found |
| ASAC-Net | Information Fusion | 2026 | Cross-subject motor imagery / emotion | To be found |

### Execution plan per baseline

1. Search GitHub + Papers With Code for open-source implementation
2. If found: clone, adapt minimally to run on Shin 2017, record metrics
3. If not found: note it, deprioritize, skip unless critical
4. Read paper in parallel while running code
5. Document: what the model learns, where it fails, design decisions

### Notebooks

```
notebooks/
├── 01_dataset_exploration.ipynb     ← load data, verify shapes, visualize signals
├── 02_baseline_efrm.ipynb           ← EFRM reproduction
├── 03_baseline_dcagin.ipynb         ← DC-AGIN reproduction
├── 04_baseline_asacnet.ipynb        ← ASAC-Net reproduction
├── 05_results_comparison.ipynb      ← side-by-side metric tables, failure analysis
└── 06_backbone_design.ipynb         ← Phase 2 starts here
```

### Evaluation protocol for all baselines

- **Split:** Leave-One-Subject-Out (LOSO) — subject-independent, paper-quality
- **Metrics:** classification accuracy, F1-score
- **Tasks:** run each baseline on both downstream tasks where possible

---

## Phase 2 — Novel Backbone (Outline, to be detailed after Phase 1)

### Core idea

CLIP-style symmetric contrastive pretraining:

```
EEG segment   →  EEG Encoder   →  e_eeg   →  Projection Head  →  z_eeg
fNIRS segment →  fNIRS Encoder →  e_fnirs →  Projection Head  →  z_fnirs

Loss: symmetric InfoNCE (NT-Xent)
  L = 0.5 * L(EEG→fNIRS) + 0.5 * L(fNIRS→EEG)

Positive pairs: same subject, same trial, same time window
Negative pairs: all other windows in the batch
```

After pretraining: discard projection heads. Use `e_eeg` and `e_fnirs` as pluggable embeddings.

### Evaluation

- **Linear probe:** freeze backbone, train linear classifier on top — measures representation quality
- **Few-shot fine-tuning:** unfreeze backbone, fine-tune with small labeled set
- **Cross-modal retrieval:** given an EEG embedding, retrieve the matching fNIRS embedding
- **Two downstream tasks:** n-back cognitive load (Dataset A) and word generation (Dataset C)

### Compute constraints

- Single consumer GPU or Google Colab
- Architecture choices deferred to after Phase 1 — will be informed by what existing models use and where they fail

### Ablations (paper contributions)

- Swap InfoNCE for VICReg/Barlow Twins — does loss choice matter?
- With vs without cross-subject contrastive term (as in DC-AGIN)
- Unimodal EEG-only vs EEG+fNIRS joint backbone — does fNIRS actually help?

---

## Project Structure

```
foundation-brain/
├── notebooks/
├── data/
│   └── raw/                  ← Shin 2017 dataset downloaded here
├── baselines/                ← cloned open-source repos
├── docs/
│   └── superpowers/specs/
│       └── 2026-06-06-foundation-brain-design.md
└── requirements.txt
```

---

## Out of Scope

- MEG modality (Phase 3, future work)
- Public GitHub / reproducibility packaging (deferred)
- Weights & Biases or MLflow tracking (notebooks only for now)
- Fine-tuning infrastructure beyond simple linear probe (deferred to after Phase 1)
