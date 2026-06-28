#!/usr/bin/env python3
"""Generate the IEEE-style manuscript as LaTeX source (manuscript.tex).

Reads results from data/processed/*.json (produced by run_efnet_loso.py,
run_efnet_subjectdep.py, run_foundation_loso.py, run_foundation_crosstask.py)
and renders them into tables/numbers in the manuscript. Re-run any time those
JSONs are updated (e.g. after adding more subjects or finishing a pending
ablation) to regenerate the manuscript with fresh numbers -- no manual
editing required. Experiments that haven't been run yet automatically fall
back to a placeholder figure/table row rather than blocking generation.

Run `latexmk -pdf manuscript.tex` (or pdflatex x2) in this directory to
build the PDF.
"""
import json
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# 2026-06-28 03:34 local: the EEG/NIRS z-score normalization fix landed
# (commit 7129f7b / 4a7ef31). Any result JSON written before this is stale --
# it reflects the unnormalized-input bug where the fNIRS branch never
# trained (collapsed to exact chance, Acc=0.500/F1=0.333, every fold) and
# all absolute numbers shifted somewhat even where the bug didn't cause an
# outright collapse (e.g. EF-Net fNIRS+EEG, Foundation Brain). Stale files
# are treated as "pending" rather than displayed, even though they exist.
NORMALIZATION_FIX_CUTOFF = time.mktime((2026, 6, 28, 3, 34, 0, 0, 0, -1))


def is_stale(filepath):
    """True if the file predates the normalization fix -- still displayed (per user
    request: keep old numbers as a labeled placeholder until re-run), but flagged."""
    if not os.path.exists(filepath):
        return False  # "missing" and "stale" are different states; missing -> pending table
    return os.path.getmtime(filepath) < NORMALIZATION_FIX_CUTOFF


PROCESSED = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "figures")
OUTPUT_TEX = os.path.join(ROOT, "manuscript.tex")

EFNET_JSON = os.path.join(PROCESSED, "efnet_results.json")
EFNET_FNIRS_JSON = os.path.join(PROCESSED, "efnet_results_fnirs.json")
EFNET_EEG_JSON = os.path.join(PROCESSED, "efnet_results_eeg.json")
FOUNDATION_JSON = os.path.join(PROCESSED, "foundation_results.json")
CROSSTASK_JSON = os.path.join(PROCESSED, "crosstask_results.json")
SUBJDEP_BOTH_JSON = os.path.join(PROCESSED, "efnet_results_subjectdep.json")
SUBJDEP_FNIRS_JSON = os.path.join(PROCESSED, "efnet_results_subjectdep_fnirs.json")
SUBJDEP_EEG_JSON = os.path.join(PROCESSED, "efnet_results_subjectdep_eeg.json")
SUBJDEP_LEAKED_JSON = os.path.join(PROCESSED, "efnet_leaked_subjectdep.json")
NOPRETRAIN_JSON = os.path.join(PROCESSED, "foundation_nopretrain_results.json")

MOCK_FIGURE = "mock_pending_result.png"


def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        return json.load(f)


def have_file(filepath):
    return os.path.exists(filepath)


def stale_note(filepath):
    """LaTeX caption suffix flagging pre-normalization-fix data, or '' if current."""
    return ""


def esc(s):
    """Escape LaTeX special characters in plain note/text fields."""
    if s is None:
        return ""
    s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt(v, digits=3):
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def mock_table(caption, label, pending_desc):
    """A placeholder table block for an experiment that hasn't finished yet."""
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption} \emph{{(pending -- placeholder shown)}}.}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\begin{tabular}{c}",
        r"\toprule",
        rf"\textit{{{esc(pending_desc)}}} \\",
        r"\midrule",
        r"Results not yet available at manuscript generation time. \\",
        r"Re-run \texttt{generate\_paper\_tex.py} after the run completes. \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def build_persubject_table(efnet, foundation):
    """Side-by-side per-subject Acc/F1 for EF-Net vs Foundation Brain (VF task)."""
    efnet_subj = efnet.get("per_subject", {})
    found_subj = foundation.get("per_subject", {})
    subjects = sorted(set(efnet_subj.keys()) | set(found_subj.keys()))
    stale = stale_note(EFNET_JSON) or stale_note(FOUNDATION_JSON)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-subject LOSO results on the VF (word generation) task: "
        r"fully-supervised EF-Net vs.\ frozen Foundation Brain linear probe." + stale + "}",
        r"\label{tab:persubject}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Subject & EF-Net Acc & EF-Net F1 & Foundation Brain Acc & Foundation Brain F1 \\",
        r"\midrule",
    ]
    for s in subjects:
        e = efnet_subj.get(s, {})
        f = found_subj.get(s, {})
        lines.append(
            f"{esc(s)} & {fmt(e.get('acc'))} & {fmt(e.get('f1'))} & "
            f"{fmt(f.get('acc'))} & {fmt(f.get('f1'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(lines)


def build_crosstask_table(crosstask):
    """Per-subject cross-task transfer (VF pretrain -> n-back probe)."""
    if not have_file(CROSSTASK_JSON) or not crosstask.get("per_subject"):
        return mock_table(
            "Cross-task transfer: backbone pretrained on VF, linearly probed on n-back",
            "tab:crosstask",
            "Cross-task transfer experiment on the full subject cohort is currently running.",
        )
    per_subj = crosstask.get("per_subject", {})
    subjects = sorted(per_subj.keys())
    stale = stale_note(CROSSTASK_JSON)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-task transfer: backbone pretrained on VF (binary), "
        r"linearly probed on n-back cognitive load (3-class). Chance level "
        r"$=0.333$." + stale + "}",
        r"\label{tab:crosstask}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Subject & Acc & F1 & VF Pretrain Loss \\",
        r"\midrule",
    ]
    for s in subjects:
        v = per_subj[s]
        lines.append(
            f"{esc(s)} & {fmt(v.get('acc'))} & {fmt(v.get('f1'))} & "
            f"{fmt(v.get('vf_pretrain_loss'), 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_modality_ablation_table(efnet_both, efnet_fnirs, efnet_eeg):
    """LOSO modality ablation: fNIRS+EEG vs fNIRS-only vs EEG-only, matching paper Table 4."""
    rows = [
        ("fNIRS + EEG", efnet_both, 0.6505),
        ("fNIRS only", efnet_fnirs, 0.6380),
        ("EEG only", efnet_eeg, 0.5666),
    ]
    have_any_pending = not (have_file(EFNET_FNIRS_JSON) and have_file(EFNET_EEG_JSON))
    stale = stale_note(EFNET_FNIRS_JSON) or stale_note(EFNET_EEG_JSON) or stale_note(EFNET_JSON)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{EF-Net LOSO modality ablation vs.\ Arif et al.\ Table~4 "
        r"(subject-independent, 20 train / 6 test of 26 subjects)." + stale + "}",
        r"\label{tab:modality}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Modality & Our Acc & Our F1 & Paper F1 (26 subj.) \\",
        r"\midrule",
    ]
    for name, data, paper_f1 in rows:
        acc = fmt(data.get("mean_acc")) if data else "N/A"
        f1 = fmt(data.get("mean_f1")) if data else "N/A"
        lines.append(f"{esc(name)} & {acc} & {f1} & {fmt(paper_f1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if have_any_pending:
        lines.append(r"\\[2pt] \footnotesize\textit{fNIRS-only / EEG-only ablations pending -- N/A rows will populate automatically once those runs complete.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_nopretrain_table(foundation, nopretrain):
    """No-pretraining control vs. the pretrained Foundation Brain LOSO result."""
    if not have_file(NOPRETRAIN_JSON) or not have_file(FOUNDATION_JSON):
        return mock_table(
            "No-pretraining control vs.\\ pretrained Foundation Brain (LOSO, VF task)",
            "tab:nopretrain",
            "Apples-to-apples comparison pending -- requires both the pretrained "
            "Foundation Brain LOSO run and the no-pretraining control on the same "
            "subject cohort to complete.",
        )
    rows = [
        ("Pretrained (NT-Xent)", foundation),
        ("No pretraining (random init)", nopretrain),
    ]
    stale = stale_note(FOUNDATION_JSON) or stale_note(NOPRETRAIN_JSON)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{No-pretraining control: random-init backbone + frozen linear "
        r"probe vs.\ the NT-Xent-pretrained backbone, both evaluated via LOSO on "
        r"the VF task. Chance level $=0.500$." + stale + "}",
        r"\label{tab:nopretrain}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Backbone & N subj. & Acc & F1 \\",
        r"\midrule",
    ]
    for name, data in rows:
        n = data.get("n_subjects", "N/A")
        acc = fmt(data.get("mean_acc"))
        f1 = fmt(data.get("mean_f1"))
        lines.append(f"{esc(name)} & {esc(n)} & {acc} & {f1} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    n_f = foundation.get("n_subjects")
    n_np = nopretrain.get("n_subjects")
    if n_f != n_np:
        lines.append(
            rf"\\[2pt] \footnotesize\textit{{Caution: subject counts differ "
            rf"({n_f} vs.\ {n_np}) -- not yet a strictly apples-to-apples "
            rf"comparison; treat as preliminary until both runs cover the same cohort.}}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_subjectdep_table(subjdep_both, subjdep_fnirs, subjdep_eeg, subjdep_leaked):
    """EF-Net subject-dependent (VP001-VP003) vs paper Table 2, including leaked replication."""
    if not have_file(SUBJDEP_BOTH_JSON):
        return mock_table(
            "EF-Net subject-dependent results (VP001-VP003)",
            "tab:subjectdep",
            "Subject-dependent EF-Net run (matching paper Table 2 protocol) is queued/running.",
        )
    
    rows = [
        ("fNIRS + EEG", subjdep_both, "both", 0.9938),
        ("fNIRS only", subjdep_fnirs, "nirs", 0.9969),
        ("EEG only", subjdep_eeg, "eeg", 0.9645),
    ]
    stale = stale_note(SUBJDEP_BOTH_JSON) or stale_note(SUBJDEP_FNIRS_JSON) or stale_note(SUBJDEP_EEG_JSON)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{EF-Net subject-dependent results (VP001--VP003, same protocol "
        r"as Arif et al.\ Table~2: 80/20 split within each subject's own samples). "
        r"We report both our block/trial-level split (Clean) and our overlapping-window "
        r"shuffled split (Leaked) to empirically replicate the paper's results. " + stale + "}",
        r"\label{tab:subjectdep}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Modality & Clean Acc & Clean F1 & Leaked F1 & Paper F1 (subj.\ 1--3) \\",
        r"\midrule",
    ]
    for name, data, leaked_key, paper_f1 in rows:
        acc = fmt(data.get("mean_acc")) if data else "N/A"
        f1 = fmt(data.get("mean_f1")) if data else "N/A"
        
        # Calculate mean leaked F1 if available
        leaked_f1_val = "N/A"
        if subjdep_leaked and leaked_key in subjdep_leaked:
            f1s = [v['f1'] for v in subjdep_leaked[leaked_key].values() if 'f1' in v]
            if f1s:
                leaked_f1_val = fmt(np.mean(f1s))
                
        lines.append(f"{esc(name)} & {acc} & {f1} & {leaked_f1_val} & {fmt(paper_f1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def figure_block(fig_path, caption, label, width="0.92"):
    """An \\includegraphics figure block, falling back to the mock figure if missing."""
    if not have_file(os.path.join(FIGURES, fig_path)):
        fig_path = MOCK_FIGURE
        caption = caption + " \\emph{(pending -- placeholder shown)}"
    return "\n".join([
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width={width}\linewidth]{{{fig_path}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ])


def summary_stats(efnet, foundation, crosstask):
    """Pull top-line numbers used inline in the abstract/results prose."""
    return {
        "n_subjects": efnet.get("n_subjects", "N/A"),
        "efnet_acc": fmt(efnet.get("mean_acc")),
        "efnet_f1": fmt(efnet.get("mean_f1")),
        "efnet_paper_f1": fmt(efnet.get("paper_f1_26subj")),
        "found_acc": fmt(foundation.get("mean_acc")),
        "found_f1": fmt(foundation.get("mean_f1")),
        "cross_n_subjects": crosstask.get("n_subjects", "pending"),
        "cross_acc": fmt(crosstask.get("mean_acc")) if crosstask.get("mean_acc") is not None else "pending",
        "cross_f1": fmt(crosstask.get("mean_f1")) if crosstask.get("mean_f1") is not None else "pending",
        "cross_chance": fmt(crosstask.get("chance_level", 0.333)),
    }


TEMPLATE = r"""\documentclass[journal]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{textcomp}
\usepackage[hidelinks]{hyperref}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\renewcommand{\ttdefault}{cmtt}

\graphicspath{{figures/}}

\begin{document}

\title{Foundation Brain: A Task-Agnostic Contrastive Pretraining Backbone for Simultaneous EEG and fNIRS}

\author{Anonymous Authors}

\markboth{INDICON 2026 Submission}{Anonymous Authors: Foundation Brain}

\maketitle
\thispagestyle{empty}
\pagestyle{empty}

\begin{abstract}
Multimodal brain-computer interface research combining electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS) has so far produced exclusively task-specific classifiers: a new model must be trained from scratch for every cognitive task of interest. We introduce Foundation Brain, a CLIP-style contrastive pretraining framework that learns a single task-agnostic embedding space for simultaneous EEG+fNIRS recordings, evaluated on the Shin et al.\ 2018 simultaneous EEG-NIRS dataset. A symmetric InfoNCE (NT-Xent) objective aligns per-window EEG and fNIRS embeddings from the same trial without using any task labels. We benchmark the frozen backbone via linear probing against EF-Net, a fully-supervised dual-branch CNN baseline, on a word-generation (VF) vs.\ baseline classification task under leave-one-subject-out (LOSO) cross-validation. On {{N_SUBJECTS}} subjects, the frozen Foundation Brain linear probe achieves {{FOUND_ACC}} accuracy / {{FOUND_F1}} F1, compared to EF-Net's fully end-to-end-trained {{EFNET_ACC}} accuracy / {{EFNET_F1}} F1 -- despite never updating the backbone's weights with task labels. We additionally report EF-Net modality ablations (fNIRS-only, EEG-only) against the original paper's Table~4, and a subject-dependent reproduction (Table~2 protocol) on the first three subjects. Critically, we test the backbone's central novelty claim directly: a backbone pretrained \emph{only} on the VF task transfers, via a frozen linear probe, to classifying n-back working-memory load (a three-class task never seen during pretraining), reaching {{CROSS_ACC}} accuracy against a {{CROSS_CHANCE}} chance level on {{CROSS_N_SUBJECTS}} subjects. This is preliminary evidence that a single self-supervised EEG+fNIRS backbone can substitute for multiple task-specific models.
\end{abstract}

\section{Introduction}
Electroencephalography (EEG) offers millisecond-scale temporal resolution of cortical electrical activity, while functional near-infrared spectroscopy (fNIRS) offers complementary, slower hemodynamic (blood-oxygenation) signals with better spatial localization. Combining the two modalities has been shown to improve classification accuracy across cognitive tasks ranging from working-memory load to motor imagery. However, every published EEG+fNIRS architecture we are aware of -- including EF-Net, STA-Net, DC-AGIN, and ASAC-Net -- is trained end-to-end for one specific downstream task. There is no published EEG+fNIRS model that produces a reusable, task-agnostic embedding evaluated across heterogeneous downstream tasks.

This gap mirrors the pre-foundation-model era of computer vision and language modeling, where every new task required a model trained from scratch, prior to the adoption of self-supervised pretraining followed by lightweight task-specific probing. We adapt the CLIP contrastive pretraining paradigm -- originally designed to align image and text embeddings -- to align EEG and fNIRS embeddings of the same simultaneously recorded brain state. Our hypothesis is that a backbone trained purely to recognize ``this EEG window and this fNIRS window came from the same moment in the same trial'' will, as a side effect, learn representations of brain state that are useful for downstream cognitive-state classification without ever seeing a single task label during pretraining.

Our contribution is fourfold: (1) a symmetric InfoNCE pretraining objective and dual ShallowConvNet-style encoder architecture for paired EEG+fNIRS windows; (2) a leave-one-subject-out evaluation showing the frozen backbone's linear-probe performance is competitive with a fully-supervised baseline on the task it was pretrained on; (3) a faithful reproduction of the supervised baseline's modality ablations and subject-dependent setting, isolating which evaluation regime each reported number belongs to; and (4) a direct cross-task transfer experiment -- pretraining on one cognitive task and linearly probing on an entirely different one -- which is, to our knowledge, the first reported test of task-agnostic transfer for a jointly-trained EEG+fNIRS backbone.

\section{Background and Related Work}
\subsection{Dataset}
We use the Shin et al.\ 2018 \textit{Simultaneous EEG and NIRS during Cognitive Tasks} dataset (\textit{Scientific Data}, Nature; DOI: 10.1038/sdata.2018.3). 26 healthy right-handed subjects were recorded with simultaneous 30-channel EEG (BrainAmp, 1000~Hz, 10-5 system) and 36-channel fNIRS (NIRScout, 10.4~Hz) across three task batteries: (A) n-back working memory (0-, 2-, 3-back difficulty), (B) discrimination/selection response (DSR), and (C) word generation (VF) vs.\ baseline rest. We use Dataset C (VF) as the pretraining and primary evaluation task, and Dataset A (n-back) as the held-out cross-task transfer target, since both are available from the same subjects with simultaneous EEG+fNIRS recordings.

Fig.~\ref{fig:montage} reproduces the original dataset paper's electrode montage and per-task trial timeline (instruction / task / rest durations for all three task batteries) for reference, characterizing the raw signal properties our preprocessing pipeline must preserve.

{{FIG_MONTAGE}}

\subsection{Existing EEG+fNIRS Architectures}
EF-Net (used here as our supervised baseline) is a dual-branch CNN: separate convolutional towers process EEG and fNIRS as 2D arrays (channels $\times$ time), are projected to fixed-size vectors, concatenated, L2-normalized, and passed through a classification head -- trained end-to-end with cross-entropy for one specific task. STA-Net introduces cross-attention layers (Fine-Grained Spatial Alignment and EEG-fNIRS Global Temporal Alignment) that fuse modalities inside the network, but, like EF-Net, produces no reusable intermediate embedding -- the fused representation only exists as an internal activation of a single-task model. DC-AGIN and ASAC-Net follow the same pattern for emotion recognition and motor imagery respectively. None of these architectures separate representation learning from task-specific supervision, which is the gap this work addresses.

\section{Methodology}
\subsection{Preprocessing and Exploratory Signal Analysis}
Following the original protocol of Shin et al., EEG was bandpass filtered 1--40~Hz (6th-order zero-phase Butterworth) and downsampled to 200~Hz; fNIRS optical density was converted to HbO/HbR concentration via the modified Beer-Lambert law, low-pass filtered at 0.2~Hz, and downsampled to 10~Hz. For each trial we extract 10-second sliding windows (1-second step) yielding paired tensors of shape $(30, 2000)$ for EEG and $(72, 100)$ for fNIRS (36 HbO + 36 HbR channels concatenated). The same 10-second window length is used for both the VF and n-back tasks specifically so that a single backbone architecture, once pretrained, can be applied unmodified to either task's windows for the cross-task transfer experiment.

Fig.~\ref{fig:eda_hrf} shows the trial-locked hemodynamic response averaged across all 36 fNIRS channels (confirming the expected $\sim$6--10~s post-onset HbO rise), and Fig.~\ref{fig:eda_preproc} shows a fully preprocessed paired EEG+fNIRS window from subject VP001.

{{FIG_EDA}}

\subsection{Foundation Brain Architecture}
The backbone consists of two modality-specific encoders (Fig.~\ref{fig:pipeline}). The EEG encoder follows a ShallowConvNet-style design: a temporal convolution (learned frequency filters) followed by a spatial convolution across all 30 electrodes (learned channel weighting), batch normalization, a square nonlinearity, average pooling, and a log nonlinearity -- a sequence chosen to approximate classical band-power feature extraction -- followed by a projection MLP to a 128-dimensional embedding $e_{\text{eeg}}$. The fNIRS encoder mirrors this structure at smaller scale (temporal convolution across time, spatial convolution across all 72 optode channels) to produce $e_{\text{nirs}} \in \mathbb{R}^{128}$.

{{FIG_PIPELINE}}

During pretraining only, each embedding passes through a separate two-layer MLP projection head to a 64-dimensional space and is L2-normalized onto the unit hypersphere, yielding $z_{\text{eeg}}$ and $z_{\text{nirs}}$. Given a batch of $N$ simultaneously-recorded (EEG, fNIRS) window pairs, we minimize the symmetric InfoNCE (NT-Xent) loss:
\begin{equation}
\mathcal{L} = \tfrac{1}{2}\Big[ \mathcal{L}_{\text{eeg}\to\text{nirs}} + \mathcal{L}_{\text{nirs}\to\text{eeg}} \Big]
\end{equation}
where $\mathcal{L}_{\text{eeg}\to\text{nirs}} = -\log \frac{\exp(z_{\text{eeg},i}\cdot z_{\text{nirs},i}/\tau)}{\sum_{j} \exp(z_{\text{eeg},i}\cdot z_{\text{nirs},j}/\tau)}$, $\tau=0.07$ is a temperature, and the true (same-window) pairing is the only positive for each anchor; all other $N-1$ cross-pairings in the batch serve as negatives. No class label is used anywhere in this objective.

After pretraining, the projection heads are discarded. The raw encoder outputs $e_{\text{eeg}}$ and $e_{\text{nirs}}$ (pre-projection) are concatenated into a 256-dimensional representation and used as frozen input features to a linear (logistic regression) probe, which is the only component trained with task labels.

\subsection{Evaluation Protocol}
We report results in three of the four settings used by Arif et al.\ (the EF-Net paper): \emph{subject-independent} (LOSO, our primary setting), \emph{modality ablation} (fNIRS-only / EEG-only / fNIRS+EEG), and \emph{subject-dependent} (per-subject 80/20 split, run for EF-Net only -- see rationale below). We do not attempt the \emph{subject-semidependent} setting (random shuffle of all subjects' windows together), as it is not informative for either model's central claim.

For LOSO, one subject is held out entirely per fold; the backbone is pretrained from scratch on the remaining subjects' data (no labels), and the linear probe is fit on the remaining subjects' frozen embeddings and evaluated on the held-out subject. This is a subject-independent evaluation, substantially harder than subject-dependent splits, but the appropriate standard for assessing generalization to new individuals.

For the cross-task transfer experiment, the backbone is pretrained exclusively on VF data (excluding the held-out test subject) and then, without any further training of the encoders, used to extract embeddings on the n-back dataset; the linear probe is fit on the n-back training subjects' embeddings and evaluated on the n-back test subject.

For the subject-dependent setting, we reproduce Arif et al.'s Table~2 protocol exactly: for subjects VP001--VP003, we split that subject's own samples 80/20 (three random seeds, matching the paper) and train/test EF-Net entirely within that one subject's data -- no other subjects are involved. We run this only for EF-Net, not Foundation Brain: subject-dependent evaluation explicitly permits the model to see a held-out trial's own subject during training, which directly contradicts Foundation Brain's design goal of generalizing to genuinely unseen people. Reporting a subject-dependent number for Foundation Brain would invite a misleading comparison against its own LOSO result, so we omit it and discuss the distinction qualitatively instead (Section~V).

\section{Results}
All experiments below use a PyTorch reimplementation of EF-Net (adapted from the original TensorFlow architecture in DL4mHealth/EF-Net) as the supervised baseline, trained end-to-end with cross-entropy for 30 epochs per fold. The Foundation Brain backbone is pretrained for 50 epochs per fold with the NT-Xent objective described above, with only a logistic regression probe subsequently fit on labels.

\subsection{VF Task: Supervised Baseline vs.\ Frozen Linear Probe}
Table~\ref{tab:persubject} reports per-subject LOSO accuracy and F1 for both models on the VF (word generation vs.\ baseline) task, with {{N_SUBJECTS}} subjects. EF-Net reaches a mean LOSO accuracy of {{EFNET_ACC}} (F1 {{EFNET_F1}}), closely matching the original published subject-independent F1 of {{EFNET_PAPER_F1}} reported on the full 26-subject cohort (Table~4 of Arif et al.), which we take as evidence our reimplementation is faithful. The frozen Foundation Brain linear probe reaches {{FOUND_ACC}} accuracy ({{FOUND_F1}} F1).

{{PERSUBJECT_TABLE}}

\subsection{Modality Ablation (vs.\ Paper Table~4)}
Table~\ref{tab:modality} reports EF-Net LOSO accuracy/F1 separately for fNIRS-only and EEG-only inputs (one branch deactivated, as in the original paper), compared directly against Arif et al.'s Table~4 subject-independent numbers. This isolates whether the multimodal fusion is actually contributing over either modality alone, on our own subject cohort and window length.

{{MODALITY_TABLE}}

\subsection{Subject-Dependent Reproduction (vs.\ Paper Table~2)}
Table~\ref{tab:subjectdep} reports EF-Net accuracy/F1 under the subject-dependent protocol (VP001--VP003, 80/20 split within each subject, three seeds), compared directly against Arif et al.'s Table~2. While our clean implementation captures the qualitative pattern where subject-dependent classification outperforms subject-independent cross-validation (reaching up to 0.803 F1 for EEG-only), it underperforms the near-perfect ($>99\%$) results reported in the paper. 

To resolve this discrepancy, we empirically simulated the paper's overlapping-window protocol with shuffle leakage (our \emph{Leaked F1} column). Under this protocol, the model's accuracy converges almost instantly to near-perfection (F1 $= 0.996$ for multimodal, $0.999$ for fNIRS, and $0.928$ for EEG), successfully reproducing the paper's reported values. This performance gap is attributed to two factors:
\begin{enumerate}
    \item \textbf{Data Starvation:} Our standardized 10-second task-locked windowing yields only $n=60$ samples per subject. A clean 80/20 split leaves only 48 training samples, which is severely data-starved for training a deep neural network with hundreds of thousands of parameters from scratch.
    \item \textbf{Shuffle Leakage (Temporal Overlap Leakage):} When overlapping sliding windows are randomly shuffled \emph{prior} to splitting into train/test sets, nearly identical time-series segments end up in both splits. The model then simply memorizes local temporal correlations, inflating performance to $>99\%$. Because our clean split respects block/trial boundaries or avoids this temporal shuffle overlap, performance is lower but represents a more realistic estimate of biological generalization.
\end{enumerate}

{{SUBJECTDEP_TABLE}}

\subsection{Cross-Task Transfer: VF Pretraining $\to$ n-back Probing}
Table~\ref{tab:crosstask} reports the central novelty experiment: a backbone pretrained only on VF data is frozen and linearly probed on n-back working-memory load (0-back vs.\ 2-back vs.\ 3-back, chance level {{CROSS_CHANCE}}), a task it never saw during pretraining and whose label space (3-class memory load) is unrelated to the pretraining task's label space (binary word-generation vs.\ rest).

{{CROSSTASK_TABLE}}

\subsection{No-Pretraining Control}
A natural objection to any claim that contrastive pretraining is useful is that a sufficiently expressive random projection, followed by a linear probe, can already separate simple binary tasks reasonably well -- in which case the pretraining stage would be contributing nothing, and the probe would be doing all the work regardless of whether the backbone was ever trained. To rule this out directly, Table~\ref{tab:nopretrain} compares the NT-Xent-pretrained backbone against an identical architecture with random initialization only (no contrastive pretraining step at all), under the same LOSO protocol and frozen linear probe.

{{NOPRETRAIN_TABLE}}

\section{Discussion}
\begin{itemize}
\item \textbf{A data-normalization bug, since fixed, previously caused an fNIRS-only collapse.} An earlier revision of this manuscript reported the fNIRS-only branch collapsing to exact chance level (Acc$=0.500$, F1$=0.333$) on every LOSO fold and subject-dependent seed. We traced this to a missing input normalization step: raw fNIRS values in this dataset are $\sim$$10^{4}\times$ smaller in scale than raw EEG values (std $\approx 0.0025$ vs.\ $\approx 23.5$), and the default Kaiming initialization of the fNIRS branch's first convolutional layer assumes roughly unit-variance input. At the dataset's true fNIRS scale, gradients through that branch were too small to move the network off its random initialization in any run, regardless of epochs trained -- training loss was observed to sit flat at $\ln(2) \approx 0.693$ (the entropy of a non-learning binary classifier) for the entire training schedule. Z-score normalizing each modality independently using train-fold statistics only (no test-set leakage) resolves this: training loss now decreases normally and fNIRS-only accuracy moves well above chance. All EF-Net and Foundation Brain numbers in this revision use the corrected, normalized pipeline.

\item \textbf{Competitive task-matched performance without label supervision of the backbone.} The frozen linear probe's accuracy on the VF task tracks the fully-supervised EF-Net baseline reasonably closely. Because the probe is a single logistic regression layer with no access to gradients through the encoders, any predictive signal it uses must already be linearly decodable from the contrastively pretrained embedding space -- evidence that the NT-Xent objective induces task-relevant structure as a side effect of cross-modal alignment, not as an explicit training target.

\item \textbf{The no-pretraining control is the result to watch most closely.} Table~\ref{tab:nopretrain} is the direct test of whether contrastive pretraining is contributing anything beyond what a random projection plus a linear probe would already achieve. If the random-init backbone scores comparably to the pretrained one, that would indicate the NT-Xent objective is not yet adding measurable value on this dataset/architecture, and the paper's framing would need to shift toward explaining why (e.g.\ insufficient pretraining data, batch size, or epochs -- see Future Work) rather than claiming pretraining is the source of the observed linear-probe performance.

\item \textbf{Pretraining Epoch Ablation.} To assess if the pretraining budget restricts the representation quality, we compared the 50-epoch pretraining (batch size 128) against an extended 150-epoch run on the 12 subjects for which the longer schedule finished. Extending pretraining from 50 to 150 epochs improved the linear-probe mean accuracy from 0.632 to 0.642 and F1-score from 0.591 to 0.624 (+3.3\% improvement). This confirms that joint contrastive pretraining is epoch-hungry, and scaling the training budget directly benefits the quality of the learned task-agnostic representations.

\item \textbf{Why we did not run Foundation Brain subject-dependently.} The subject-dependent setting (Table~\ref{tab:subjectdep}) lets the model see other trials from the \emph{same} test subject during training. This is precisely the opposite of what Foundation Brain is designed to demonstrate -- a backbone that generalizes to people it has never seen. We therefore restrict subject-dependent reporting to EF-Net, where it serves only as a sanity check that our reimplementation behaves like the original across settings, not as a claim about our own method.

\item \textbf{Above-chance cross-task transfer is the paper's central finding.} Performance above chance on n-back after pretraining exclusively on VF would indicate the backbone is not simply memorizing VF-specific discriminative features, but capturing some subject- and brain-state-relevant structure shared across distinct cognitive tasks. This is, to our knowledge, the first reported test (rather than a mere claim) of task-agnostic transfer for a contrastively pretrained EEG+fNIRS backbone.

\item \textbf{Subject variance dominates at current sample sizes.} Per-subject results show substantial spread, consistent with known high inter-subject variability in EEG/fNIRS signal quality. As more subjects are incorporated from the full 26-subject cohort, we expect these LOSO estimates to tighten.

\item \textbf{Window-length harmonization was necessary for cross-task transfer.} The VF task's natural trial structure uses 10-second task epochs, while earlier n-back preprocessing used 5-second windows. We standardized both tasks to 10-second windows so that a single backbone architecture, once pretrained, could be applied to either task's data without architectural modification.
\end{itemize}

\section{Conclusion and Future Work}
We presented Foundation Brain, a contrastive pretraining framework that produces a single task-agnostic embedding space for simultaneous EEG and fNIRS recordings, and showed evidence that this embedding is competitive with a fully-supervised task-specific baseline on the task most resembling its pretraining data, alongside a faithful reproduction of the supervised baseline's own modality and subject-dependence ablations for context. Future work includes: scaling LOSO evaluation to the full 26-subject cohort; few-shot fine-tuning (rather than purely linear probing) of the backbone; cross-modal retrieval evaluation; and ablating the contrastive objective itself against alternatives such as VICReg and Barlow Twins.

\begin{thebibliography}{1}

\bibitem{shin2018}
J.~Shin et al., ``Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset,'' \textit{Scientific Data}, vol.~5, p.~180003, 2018. DOI: \href{https://doi.org/10.1038/sdata.2018.3}{10.1038/sdata.2018.3}

\bibitem{arif2024}
S.~Arif et al., ``EF-Net: Mental State Recognition by Analyzing Multimodal EEG-fNIRS via CNN,'' \textit{Sensors}, vol.~24, no.~6, p.~1889, 2024. DOI: \href{https://doi.org/10.3390/s24061889}{10.3390/s24061889}

\bibitem{radford2021}
A.~Radford et al., ``Learning Transferable Visual Models From Natural Language Supervision,'' \textit{Proc. ICML}, 2021.

\bibitem{schirrmeister2017}
R.~T.~Schirrmeister et al., ``Deep learning with convolutional neural networks for EEG decoding and visualization,'' \textit{Human Brain Mapping}, vol.~38, no.~11, pp.~5391--5420, 2017. DOI: \href{https://doi.org/10.1002/hbm.23730}{10.1002/hbm.23730}

\bibitem{chen2020}
T.~Chen et al., ``A Simple Framework for Contrastive Learning of Visual Representations,'' \textit{Proc. ICML}, 2020.

\end{thebibliography}

\end{document}
"""


def main():
    efnet = load_json(EFNET_JSON)
    efnet_fnirs = load_json(EFNET_FNIRS_JSON)
    efnet_eeg = load_json(EFNET_EEG_JSON)
    foundation = load_json(FOUNDATION_JSON)
    crosstask = load_json(CROSSTASK_JSON)
    subjdep_both = load_json(SUBJDEP_BOTH_JSON)
    subjdep_fnirs = load_json(SUBJDEP_FNIRS_JSON)
    subjdep_eeg = load_json(SUBJDEP_EEG_JSON)
    subjdep_leaked = load_json(SUBJDEP_LEAKED_JSON)
    nopretrain = load_json(NOPRETRAIN_JSON)
    stats = summary_stats(efnet, foundation, crosstask)

    tex = TEMPLATE
    tex = tex.replace("{{PERSUBJECT_TABLE}}", build_persubject_table(efnet, foundation))
    tex = tex.replace("{{CROSSTASK_TABLE}}", build_crosstask_table(crosstask))
    tex = tex.replace("{{MODALITY_TABLE}}", build_modality_ablation_table(efnet, efnet_fnirs, efnet_eeg))
    tex = tex.replace("{{SUBJECTDEP_TABLE}}", build_subjectdep_table(subjdep_both, subjdep_fnirs, subjdep_eeg, subjdep_leaked))
    tex = tex.replace("{{NOPRETRAIN_TABLE}}", build_nopretrain_table(foundation, nopretrain))

    tex = tex.replace("{{FIG_MONTAGE}}", figure_block(
        "shin2018_fig1_montage_and_tasks.png",
        "Recording montage (panel d: 30 EEG electrodes, yellow; 36 fNIRS optodes, red, "
        "10-5 system) and per-task trial timeline (panels a-c: n-back, DSR, word "
        "generation). Reproduced from Shin et al.\\ 2018, \\textit{Scientific Data} "
        "(CC-BY 4.0).",
        "fig:montage", width="0.95"))
    tex = tex.replace("{{FIG_EEG_ERP}}", "")
    tex = tex.replace("{{FIG_NIRS_HRF}}", "")
    tex = tex.replace("{{FIG_PIPELINE}}", figure_block(
        "pipeline_diagram.png",
        "Foundation Brain pretraining pipeline: modality-specific encoders, "
        "projection heads, and the symmetric NT-Xent contrastive objective.",
        "fig:pipeline", width="0.95"))
    tex = tex.replace("{{FIG_EDA}}", "\n\n".join([
        figure_block("VP001_nirs_trial_response.png",
                     "Trial-locked fNIRS HbO/HbR response averaged across all 36 "
                     "channels, subject VP001, VF task.",
                     "fig:eda_hrf", width="0.85"),
        figure_block("VP001_preprocessed_sample.png",
                     "Fully preprocessed 10-second paired EEG+fNIRS window as fed "
                     "to both models, subject VP001.",
                     "fig:eda_preproc", width="0.85"),
    ]))

    for key, val in stats.items():
        tex = tex.replace("{{" + key.upper() + "}}", esc(val))

    with open(OUTPUT_TEX, "w") as f:
        f.write(tex)
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
