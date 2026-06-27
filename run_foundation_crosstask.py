"""
Foundation Brain — Cross-Task Generalization Evaluation

Proves the core novelty claim:
  "A backbone pretrained on VF (Dataset C) produces embeddings that are
   linearly separable for n-back cognitive load (Dataset A) — without
   any retraining."

Protocol:
  1. Load VF-pretrained Foundation Brain backbone (retrained here per LOSO fold)
  2. Extract embeddings on n-back windows (frozen backbone, no labels used)
  3. Fit linear probe on n-back embeddings
  4. Report 3-class accuracy / F1 (0-back vs 2-back vs 3-back)

Compare against:
  - EF-Net retrained from scratch on n-back (supervised upper bound)
  - Chance level: 33.3%

Run from project root:
    python run_foundation_crosstask.py

Outputs:
    data/processed/crosstask_results.json
"""

import pickle
import numpy as np
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Reuse architecture from Phase 2
from run_foundation_loso import (
    FoundationBrain, NTXentLoss, EEGNIRSDataset,
    pretrain_one_fold, extract_embeddings, linear_probe, log
)


def run_crosstask_loso(vf_data, nback_data,
                       n_pretrain_epochs=50, batch_size=64,
                       lr=3e-4, temperature=0.07):
    """
    For each LOSO fold:
      - Pretrain backbone on VF data (all subjects except test)
      - Extract embeddings on n-back data (test subject)
      - Linear probe: trained on n-back train subjects, tested on n-back test subject

    This is the cross-task transfer proof:
      pretraining task  = VF (binary)
      evaluation task   = n-back (3-class)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log(f"Using device: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use subjects present in BOTH datasets
    subjects = sorted(set(vf_data.keys()) & set(nback_data.keys()))
    log(f"Subjects with both VF and n-back data: {subjects}")

    results = {}
    fold_bar = tqdm(subjects, desc='LOSO folds', unit='fold', file=sys.stdout)

    for test_subj in fold_bar:
        fold_bar.set_postfix(test=test_subj)
        train_subjs = [s for s in subjects if s != test_subj]

        # ── Step 1: Pretrain backbone on VF (no labels used) ──
        tr_eeg  = np.concatenate([vf_data[s]['eeg']    for s in train_subjs])
        tr_nirs = np.concatenate([vf_data[s]['nirs']   for s in train_subjs])
        tr_lbl  = np.concatenate([vf_data[s]['labels'] for s in train_subjs])

        # Z-score normalize using VF train-fold statistics -- see run_efnet_loso.py for
        # the original diagnosis (NIRS ~1e-3 scale vs EEG ~1e1 scale causes near-zero
        # gradients through an unnormalized NIRS branch).
        vf_eeg_mean, vf_eeg_std = tr_eeg.mean(), tr_eeg.std()
        vf_nirs_mean, vf_nirs_std = tr_nirs.mean(), tr_nirs.std()
        tr_eeg  = (tr_eeg  - vf_eeg_mean)  / (vf_eeg_std  + 1e-8)
        tr_nirs = (tr_nirs - vf_nirs_mean) / (vf_nirs_std + 1e-8)

        vf_train_loader = DataLoader(
            EEGNIRSDataset(tr_eeg, tr_nirs, tr_lbl),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        model = FoundationBrain(embed_dim=128, eeg_shape=tr_eeg.shape[1:],
                                nirs_shape=tr_nirs.shape[1:]).to(device)
        final_loss = pretrain_one_fold(
            model, vf_train_loader, device,
            n_epochs=n_pretrain_epochs, lr=lr, temperature=temperature
        )
        log(f"\n  [{test_subj}] VF pretrain loss: {final_loss:.4f}")

        # ── Step 2: Extract n-back embeddings (backbone frozen) ──
        # Train probe on n-back train subjects
        nb_tr_eeg  = np.concatenate([nback_data[s]['eeg']    for s in train_subjs])
        nb_tr_nirs = np.concatenate([nback_data[s]['nirs']   for s in train_subjs])
        nb_tr_lbl  = np.concatenate([nback_data[s]['labels'] for s in train_subjs])

        nb_te_eeg  = nback_data[test_subj]['eeg']
        nb_te_nirs = nback_data[test_subj]['nirs']
        nb_te_lbl  = nback_data[test_subj]['labels']

        # Normalize n-back data with its own train-fold statistics (not VF's) -- the
        # encoder just needs internally-consistent scale at inference time, and n-back's
        # raw EEG/NIRS amplitude range need not exactly match VF's.
        nb_eeg_mean, nb_eeg_std = nb_tr_eeg.mean(), nb_tr_eeg.std()
        nb_nirs_mean, nb_nirs_std = nb_tr_nirs.mean(), nb_tr_nirs.std()
        nb_tr_eeg  = (nb_tr_eeg  - nb_eeg_mean)  / (nb_eeg_std  + 1e-8)
        nb_tr_nirs = (nb_tr_nirs - nb_nirs_mean) / (nb_nirs_std + 1e-8)
        nb_te_eeg  = (nb_te_eeg  - nb_eeg_mean)  / (nb_eeg_std  + 1e-8)
        nb_te_nirs = (nb_te_nirs - nb_nirs_mean) / (nb_nirs_std + 1e-8)

        nb_train_loader = DataLoader(
            EEGNIRSDataset(nb_tr_eeg, nb_tr_nirs, nb_tr_lbl),
            batch_size=batch_size, shuffle=False
        )
        nb_test_loader = DataLoader(
            EEGNIRSDataset(nb_te_eeg, nb_te_nirs, nb_te_lbl),
            batch_size=batch_size, shuffle=False
        )

        train_emb, train_lbl_np = extract_embeddings(model, nb_train_loader, device)
        test_emb,  test_lbl_np  = extract_embeddings(model, nb_test_loader,  device)

        # ── Step 3: Linear probe on n-back embeddings ──
        acc, f1 = linear_probe(train_emb, train_lbl_np, test_emb, test_lbl_np)

        results[test_subj] = {
            'acc': float(acc),
            'f1':  float(f1),
            'vf_pretrain_loss': float(final_loss)
        }
        log(f"  [{test_subj}] n-back linear probe: Acc={acc:.3f}  F1={f1:.3f}")

        # Label distribution for reference
        for c, name in [(0,'0-back'),(1,'2-back'),(2,'3-back')]:
            n = int((nb_te_lbl == c).sum())
            log(f"    {name}: {n} test windows")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])

    log(f"\n{'='*60}")
    log(f"Cross-Task Generalization: VF pretrain → n-back probe")
    log(f"  Mean Accuracy : {mean_acc:.3f}  (chance=0.333)")
    log(f"  Mean F1       : {mean_f1:.3f}")
    log(f"{'='*60}")
    log(f"Interpretation:")
    log(f"  > 0.333 = backbone captures task-agnostic brain states")
    log(f"  > VF-only F1 = confirms cross-task transfer (not task memorization)")
    log(f"{'='*60}")

    return results, mean_acc, mean_f1


if __name__ == '__main__':
    # Preprocess VF if not already done
    vf_path = 'data/processed/dataset_vf_windows.pkl'
    if not os.path.exists(vf_path):
        log("VF windows not found — running preprocessing...")
        import subprocess, sys
        subprocess.run([sys.executable, 'preprocess_vf.py'], check=True)

    # Load VF windows (pretrain source)
    log("Loading VF windows (pretraining task)...")
    with open(vf_path, 'rb') as f:
        vf_data = pickle.load(f)
    log(f"  VF subjects: {sorted(vf_data.keys())}")

    # Preprocess n-back if not already done
    nback_path = 'data/processed/dataset_nback_windows.pkl'
    if not os.path.exists(nback_path):
        log("n-back windows not found — running preprocessing...")
        import subprocess, sys
        subprocess.run([sys.executable, 'preprocess_nback.py'], check=True)

    log("Loading n-back windows (evaluation task)...")
    with open(nback_path, 'rb') as f:
        nback_data = pickle.load(f)
    log(f"  n-back subjects: {sorted(nback_data.keys())}")

    results, mean_acc, mean_f1 = run_crosstask_loso(
        vf_data, nback_data,
        n_pretrain_epochs=50,
        batch_size=64,
        lr=3e-4,
        temperature=0.07
    )

    output = {
        'experiment': 'Cross-task generalization: VF pretrain → n-back linear probe',
        'pretrain_task': 'VF (binary: VF vs baseline)',
        'eval_task': 'n-back (3-class: 0-back vs 2-back vs 3-back)',
        'n_subjects': len(results),
        'evaluation': 'LOSO',
        'n_pretrain_epochs': 50,
        'temperature': 0.07,
        'embed_dim': 128,
        'chance_level': 0.333,
        'mean_acc': round(mean_acc, 4),
        'mean_f1':  round(mean_f1,  4),
        'per_subject': {
            s: {
                'acc': round(v['acc'], 4),
                'f1':  round(v['f1'],  4),
                'vf_pretrain_loss': round(v['vf_pretrain_loss'], 5)
            }
            for s, v in results.items()
        }
    }

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/crosstask_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    log("Saved → data/processed/crosstask_results.json")
