"""
EF-Net subject-dependent evaluation, matching Table 2 of Arif et al. 2024
(Sensors), where "both training and testing are performed using different
parts of samples from the same individual subject." The paper runs this on
subjects 1-3 with an 80/20 split of that subject's own samples (no other
subjects involved at all).

Our windowing yields only 60 samples/subject (vs. ~360 in the paper, which
used much shorter/overlapping windows), so results here are noisier and not
directly comparable in absolute terms -- this is a faithful reproduction of
their split *protocol*, not their exact sample counts.

Run from project root: python run_efnet_subjectdep.py
Outputs: data/processed/efnet_results_subjectdep.json
"""
import pickle
import numpy as np
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from run_efnet_loso import EFNet, EEGNIRSDataset, DeviceResidentBatcher, log

SUBJECTS = ['VP001', 'VP002', 'VP003']  # matches paper's Table 2
SUBJECTS = ['VP001', 'VP002', 'VP003']  # matches paper's Table 2
SEEDS = [38, 43, 45]  # matches paper's three random seeds


def run_subject_dependent(all_data, n_epochs=30, batch_size=32, lr=1e-3,
                          modality='both', test_size=0.2):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log(f"Using device: {device}")

    results = {}
    for subj in SUBJECTS:
        if subj not in all_data:
            log(f"  Skipping {subj}: not in dataset")
            continue

        eeg, nirs, labels = all_data[subj]['eeg'], all_data[subj]['nirs'], all_data[subj]['labels']
        seed_accs, seed_f1s = [], []

        for seed in SEEDS:
            idx = np.arange(len(labels))
            train_idx, test_idx = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=labels
            )

            # Z-score normalize using train-split statistics only -- see run_efnet_loso.py
            # for the diagnosis (NIRS is ~1e-3 scale vs EEG's ~1e1 scale; without this the
            # NIRS branch's conv weights never leave random init and collapse to chance).
            eeg_mean, eeg_std = eeg[train_idx].mean(), eeg[train_idx].std()
            nirs_mean, nirs_std = nirs[train_idx].mean(), nirs[train_idx].std()
            eeg_norm  = (eeg  - eeg_mean)  / (eeg_std  + 1e-8)
            nirs_norm = (nirs - nirs_mean) / (nirs_std + 1e-8)

            train_loader = DeviceResidentBatcher(
                eeg_norm[train_idx], nirs_norm[train_idx], labels[train_idx],
                batch_size=batch_size, shuffle=True, device=device
            )
            test_loader = DeviceResidentBatcher(
                eeg_norm[test_idx], nirs_norm[test_idx], labels[test_idx],
                batch_size=batch_size, shuffle=False, device=device
            )

            model = EFNet(n_classes=2, eeg_shape=eeg.shape[1:], nirs_shape=nirs.shape[1:],
                          modality=modality).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            epoch_bar = tqdm(range(n_epochs), desc=f'  {subj} seed={seed}',
                             unit='epoch', leave=False, file=sys.stdout)
            for _ in epoch_bar:
                model.train()
                epoch_loss = 0.0
                for eeg_b, nirs_b, lbl_b in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(eeg_b, nirs_b), lbl_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                epoch_bar.set_postfix(loss=f'{epoch_loss / len(train_loader):.4f}')

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for eeg_b, nirs_b, lbl_b in test_loader:
                    preds.extend(model(eeg_b, nirs_b).argmax(dim=1).cpu().numpy())
                    trues.extend(lbl_b.cpu().numpy())

            acc = accuracy_score(trues, preds)
            f1  = f1_score(trues, preds, average='weighted')
            seed_accs.append(acc)
            seed_f1s.append(f1)
            log(f"  {subj} seed={seed}: Acc={acc:.3f}  F1={f1:.3f}")

        results[subj] = {
            'acc_mean': float(np.mean(seed_accs)), 'acc_std': float(np.std(seed_accs)),
            'f1_mean':  float(np.mean(seed_f1s)),  'f1_std':  float(np.std(seed_f1s)),
        }

    mean_acc = np.mean([v['acc_mean'] for v in results.values()])
    mean_f1  = np.mean([v['f1_mean']  for v in results.values()])
    log(f"\n{'='*50}")
    log(f"Subject-Dependent Mean Accuracy : {mean_acc:.3f}")
    log(f"Subject-Dependent Mean F1       : {mean_f1:.3f}")
    log(f"(Paper Table 2, fNIRS+EEG, subjects 1-3: F1 = 0.9938 +/- 0.0071)")
    return results, mean_acc, mean_f1


# Paper's Table 2 reference numbers (subjects 1-3, fNIRS+EEG / fNIRS / EEG)
PAPER_F1_BY_MODALITY = {'both': 0.9938, 'nirs': 0.9969, 'eeg': 0.9645}
OUT_FILE_BY_MODALITY = {
    'both': 'efnet_results_subjectdep.json',
    'nirs': 'efnet_results_subjectdep_fnirs.json',
    'eeg':  'efnet_results_subjectdep_eeg.json',
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=['both', 'eeg', 'nirs'], default='both')
    args = parser.parse_args()
    modality = args.modality

    log("Loading data...")
    with open('data/processed/dataset_vf_windows.pkl', 'rb') as f:
        all_data = pickle.load(f)
    log(f"Subjects available: {sorted(all_data.keys())}")
    log(f"Running subject-dependent on: {SUBJECTS}, modality={modality}")

    results, mean_acc, mean_f1 = run_subject_dependent(all_data, modality=modality)

    out = {
        'model': f'EF-Net (PyTorch reimplementation, modality={modality})',
        'setting': 'subject-dependent',
        'dataset': 'Shin 2017 Dataset C (VF task)',
        'modality': modality,
        'subjects': SUBJECTS,
        'seeds': SEEDS,
        'mean_acc': round(mean_acc, 4),
        'mean_f1':  round(mean_f1, 4),
        'paper_f1_subj1to3': PAPER_F1_BY_MODALITY[modality],
        'per_subject': {s: {k: round(v, 4) for k, v in r.items()} for s, r in results.items()},
        'caveat': ('Our windowing yields ~60 samples/subject vs. the paper\'s ~360; '
                   'this reproduces their split protocol, not their sample density.'),
    }
    os.makedirs('data/processed', exist_ok=True)
    out_path = f'data/processed/{OUT_FILE_BY_MODALITY[modality]}'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved results to {out_path}")
