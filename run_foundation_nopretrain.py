"""
Foundation Brain -- No-Pretraining Control

The critical control for the paper's central claim: if a randomly-initialized
(never contrastively pretrained) backbone scores near chance via the linear
probe, that demonstrates the NT-Xent pretraining is actually contributing
useful structure -- not that any random projection plus a logistic regression
probe would do just as well.

Protocol: identical to run_foundation_loso.py's LOSO loop, except the
pretrain_one_fold() call is skipped entirely. The backbone is instantiated
fresh per fold (random init) and immediately frozen for embedding extraction.

Run from project root:
    python run_foundation_nopretrain.py

Outputs:
    data/processed/foundation_nopretrain_results.json
"""
import pickle
import numpy as np
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from run_foundation_loso import (
    FoundationBrain, EEGNIRSDataset, extract_embeddings, linear_probe, log
)


def run_nopretrain_loso(all_data, batch_size=64):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log(f"Using device: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    subjects = sorted(all_data.keys())
    results = {}

    fold_bar = tqdm(subjects, desc='LOSO folds (no pretrain)', unit='fold', file=sys.stdout)

    for test_subj in fold_bar:
        fold_bar.set_postfix(test=test_subj)

        tr_eeg  = np.concatenate([all_data[s]['eeg']    for s in subjects if s != test_subj])
        tr_nirs = np.concatenate([all_data[s]['nirs']   for s in subjects if s != test_subj])
        tr_lbl  = np.concatenate([all_data[s]['labels'] for s in subjects if s != test_subj])

        te_eeg  = all_data[test_subj]['eeg']
        te_nirs = all_data[test_subj]['nirs']
        te_lbl  = all_data[test_subj]['labels']

        train_loader = DataLoader(
            EEGNIRSDataset(tr_eeg, tr_nirs, tr_lbl),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            EEGNIRSDataset(te_eeg, te_nirs, te_lbl),
            batch_size=batch_size, shuffle=False
        )

        # Random-init backbone -- no NT-Xent pretraining step at all.
        model = FoundationBrain(embed_dim=128, eeg_shape=tr_eeg.shape[1:],
                                nirs_shape=tr_nirs.shape[1:]).to(device)
        model.eval()

        train_emb, train_lbl_np = extract_embeddings(model, train_loader, device)
        test_emb, test_lbl_np = extract_embeddings(model, test_loader, device)

        acc, f1 = linear_probe(train_emb, train_lbl_np, test_emb, test_lbl_np)

        results[test_subj] = {'acc': float(acc), 'f1': float(f1)}
        log(f"\n  {test_subj}: Acc={acc:.3f}  F1={f1:.3f}")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])

    log(f"\n{'='*60}")
    log(f"No-Pretraining Control LOSO Results")
    log(f"  Mean Accuracy : {mean_acc:.3f}")
    log(f"  Mean F1       : {mean_f1:.3f}")
    log(f"{'='*60}")

    return results, mean_acc, mean_f1


if __name__ == '__main__':
    log("Loading preprocessed VF windows...")
    with open('data/processed/dataset_vf_windows.pkl', 'rb') as f:
        all_data = pickle.load(f)

    log(f"Subjects: {sorted(all_data.keys())}")

    results, mean_acc, mean_f1 = run_nopretrain_loso(all_data, batch_size=64)

    output = {
        'model': 'Foundation Brain (random-init backbone, NO pretraining) + linear probe',
        'dataset': 'Shin 2017 Dataset C (VF task)',
        'n_subjects': len(all_data),
        'evaluation': 'LOSO -- no pretraining (control)',
        'embed_dim': 128,
        'mean_acc': round(mean_acc, 4),
        'mean_f1': round(mean_f1, 4),
        'per_subject': {
            s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
            for s, v in results.items()
        }
    }

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/foundation_nopretrain_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    log("Saved -> data/processed/foundation_nopretrain_results.json")
