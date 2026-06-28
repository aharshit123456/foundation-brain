"""
Standalone script to run EF-Net LOSO evaluation on VP001-VP005.
Run from project root: python run_efnet_loso.py
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
from tqdm import tqdm


def log(msg):
    print(msg, flush=True)


class EFNet(nn.Module):
    """
    PyTorch reimplementation of EF-Net (Arif et al. 2024, Sensors).
    Adapted from TF original in baselines/EF-Net/EEG-fNIRS/hybrid_model_structures.py.
    eeg_shape/nirs_shape should match the actual window length used by the dataset.
    """
    def __init__(self, n_classes=2, eeg_shape=(30, 1000), nirs_shape=(72, 50), modality='both'):
        super().__init__()
        assert modality in ('both', 'eeg', 'nirs')
        self.modality = modality
        self.eeg_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 7)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 7)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 7)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(4, 4)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(4, 4)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        self.nirs_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 4)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 4)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        with torch.no_grad():
            eeg_out  = self.eeg_branch(torch.zeros(1, 1, *eeg_shape))
            nirs_out = self.nirs_branch(torch.zeros(1, 1, *nirs_shape))

        self.eeg_fc = nn.Sequential(
            nn.Linear(eeg_out.shape[1], 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.nirs_fc = nn.Sequential(
            nn.Linear(nirs_out.shape[1], 128), nn.ReLU()
        )
        classifier_in = {'both': 256, 'eeg': 128, 'nirs': 128}[modality]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, eeg, nirs):
        if self.modality == 'eeg':
            combined = F.normalize(self.eeg_fc(self.eeg_branch(eeg)), p=2, dim=1)
        elif self.modality == 'nirs':
            combined = F.normalize(self.nirs_fc(self.nirs_branch(nirs)), p=2, dim=1)
        else:
            e = self.eeg_fc(self.eeg_branch(eeg))
            f = self.nirs_fc(self.nirs_branch(nirs))
            combined = F.normalize(torch.cat([e, f], dim=1), p=2, dim=1)
        return self.classifier(combined)


class EEGNIRSDataset(Dataset):
    def __init__(self, eeg, nirs, labels):
        self.eeg    = torch.FloatTensor(eeg).unsqueeze(1)
        self.nirs   = torch.FloatTensor(nirs).unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.eeg[i], self.nirs[i], self.labels[i]


def run_loso(all_data, n_epochs=30, batch_size=32, lr=1e-3, modality='both'):
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
    results  = {}

    # Outer progress bar — one tick per LOSO fold
    fold_bar = tqdm(subjects, desc='LOSO folds', unit='fold', file=sys.stdout)

    for test_subj in fold_bar:
        fold_bar.set_postfix(test=test_subj)

        tr_eeg  = np.concatenate([all_data[s]['eeg']    for s in subjects if s != test_subj])
        tr_nirs = np.concatenate([all_data[s]['nirs']   for s in subjects if s != test_subj])
        tr_lbl  = np.concatenate([all_data[s]['labels'] for s in subjects if s != test_subj])

        te_eeg  = all_data[test_subj]['eeg']
        te_nirs = all_data[test_subj]['nirs']
        te_lbl  = all_data[test_subj]['labels']

        # Z-score normalize each modality using train-set statistics only (no test leakage).
        # NIRS values are ~1e-3 scale vs EEG's ~1e1 scale -- without this, the NIRS branch's
        # conv weights (initialized for ~unit-variance input) see near-zero gradients and
        # never leave random init, collapsing to a single-class prediction regardless of
        # epochs trained. See run_efnet_loso.py git history / project notes for the
        # diagnosis: loss frozen at ln(2)=0.693 and chance-level accuracy on every fold.
        eeg_mean, eeg_std = tr_eeg.mean(), tr_eeg.std()
        nirs_mean, nirs_std = tr_nirs.mean(), tr_nirs.std()
        tr_eeg  = (tr_eeg  - eeg_mean)  / (eeg_std  + 1e-8)
        tr_nirs = (tr_nirs - nirs_mean) / (nirs_std + 1e-8)
        te_eeg  = (te_eeg  - eeg_mean)  / (eeg_std  + 1e-8)
        te_nirs = (te_nirs - nirs_mean) / (nirs_std + 1e-8)

class DeviceResidentBatcher:
    """
    Replaces DataLoader for small, fully-in-memory datasets.
    """
    def __init__(self, eeg, nirs, labels=None, batch_size=64, shuffle=True,
                drop_last=False, device='cpu'):
        self.eeg    = torch.FloatTensor(eeg).unsqueeze(1).to(device)
        self.nirs   = torch.FloatTensor(nirs).unsqueeze(1).to(device)
        self.labels = torch.LongTensor(labels).to(device) if labels is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = self.eeg.shape[0]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.eeg.device) if self.shuffle \
            else torch.arange(self.n, device=self.eeg.device)
        n_batches = len(self)
        for b in range(n_batches):
            sl = idx[b * self.batch_size : (b + 1) * self.batch_size]
            if self.labels is not None:
                yield self.eeg[sl], self.nirs[sl], self.labels[sl]
            else:
                yield self.eeg[sl], self.nirs[sl]


def run_loso(all_data, n_epochs=30, batch_size=32, lr=1e-3, modality='both'):
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
    results  = {}

    # Outer progress bar — one tick per LOSO fold
    fold_bar = tqdm(subjects, desc='LOSO folds', unit='fold', file=sys.stdout)

    for test_subj in fold_bar:
        fold_bar.set_postfix(test=test_subj)

        tr_eeg  = np.concatenate([all_data[s]['eeg']    for s in subjects if s != test_subj])
        tr_nirs = np.concatenate([all_data[s]['nirs']   for s in subjects if s != test_subj])
        tr_lbl  = np.concatenate([all_data[s]['labels'] for s in subjects if s != test_subj])

        te_eeg  = all_data[test_subj]['eeg']
        te_nirs = all_data[test_subj]['nirs']
        te_lbl  = all_data[test_subj]['labels']

        # Z-score normalize each modality using train-set statistics only (no test leakage).
        # NIRS values are ~1e-3 scale vs EEG's ~1e1 scale -- without this, the NIRS branch's
        # conv weights (initialized for ~unit-variance input) see near-zero gradients and
        # never leave random init, collapsing to a single-class prediction regardless of
        # epochs trained. See run_efnet_loso.py git history / project notes for the
        # diagnosis: loss frozen at ln(2)=0.693 and chance-level accuracy on every fold.
        eeg_mean, eeg_std = tr_eeg.mean(), tr_eeg.std()
        nirs_mean, nirs_std = tr_nirs.mean(), tr_nirs.std()
        tr_eeg  = (tr_eeg  - eeg_mean)  / (eeg_std  + 1e-8)
        tr_nirs = (tr_nirs - nirs_mean) / (nirs_std + 1e-8)
        te_eeg  = (te_eeg  - eeg_mean)  / (eeg_std  + 1e-8)
        te_nirs = (te_nirs - nirs_mean) / (nirs_std + 1e-8)

        train_loader = DeviceResidentBatcher(tr_eeg, tr_nirs, tr_lbl,
                                             batch_size=batch_size, shuffle=True, device=device)
        test_loader  = DeviceResidentBatcher(te_eeg, te_nirs, te_lbl,
                                             batch_size=batch_size, shuffle=False, device=device)

        eeg_shape  = tr_eeg.shape[1:]
        nirs_shape = tr_nirs.shape[1:]
        model     = EFNet(n_classes=2, eeg_shape=eeg_shape, nirs_shape=nirs_shape,
                          modality=modality).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Inner progress bar — one tick per epoch
        epoch_bar = tqdm(range(n_epochs), desc=f'  {test_subj} training',
                         unit='epoch', leave=False, file=sys.stdout)
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            for eeg_b, nirs_b, lbl_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(eeg_b, nirs_b), lbl_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            epoch_bar.set_postfix(loss=f'{avg_loss:.4f}')

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for eeg_b, nirs_b, lbl_b in test_loader:
                preds.extend(model(eeg_b, nirs_b).argmax(dim=1).cpu().numpy())
                trues.extend(lbl_b.cpu().numpy())

        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='weighted')
        results[test_subj] = {'acc': acc, 'f1': f1}
        log(f"\n  {test_subj}: Acc={acc:.3f}  F1={f1:.3f}")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])
    log(f"\n{'='*50}")
    log(f"LOSO Mean Accuracy : {mean_acc:.3f}")
    log(f"LOSO Mean F1       : {mean_f1:.3f}")
    log(f"(Paper subject-independent F1 = 0.6505 on full 26 subjects, fNIRS+EEG)")
    return results, mean_acc, mean_f1


# Paper's Table 4 (subject-independent, 20 train / 6 test subjects) reference numbers
PAPER_F1_BY_MODALITY = {'both': 0.6505, 'nirs': 0.6380, 'eeg': 0.5666}
OUT_FILE_BY_MODALITY  = {
    'both': 'efnet_results.json',
    'nirs': 'efnet_results_fnirs.json',
    'eeg':  'efnet_results_eeg.json',
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

    log(f"Subjects: {sorted(all_data.keys())}")
    for s, d in all_data.items():
        log(f"  {s}: eeg={d['eeg'].shape}, nirs={d['nirs'].shape}, labels={d['labels'].shape}")
    log(f"Modality: {modality}")

    results, mean_acc, mean_f1 = run_loso(all_data, n_epochs=30, batch_size=32, lr=1e-3,
                                          modality=modality)

    efnet_results = {
        'model': f'EF-Net (PyTorch reimplementation, modality={modality})',
        'dataset': 'Shin 2017 Dataset C (VF task)',
        'modality': modality,
        'n_subjects': len(all_data),
        'evaluation': 'LOSO',
        'mean_acc': round(mean_acc, 4),
        'mean_f1':  round(mean_f1, 4),
        'paper_f1_26subj': PAPER_F1_BY_MODALITY[modality],
        'per_subject': {s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
                        for s, v in results.items()}
    }
    os.makedirs('data/processed', exist_ok=True)
    out_path = f'data/processed/{OUT_FILE_BY_MODALITY[modality]}'
    with open(out_path, 'w') as f:
        json.dump(efnet_results, f, indent=2)
    log(f"Saved results to {out_path}")
