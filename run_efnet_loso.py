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
    Input shapes adapted to 5s windows: EEG (B,1,30,1000), NIRS (B,1,72,50).
    """
    def __init__(self, n_classes=2):
        super().__init__()
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
            eeg_out  = self.eeg_branch(torch.zeros(1, 1, 30, 1000))
            nirs_out = self.nirs_branch(torch.zeros(1, 1, 72, 50))

        self.eeg_fc = nn.Sequential(
            nn.Linear(eeg_out.shape[1], 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.nirs_fc = nn.Sequential(
            nn.Linear(nirs_out.shape[1], 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, eeg, nirs):
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


def run_loso(all_data, n_epochs=30, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        train_loader = DataLoader(EEGNIRSDataset(tr_eeg, tr_nirs, tr_lbl),
                                  batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(EEGNIRSDataset(all_data[test_subj]['eeg'],
                                                  all_data[test_subj]['nirs'],
                                                  all_data[test_subj]['labels']),
                                  batch_size=batch_size, shuffle=False)

        model     = EFNet(n_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Inner progress bar — one tick per epoch
        epoch_bar = tqdm(range(n_epochs), desc=f'  {test_subj} training',
                         unit='epoch', leave=False, file=sys.stdout)
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            for eeg_b, nirs_b, lbl_b in train_loader:
                eeg_b, nirs_b, lbl_b = eeg_b.to(device), nirs_b.to(device), lbl_b.to(device)
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
                eeg_b, nirs_b = eeg_b.to(device), nirs_b.to(device)
                preds.extend(model(eeg_b, nirs_b).argmax(dim=1).cpu().numpy())
                trues.extend(lbl_b.numpy())

        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='weighted')
        results[test_subj] = {'acc': acc, 'f1': f1}
        log(f"\n  {test_subj}: Acc={acc:.3f}  F1={f1:.3f}")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])
    log(f"\n{'='*50}")
    log(f"LOSO Mean Accuracy : {mean_acc:.3f}")
    log(f"LOSO Mean F1       : {mean_f1:.3f}")
    log(f"(Paper subject-independent F1 = 0.6505 on full 26 subjects)")
    return results, mean_acc, mean_f1


if __name__ == '__main__':
    log("Loading data...")
    with open('data/processed/dataset_vf_windows.pkl', 'rb') as f:
        all_data = pickle.load(f)

    log(f"Subjects: {sorted(all_data.keys())}")
    for s, d in all_data.items():
        log(f"  {s}: eeg={d['eeg'].shape}, nirs={d['nirs'].shape}, labels={d['labels'].shape}")

    results, mean_acc, mean_f1 = run_loso(all_data, n_epochs=30, batch_size=32, lr=1e-3)

    efnet_results = {
        'model': 'EF-Net (PyTorch reimplementation)',
        'dataset': 'Shin 2017 Dataset C (VF task)',
        'n_subjects': len(all_data),
        'evaluation': 'LOSO',
        'mean_acc': round(mean_acc, 4),
        'mean_f1':  round(mean_f1, 4),
        'paper_f1_26subj': 0.6505,
        'per_subject': {s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
                        for s, v in results.items()}
    }
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/efnet_results.json', 'w') as f:
        json.dump(efnet_results, f, indent=2)
    log("Saved results to data/processed/efnet_results.json")
