"""
Foundation Brain — Phase 2
CLIP-style contrastive pretraining on EEG+fNIRS pairs (NT-Xent / symmetric InfoNCE),
followed by a linear probe evaluated via LOSO.

Run from project root:
    python run_foundation_loso.py

Outputs:
    data/processed/foundation_results.json
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


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────

class EEGEncoder(nn.Module):
    """
    ShallowConvNet-style EEG encoder.
    Input: (B, 1, 30, 1000)  — 1 channel, 30 electrodes, 1000 timepoints (5s @ 200Hz)
    Output: (B, 128)          — embedding e_eeg
    """
    def __init__(self, embed_dim=128, input_shape=(30, 1000)):
        super().__init__()
        n_channels = input_shape[0]
        # Temporal convolution — learns frequency filters
        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12))
        # Spatial convolution — learns electrode weighting (depthwise over channels)
        self.spatial  = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=1)
        self.bn        = nn.BatchNorm2d(40)
        self.pool      = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop      = nn.Dropout(0.5)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self._forward_conv(dummy)
            flat = out.shape[1]

        self.proj = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
        )

    def _forward_conv(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2                     # square activation (ShallowConvNet)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))   # log activation
        x = self.drop(x)
        return x.flatten(1)

    def forward(self, x):
        return self.proj(self._forward_conv(x))


class NIRSEncoder(nn.Module):
    """
    Small CNN fNIRS encoder.
    Input: (B, 1, 72, 50)  — 1 channel, 72 optodes (HbO+HbR), 50 timepoints (5s @ 10Hz)
    Output: (B, 128)        — embedding e_nirs
    """
    def __init__(self, embed_dim=128, input_shape=(72, 50)):
        super().__init__()
        n_channels = input_shape[0]
        # Temporal convolution across time
        self.temporal = nn.Conv2d(1,  32, kernel_size=(1, 5), padding=(0, 2))
        # Spatial convolution across optodes
        self.spatial  = nn.Conv2d(32, 64, kernel_size=(n_channels, 1))
        self.bn        = nn.BatchNorm2d(64)
        self.pool      = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2))
        self.drop      = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self._forward_conv(dummy)
            flat = out.shape[1]

        self.proj = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
        )

    def _forward_conv(self, x):
        x = self.temporal(x)
        x = F.elu(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x.flatten(1)

    def forward(self, x):
        return self.proj(self._forward_conv(x))


# ─────────────────────────────────────────────
# Projection Head + Foundation Model
# ─────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP projection head. Discarded after pretraining."""
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)   # L2-normalize onto unit hypersphere


class FoundationBrain(nn.Module):
    def __init__(self, embed_dim=128, eeg_shape=(30, 1000), nirs_shape=(72, 50)):
        super().__init__()
        self.eeg_encoder  = EEGEncoder(embed_dim, input_shape=eeg_shape)
        self.nirs_encoder = NIRSEncoder(embed_dim, input_shape=nirs_shape)
        self.eeg_head     = ProjectionHead(embed_dim, 256, 64)
        self.nirs_head    = ProjectionHead(embed_dim, 256, 64)

    def encode(self, eeg, nirs):
        """Returns raw embeddings (used for linear probe, no projection head)."""
        return self.eeg_encoder(eeg), self.nirs_encoder(nirs)

    def forward(self, eeg, nirs):
        """Returns L2-normalized projected embeddings (used during pretraining)."""
        e_eeg  = self.eeg_encoder(eeg)
        e_nirs = self.nirs_encoder(nirs)
        z_eeg  = self.eeg_head(e_eeg)
        z_nirs = self.nirs_head(e_nirs)
        return z_eeg, z_nirs


# ─────────────────────────────────────────────
# NT-Xent Loss (symmetric InfoNCE)
# ─────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Symmetric InfoNCE loss (NT-Xent).
    Given N paired (eeg, nirs) embeddings, both already L2-normalized:
      - Positive pair i: (z_eeg_i, z_nirs_i)
      - Negatives for i: all z_nirs_j where j != i

    L = 0.5 * L(eeg→nirs) + 0.5 * L(nirs→eeg)

    Temperature τ controls sharpness. Lower τ = harder negatives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z_eeg, z_nirs):
        N = z_eeg.shape[0]

        # Similarity matrix: S[i,j] = cosine_sim(z_eeg_i, z_nirs_j) / τ
        # Both are already L2-normalized so dot product = cosine similarity
        S = torch.mm(z_eeg, z_nirs.T) / self.tau   # (N, N)

        # Labels: diagonal is the positive pair
        labels = torch.arange(N, device=z_eeg.device)

        # Cross-entropy in both directions
        loss_eeg2nirs = F.cross_entropy(S,   labels)
        loss_nirs2eeg = F.cross_entropy(S.T, labels)

        return 0.5 * (loss_eeg2nirs + loss_nirs2eeg)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class EEGNIRSDataset(Dataset):
    def __init__(self, eeg, nirs, labels=None):
        self.eeg    = torch.FloatTensor(eeg).unsqueeze(1)    # (N,1,30,1000)
        self.nirs   = torch.FloatTensor(nirs).unsqueeze(1)   # (N,1,72,50)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, i):
        if self.labels is not None:
            return self.eeg[i], self.nirs[i], self.labels[i]
        return self.eeg[i], self.nirs[i]


# ─────────────────────────────────────────────
# LOSO: Pretrain + Linear Probe
# ─────────────────────────────────────────────

def pretrain_one_fold(model, train_loader, device,
                      n_epochs=50, lr=3e-4, temperature=0.07):
    """Contrastive pretraining on all training subjects for one LOSO fold."""
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    epoch_bar = tqdm(range(n_epochs), desc='    pretraining', unit='ep',
                     leave=False, file=sys.stdout)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        for eeg_b, nirs_b, _ in train_loader:
            eeg_b, nirs_b = eeg_b.to(device), nirs_b.to(device)
            optimizer.zero_grad()
            z_eeg, z_nirs = model(eeg_b, nirs_b)
            loss = criterion(z_eeg, z_nirs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        epoch_bar.set_postfix(loss=f'{avg:.4f}')
        scheduler.step()

    return avg


def extract_embeddings(model, loader, device):
    """Extract e_eeg and e_nirs embeddings (before projection head)."""
    model.eval()
    eegs, nirss, lbls = [], [], []
    with torch.no_grad():
        for eeg_b, nirs_b, lbl_b in loader:
            eeg_b, nirs_b = eeg_b.to(device), nirs_b.to(device)
            e_eeg, e_nirs = model.encode(eeg_b, nirs_b)
            # Concatenate both modality embeddings → full brain state vector
            combined = torch.cat([e_eeg, e_nirs], dim=1).cpu().numpy()
            eegs.append(combined)
            lbls.append(lbl_b.numpy())
    return np.vstack(eegs), np.concatenate(lbls)


def linear_probe(train_emb, train_lbl, test_emb, test_lbl):
    """
    Freeze backbone, fit a linear classifier on embeddings.
    Uses sklearn LogisticRegression — no gradient, no overfit risk.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_test  = scaler.transform(test_emb)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                              random_state=42)
    clf.fit(X_train, train_lbl)
    preds = clf.predict(X_test)
    acc = accuracy_score(test_lbl, preds)
    f1  = f1_score(test_lbl, preds, average='weighted')
    return acc, f1


def run_loso(all_data, n_pretrain_epochs=50, batch_size=64,
             lr=3e-4, temperature=0.07):
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

    fold_bar = tqdm(subjects, desc='LOSO folds', unit='fold', file=sys.stdout)

    for test_subj in fold_bar:
        fold_bar.set_postfix(test=test_subj)

        # ── Build train / test splits ──
        tr_eeg  = np.concatenate([all_data[s]['eeg']    for s in subjects if s != test_subj])
        tr_nirs = np.concatenate([all_data[s]['nirs']   for s in subjects if s != test_subj])
        tr_lbl  = np.concatenate([all_data[s]['labels'] for s in subjects if s != test_subj])

        te_eeg  = all_data[test_subj]['eeg']
        te_nirs = all_data[test_subj]['nirs']
        te_lbl  = all_data[test_subj]['labels']

        train_loader = DataLoader(
            EEGNIRSDataset(tr_eeg, tr_nirs, tr_lbl),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            EEGNIRSDataset(te_eeg, te_nirs, te_lbl),
            batch_size=batch_size, shuffle=False
        )

        # ── Pretrain backbone with NT-Xent ──
        model = FoundationBrain(embed_dim=128, eeg_shape=tr_eeg.shape[1:],
                                nirs_shape=tr_nirs.shape[1:]).to(device)
        final_loss = pretrain_one_fold(
            model, train_loader, device,
            n_epochs=n_pretrain_epochs, lr=lr, temperature=temperature
        )

        # ── Extract embeddings ──
        train_emb, train_lbl_np = extract_embeddings(model, train_loader, device)

        # Rebuild test loader without drop_last
        test_emb, test_lbl_np = extract_embeddings(model, test_loader, device)

        # ── Linear probe ──
        acc, f1 = linear_probe(train_emb, train_lbl_np, test_emb, test_lbl_np)

        results[test_subj] = {
            'acc': float(acc),
            'f1':  float(f1),
            'final_pretrain_loss': float(final_loss)
        }
        log(f"\n  {test_subj}: Acc={acc:.3f}  F1={f1:.3f}  (pretrain loss={final_loss:.4f})")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])

    log(f"\n{'='*60}")
    log(f"Foundation Brain LOSO Results")
    log(f"  Mean Accuracy : {mean_acc:.3f}")
    log(f"  Mean F1       : {mean_f1:.3f}")
    log(f"{'='*60}")
    log(f"Comparison:")
    log(f"  EF-Net (supervised, 5 subj) : Acc=0.592  F1=0.572")
    log(f"  Foundation Brain (linear probe, 5 subj): Acc={mean_acc:.3f}  F1={mean_f1:.3f}")
    log(f"{'='*60}")

    return results, mean_acc, mean_f1


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Preprocess VF if not already done
    vf_path = 'data/processed/dataset_vf_windows.pkl'
    if not os.path.exists(vf_path):
        log("VF windows not found — running preprocessing...")
        import subprocess, sys
        subprocess.run([sys.executable, 'preprocess_vf.py'], check=True)

    log("Loading preprocessed windows...")
    with open(vf_path, 'rb') as f:
        all_data = pickle.load(f)

    log(f"Subjects: {sorted(all_data.keys())}")
    for s, d in all_data.items():
        log(f"  {s}: eeg={d['eeg'].shape}  nirs={d['nirs'].shape}  labels={d['labels'].shape}")

    results, mean_acc, mean_f1 = run_loso(
        all_data,
        n_pretrain_epochs=50,
        batch_size=64,
        lr=3e-4,
        temperature=0.07
    )

    output = {
        'model': 'Foundation Brain (EEGEncoder + NIRSEncoder, NT-Xent + linear probe)',
        'dataset': 'Shin 2017 Dataset C (VF task)',
        'n_subjects': len(all_data),
        'evaluation': 'LOSO — contrastive pretrain → linear probe',
        'n_pretrain_epochs': 50,
        'temperature': 0.07,
        'embed_dim': 128,
        'mean_acc': round(mean_acc, 4),
        'mean_f1':  round(mean_f1,  4),
        'efnet_baseline_f1': 0.572,
        'per_subject': {
            s: {
                'acc': round(v['acc'], 4),
                'f1':  round(v['f1'],  4),
                'final_pretrain_loss': round(v['final_pretrain_loss'], 5)
            }
            for s, v in results.items()
        }
    }

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/foundation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    log("Saved → data/processed/foundation_results.json")
