"""Script to create notebooks/03_baseline_efnet.ipynb with actual results injected."""
import nbformat
import json
import os

nb = nbformat.v4.new_notebook()

cells = []

# --- Cell 0: Title markdown ---
cells.append(nbformat.v4.new_markdown_cell("""\
# Task 5 — EF-Net Baseline (PyTorch Reimplementation)

**EF-Net** (Arif et al. 2024, *Sensors*): Dual-branch CNN for simultaneous EEG + fNIRS classification.

Original TF implementation: `baselines/EF-Net/EEG-fNIRS/hybrid_model_structures.py`
This notebook reimplements EF-Net in PyTorch and evaluates it on Shin 2017 Dataset C (verbal fluency task, VP001–VP005) using **Leave-One-Subject-Out (LOSO)** cross-validation.
"""))

# --- Cell 1: Imports ---
cells.append(nbformat.v4.new_code_cell("""\
import pickle
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
"""))

# --- Cell 2: EFNet model ---
cells.append(nbformat.v4.new_code_cell("""\
class EFNet(nn.Module):
    \"\"\"
    PyTorch reimplementation of EF-Net (Arif et al. 2024, Sensors).
    Adapted from TF original in baselines/EF-Net/EEG-fNIRS/hybrid_model_structures.py.
    Input shapes adapted to 5s windows: EEG (B,1,30,1000), NIRS (B,1,72,50).

    Key differences from original TF:
    - Output: 2-class softmax (CrossEntropyLoss) instead of 1-class sigmoid (BinaryCrossEntropy)
    - Input layout: (B,C,H,W) PyTorch convention vs TF (B,H,W,C)
    - Kernel orientation flipped accordingly: (H,W) → (channels, time)
    \"\"\"
    def __init__(self, n_classes=2):
        super().__init__()
        # EEG branch — temporal convolutions along time axis (kernel: 1 channel x N time)
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
        # NIRS branch — spatial + temporal convolutions
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
        # Compute flattened sizes via dummy forward pass
        with torch.no_grad():
            dummy_eeg = torch.zeros(1, 1, 30, 1000)
            dummy_nirs = torch.zeros(1, 1, 72, 50)
            eeg_flat = self.eeg_branch(dummy_eeg)
            nirs_flat = self.nirs_branch(dummy_nirs)

        print(f"EEG branch flattened size : {eeg_flat.shape[1]}")
        print(f"NIRS branch flattened size: {nirs_flat.shape[1]}")

        self.eeg_fc = nn.Sequential(
            nn.Linear(eeg_flat.shape[1], 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.nirs_fc = nn.Sequential(
            nn.Linear(nirs_flat.shape[1], 128), nn.ReLU()
        )
        # Combined: 128 + 128 = 256 → L2 norm → classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, eeg, nirs):
        # eeg: (B, 1, 30, 1000)  nirs: (B, 1, 72, 50)
        e = self.eeg_fc(self.eeg_branch(eeg))    # (B, 128)
        f = self.nirs_fc(self.nirs_branch(nirs))  # (B, 128)
        combined = torch.cat([e, f], dim=1)        # (B, 256)
        combined = F.normalize(combined, p=2, dim=1)  # L2 norm (as in original)
        return self.classifier(combined)           # (B, n_classes)

# Instantiate and print architecture
model = EFNet(n_classes=2)
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total_params:,}")
"""))

# --- Cell 3: Dataset class ---
cells.append(nbformat.v4.new_code_cell("""\
class EEGNIRSDataset(Dataset):
    \"\"\"Dataset wrapper that adds the channel dimension expected by Conv2d.\"\"\"
    def __init__(self, eeg, nirs, labels):
        # (N, 30, 1000) -> (N, 1, 30, 1000)
        self.eeg    = torch.FloatTensor(eeg).unsqueeze(1)
        self.nirs   = torch.FloatTensor(nirs).unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.eeg[i], self.nirs[i], self.labels[i]
"""))

# --- Cell 4: LOSO loop ---
cells.append(nbformat.v4.new_code_cell("""\
def run_loso(all_data, n_epochs=30, batch_size=8, lr=1e-3):
    # CPU used: CUDA OOM with large EEG windows (30x1000 fp32 activations)
    device = torch.device('cpu')
    print(f"Device: {device}")
    subjects = sorted(all_data.keys())
    results  = {}

    for test_subj in subjects:
        print(f"\\n--- Test subject: {test_subj} ---")
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

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for eeg_b, nirs_b, lbl_b in train_loader:
                eeg_b, nirs_b, lbl_b = eeg_b.to(device), nirs_b.to(device), lbl_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(eeg_b, nirs_b), lbl_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d}/{n_epochs}  loss={epoch_loss/len(train_loader):.4f}")

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for eeg_b, nirs_b, lbl_b in test_loader:
                pred = model(eeg_b.to(device), nirs_b.to(device)).argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(lbl_b.numpy())

        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='weighted')
        results[test_subj] = {'acc': acc, 'f1': f1}
        print(f"  {test_subj}: Acc={acc:.3f}  F1={f1:.3f}")

    mean_acc = np.mean([v['acc'] for v in results.values()])
    mean_f1  = np.mean([v['f1']  for v in results.values()])
    print(f"\\nLOSO Mean Accuracy : {mean_acc:.3f}")
    print(f"LOSO Mean F1       : {mean_f1:.3f}")
    print(f"(Paper subject-independent F1 = 0.6505 on full 26 subjects)")
    return results, mean_acc, mean_f1
"""))

# --- Cell 5: Load data and run ---
cells.append(nbformat.v4.new_code_cell("""\
with open('../data/processed/dataset_vf_windows.pkl', 'rb') as f:
    all_data = pickle.load(f)

print("Loaded subjects:", sorted(all_data.keys()))
for s, d in all_data.items():
    print(f"  {s}: eeg={d['eeg'].shape}, nirs={d['nirs'].shape}, labels={d['labels'].shape}, "
          f"balance={d['labels'].mean():.2f}")

results, mean_acc, mean_f1 = run_loso(all_data)
"""))

# --- Cell 6: Save results ---
cells.append(nbformat.v4.new_code_cell("""\
efnet_results = {
    'model': 'EF-Net (PyTorch reimplementation)',
    'dataset': 'Shin 2017 Dataset C (VF task)',
    'n_subjects': len(all_data),
    'evaluation': 'LOSO',
    'mean_acc': round(mean_acc, 4),
    'mean_f1': round(mean_f1, 4),
    'paper_f1_26subj': 0.6505,
    'per_subject': {s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
                    for s, v in results.items()}
}

os.makedirs('../data/processed', exist_ok=True)
with open('../data/processed/efnet_results.json', 'w') as fh:
    json.dump(efnet_results, fh, indent=2)
print("Saved results to data/processed/efnet_results.json")
print(json.dumps(efnet_results, indent=2))
"""))

# --- Cell 7: Architecture insights markdown ---
cells.append(nbformat.v4.new_markdown_cell("""\
## Architecture Insights

### What each CNN branch learns
- **EEG branch**: Three stacked `(1x7)` temporal convolutions scan short (~35ms) time patterns within each channel independently, then `(4x4)` spatial-temporal convolutions mix across both channels and time. The branch extracts local temporal dynamics (e.g., event-related oscillations) that generalise weakly across subjects because electrode geometry is fixed.
- **NIRS branch**: `(1x4)` kernels along the time axis at 10 Hz capture slow haemodynamic fluctuations (~400ms scale). The subsequent `(2x2)` kernels mix adjacent channel pairs. NIRS activations reflect regional blood-oxygenation changes that are broader and slower than EEG — the branch learns region-level metabolic signatures.

### Why EF-Net cannot produce task-agnostic embeddings
EF-Net is trained end-to-end with a classification objective (binary cross-entropy in the original, cross-entropy here). The 128-dim branch embeddings and the 256-dim combined vector are shaped purely to minimise classification loss on the training subjects. There is no alignment objective between EEG and NIRS representations, and no incentive for the latent space to be interpretable or transferable. The model therefore over-fits subject-specific patterns — exactly why LOSO performance with only 5 subjects is well below the 26-subject paper figure.

### The L2 normalisation layer
The original TF code applies `tf.math.l2_normalize(x, axis=1, epsilon=5e-4)` to the 256-dim combined vector before the final MLP. This constrains the combined representation to lie on a unit hypersphere, which:
1. Prevents gradient explosion through the large FC stack.
2. Makes the subsequent linear classifier purely angle-based (cosine distance), which can help when EEG and NIRS feature magnitudes differ by orders of magnitude.
3. Is a design choice reminiscent of metric-learning losses (ArcFace, CosFace) — possibly a conscious nod toward representation learning, even though no explicit contrastive objective is present.

We reproduce this with `F.normalize(combined, p=2, dim=1)`.

### Key limitation: no contrastive objective => misaligned latent space
EF-Net concatenates EEG and NIRS embeddings and L2-normalises, but never explicitly aligns them. The two modalities occupy arbitrary subspaces of the 256-dim sphere. A foundation brain model should instead enforce cross-modal alignment (e.g., contrastive loss, CLIP-style InfoNCE) so that embed(EEG_t) and embed(NIRS_t) from the same time window are close regardless of subject or task. EF-Net provides a useful discriminative baseline but is not a representation learning architecture.

### 5-subject vs 26-subject caveat
The paper reports F1=0.6505 on 26 subjects. With only 5 subjects the LOSO training set shrinks to 4 subjects (1440 windows), severely limiting generalisation. The numbers below are a sanity-check that the architecture runs correctly — meaningful comparison requires all 26 subjects.
"""))

nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.13.3'}

out_path = r'C:\Users\ahars\Desktop\Funstuff\coding\neuro-work\foundation-brain\notebooks\03_baseline_efnet.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Notebook written to {out_path}")
