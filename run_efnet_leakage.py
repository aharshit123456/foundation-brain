import pickle
import numpy as np
import json
import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# We will import helper classes from existing scripts
sys.path.insert(0, '.')
from preprocess_vf import load_subject, bandpass_eeg, lowpass_nirs
from run_efnet_loso import EFNet, DeviceResidentBatcher

SUBJECTS = ['VP001', 'VP002', 'VP003']
SEEDS = [38, 43, 45]
DATA_ROOT = 'data/raw'

def extract_overlapping_windows(data, window_sec=3, step_sec=0.2):
    """
    Extract heavily overlapping 3-second windows to simulate shuffle leakage.
    For each of the 60 trials (10s duration), we extract sliding windows of 3s with 0.2s step.
    This yields (10-3)/0.2 + 1 = 36 windows per trial.
    """
    eeg_fs = data['eeg_fs']
    nirs_fs = data['nirs_fs']
    eeg_win = int(window_sec * eeg_fs)    # 600
    nirs_win = int(window_sec * nirs_fs)  # 30
    eeg_step = int(step_sec * eeg_fs)     # 40
    nirs_step = int(step_sec * nirs_fs)   # 2
    
    eeg_data = data['eeg']
    nirs_data = np.concatenate([data['nirs_hbo'], data['nirs_hbr']], axis=1)
    
    eeg_mrk_ms = data['eeg_mrk_times_ms']
    nirs_mrk_ms = data['nirs_mrk_times_ms']
    labels_raw = data['labels']
    
    task_dur_eeg = int(10 * eeg_fs)
    task_dur_nirs = int(10 * nirs_fs)
    
    eeg_wins, nirs_wins, lbls = [], [], []
    for eeg_t_ms, nirs_t_ms, lbl in zip(eeg_mrk_ms, nirs_mrk_ms, labels_raw):
        eeg_onset = int(eeg_t_ms * eeg_fs / 1000)
        nirs_onset = int(nirs_t_ms * nirs_fs / 1000)
        
        t_eeg = eeg_onset
        t_nirs = nirs_onset
        while (t_eeg + eeg_win <= eeg_onset + task_dur_eeg and
               t_eeg + eeg_win <= eeg_data.shape[0] and
               t_nirs + nirs_win <= nirs_onset + task_dur_nirs and
               t_nirs + nirs_win <= nirs_data.shape[0]):
            eeg_wins.append(eeg_data[t_eeg:t_eeg + eeg_win, :].T)
            nirs_wins.append(nirs_data[t_nirs:t_nirs + nirs_win, :].T)
            lbls.append(lbl)
            t_eeg += eeg_step
            t_nirs += nirs_step
            
    return np.array(eeg_wins, dtype=np.float32), np.array(nirs_wins, dtype=np.float32), np.array(lbls, dtype=np.int64)

def run_leaked_experiment():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess raw data for VP001-VP003
    all_leaked_data = {}
    for subj in SUBJECTS:
        print(f"Preprocessing {subj} with overlapping windows...")
        data = load_subject(subj, 'vf', DATA_ROOT)
        data['eeg'] = bandpass_eeg(data['eeg'], data['eeg_fs'])
        data['nirs_hbo'] = lowpass_nirs(data['nirs_hbo'], data['nirs_fs'])
        data['nirs_hbr'] = lowpass_nirs(data['nirs_hbr'], data['nirs_fs'])
        
        eeg_wins, nirs_wins, labels = extract_overlapping_windows(data)
        all_leaked_data[subj] = {'eeg': eeg_wins, 'nirs': nirs_wins, 'labels': labels}
        print(f"  Extracted {len(labels)} overlapping windows. EEG shape: {eeg_wins.shape}, NIRS shape: {nirs_wins.shape}")

    results = {}
    modalities = ['both', 'nirs', 'eeg']
    
    for modality in modalities:
        results[modality] = {}
        print(f"\n--- Running Modality: {modality} ---")
        for subj in SUBJECTS:
            subj_data = all_leaked_data[subj]
            eeg, nirs, labels = subj_data['eeg'], subj_data['nirs'], subj_data['labels']
            
            seed_accs, seed_f1s = [], []
            for seed in SEEDS:
                # Shuffle-then-split (leaks overlap temporal correlation between train and test splits)
                idx = np.arange(len(labels))
                train_idx, test_idx = train_test_split(
                    idx, test_size=0.2, random_state=seed, stratify=labels
                )
                
                # Z-score normalize
                eeg_mean, eeg_std = eeg[train_idx].mean(), eeg[train_idx].std()
                nirs_mean, nirs_std = nirs[train_idx].mean(), nirs[train_idx].std()
                eeg_norm = (eeg - eeg_mean) / (eeg_std + 1e-8)
                nirs_norm = (nirs - nirs_mean) / (nirs_std + 1e-8)
                
                train_loader = DeviceResidentBatcher(
                    eeg_norm[train_idx], nirs_norm[train_idx], labels[train_idx],
                    batch_size=64, shuffle=True, device=device
                )
                test_loader = DeviceResidentBatcher(
                    eeg_norm[test_idx], nirs_norm[test_idx], labels[test_idx],
                    batch_size=64, shuffle=False, device=device
                )
                
                model = EFNet(n_classes=2, eeg_shape=eeg.shape[1:], nirs_shape=nirs.shape[1:],
                              modality=modality).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                
                # Train for 15 epochs (usually converges to near-zero loss extremely fast under leakage)
                for epoch in range(15):
                    model.train()
                    for eeg_b, nirs_b, lbl_b in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(model(eeg_b, nirs_b), lbl_b)
                        loss.backward()
                        optimizer.step()
                        
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for eeg_b, nirs_b, lbl_b in test_loader:
                        preds.extend(model(eeg_b, nirs_b).argmax(dim=1).cpu().numpy())
                        trues.extend(lbl_b.cpu().numpy())
                        
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds)
                seed_accs.append(acc)
                seed_f1s.append(f1)
                
            mean_acc = np.mean(seed_accs)
            mean_f1 = np.mean(seed_f1s)
            results[modality][subj] = {'acc': mean_acc, 'f1': mean_f1}
            print(f"  {subj} ({modality}): Acc={mean_acc:.4f}  F1={mean_f1:.4f}")
            
    # Save the results
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/efnet_leaked_subjectdep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved leaked results to data/processed/efnet_leaked_subjectdep.json")

if __name__ == '__main__':
    run_leaked_experiment()
