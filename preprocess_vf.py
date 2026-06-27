"""
Preprocess Shin 2017 Dataset C (verbal fluency task) for Foundation Brain evaluation.
Produces: data/processed/dataset_vf_windows.pkl

Run from project root:
    python preprocess_vf.py
"""

import pickle
import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt
import os
from pathlib import Path

DATA_ROOT = Path('data/raw')
OUT_PATH  = Path('data/processed/dataset_vf_windows.pkl')
SUBJECTS  = [f'VP{str(i).zfill(3)}' for i in range(1, 26) if i not in (12, 13)]


def load_subject(subj, task='vf', data_root=DATA_ROOT):
    """
    Load EEG, NIRS, and markers for one subject and task.

    Returns dict with:
      'eeg': np.ndarray (T_eeg, 30) -- EEG in uV at 200 Hz
      'nirs_hbo': np.ndarray (T_nirs, 36) -- HbO at 10 Hz
      'nirs_hbr': np.ndarray (T_nirs, 36) -- HbR at 10 Hz
      'eeg_fs': float -- EEG sampling rate (should be 200)
      'nirs_fs': float -- NIRS sampling rate (should be 10)
      'eeg_mrk_times_ms': np.ndarray -- trial onset times in ms (EEG clock)
      'nirs_mrk_times_ms': np.ndarray -- trial onset times in ms (NIRS clock)
      'labels': np.ndarray of int -- 1=task (VF), 0=baseline (BL)
      'class_names': list of str
    """
    subj_dir = Path(data_root) / subj

    var = f'cnt_{task}'
    mrk_var = f'mrk_{task}'

    # Load EEG
    eeg_path = subj_dir / 'EEG' / f'cnt_{task}.mat'
    eeg_mat = scipy.io.loadmat(str(eeg_path), squeeze_me=True, struct_as_record=False)
    cnt_eeg = eeg_mat[var]
    eeg_data = cnt_eeg.x.astype(np.float32)  # (T, C)
    eeg_fs = float(cnt_eeg.fs)

    # Load EEG markers
    eeg_mrk_path = subj_dir / 'EEG' / f'mrk_{task}.mat'
    eeg_mrk_mat = scipy.io.loadmat(str(eeg_mrk_path), squeeze_me=True, struct_as_record=False)
    mrk_eeg = eeg_mrk_mat[mrk_var]
    eeg_mrk_times = np.atleast_1d(mrk_eeg.time).astype(float)

    # Load NIRS
    nirs_path = subj_dir / 'NIRS' / f'cnt_{task}.mat'
    nirs_mat = scipy.io.loadmat(str(nirs_path), squeeze_me=True, struct_as_record=False)
    cnt_nirs = nirs_mat[var]
    nirs_hbo = cnt_nirs.oxy.x.astype(np.float32)    # (T, 36)
    nirs_hbr = cnt_nirs.deoxy.x.astype(np.float32)  # (T, 36)
    nirs_fs = float(cnt_nirs.oxy.fs)

    # Load NIRS markers
    nirs_mrk_path = subj_dir / 'NIRS' / f'mrk_{task}.mat'
    nirs_mrk_mat = scipy.io.loadmat(str(nirs_mrk_path), squeeze_me=True, struct_as_record=False)
    mrk_nirs = nirs_mrk_mat[mrk_var]
    nirs_mrk_times = np.atleast_1d(mrk_nirs.time).astype(float)

    # Labels: mrk.y is 2D one-hot (2, n_trials), argmax gives 0=BL, 1=VF
    y = mrk_eeg.y
    if y.ndim == 1:
        labels = np.atleast_1d(y).astype(int)
    else:
        labels = np.argmax(y, axis=0).astype(int)

    # class names
    class_names = list(mrk_eeg.className)

    return {
        'eeg': eeg_data,
        'nirs_hbo': nirs_hbo,
        'nirs_hbr': nirs_hbr,
        'eeg_fs': eeg_fs,
        'nirs_fs': nirs_fs,
        'eeg_mrk_times_ms': eeg_mrk_times,
        'nirs_mrk_times_ms': nirs_mrk_times,
        'labels': labels,
        'class_names': class_names,
    }


def bandpass_eeg(eeg, fs, low=1.0, high=40.0, order=6):
    """Apply zero-phase bandpass filter to EEG. Input: (T, C), Output: (T, C)"""
    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    try:
        return sosfiltfilt(sos, eeg, axis=0)
    except Exception as e:
        print(f'  Warning: bandpass_eeg failed ({e}), applying per-channel')
        out = np.zeros_like(eeg)
        for c in range(eeg.shape[1]):
            try:
                out[:, c] = sosfiltfilt(sos, eeg[:, c])
            except Exception as e2:
                print(f'    Channel {c} skipped: {e2}')
                out[:, c] = eeg[:, c]
        return out


def lowpass_nirs(nirs, fs, cutoff=0.2, order=6):
    """Apply zero-phase low-pass filter to NIRS. Input: (T, C), Output: (T, C)"""
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    try:
        return sosfiltfilt(sos, nirs, axis=0)
    except Exception as e:
        print(f'  Warning: lowpass_nirs failed ({e}), applying per-channel')
        out = np.zeros_like(nirs)
        for c in range(nirs.shape[1]):
            try:
                out[:, c] = sosfiltfilt(sos, nirs[:, c])
            except Exception as e2:
                print(f'    Channel {c} skipped: {e2}')
                out[:, c] = nirs[:, c]
        return out


def extract_windows(data, task='vf', window_sec=10, step_sec=1):
    """
    Extract sliding windows from each trial.

    For VF task: task duration = 10s, rest = 13-15s
    Positive pairs: (eeg_window, nirs_window) from same trial, same time offset

    EEG window shape:  (window_sec * eeg_fs, 30)  = (2000, 30)
    NIRS window shape: (window_sec * nirs_fs, 72)  = (100, 72)  [HbO + HbR concatenated]

    Returns:
      eeg_windows:  np.ndarray (N, 30, 2000)  -- transposed for model input (C, T)
      nirs_windows: np.ndarray (N, 72, 100)   -- transposed for model input (C, T)
      labels:       np.ndarray (N,)
    """
    eeg_fs = data['eeg_fs']
    nirs_fs = data['nirs_fs']
    eeg_win = int(window_sec * eeg_fs)    # 2000
    nirs_win = int(window_sec * nirs_fs)  # 100
    eeg_step = int(step_sec * eeg_fs)     # 200
    nirs_step = int(step_sec * nirs_fs)   # 10

    eeg_data = data['eeg']         # (T_eeg, 30)
    nirs_hbo = data['nirs_hbo']   # (T_nirs, 36)
    nirs_hbr = data['nirs_hbr']   # (T_nirs, 36)
    nirs_data = np.concatenate([nirs_hbo, nirs_hbr], axis=1)  # (T_nirs, 72)

    eeg_mrk_ms = data['eeg_mrk_times_ms']
    nirs_mrk_ms = data['nirs_mrk_times_ms']
    labels_raw = data['labels']  # 1=VF, 0=BL

    # trial duration in samples
    task_dur_eeg = int(10 * eeg_fs)    # 10s task period
    task_dur_nirs = int(10 * nirs_fs)

    eeg_wins, nirs_wins, lbls = [], [], []

    for i, (eeg_t_ms, nirs_t_ms, lbl) in enumerate(zip(eeg_mrk_ms, nirs_mrk_ms, labels_raw)):
        eeg_onset = int(eeg_t_ms * eeg_fs / 1000)
        nirs_onset = int(nirs_t_ms * nirs_fs / 1000)

        # slide window across trial
        t_eeg = eeg_onset
        t_nirs = nirs_onset
        while (t_eeg + eeg_win <= eeg_onset + task_dur_eeg and
               t_eeg + eeg_win <= eeg_data.shape[0] and
               t_nirs + nirs_win <= nirs_onset + task_dur_nirs and
               t_nirs + nirs_win <= nirs_data.shape[0]):
            eeg_wins.append(eeg_data[t_eeg:t_eeg + eeg_win, :].T)     # (30, 2000)
            nirs_wins.append(nirs_data[t_nirs:t_nirs + nirs_win, :].T) # (72, 100)
            lbls.append(lbl)
            t_eeg += eeg_step
            t_nirs += nirs_step

    return (np.array(eeg_wins, dtype=np.float32),
            np.array(nirs_wins, dtype=np.float32),
            np.array(lbls, dtype=np.int64))


if __name__ == '__main__':
    # Find which subjects are actually available under DATA_ROOT
    available_subjects = []
    for subj in SUBJECTS:
        if (DATA_ROOT / subj / 'EEG' / 'cnt_vf.mat').exists() and (DATA_ROOT / subj / 'NIRS' / 'cnt_vf.mat').exists():
            available_subjects.append(subj)

    if not available_subjects:
        print(f"Error: No subject data found in {DATA_ROOT}. Please run 'python data/download_dataset.py' first.")
        exit(1)

    print(f"Found available subjects for VF preprocessing: {available_subjects}")

    all_data = {}
    for subj in available_subjects:
        print(f"\n--- Preprocessing {subj} ---")
        try:
            data = load_subject(subj, 'vf', DATA_ROOT)
            print(f"  EEG shape: {data['eeg'].shape}, fs={data['eeg_fs']} Hz")
            print(f"  NIRS HbO shape: {data['nirs_hbo'].shape}, fs={data['nirs_fs']} Hz")
            print(f"  Label dist: VF={np.sum(data['labels']==1)}, BL={np.sum(data['labels']==0)}")

            data['eeg'] = bandpass_eeg(data['eeg'], data['eeg_fs'])
            data['nirs_hbo'] = lowpass_nirs(data['nirs_hbo'], data['nirs_fs'])
            data['nirs_hbr'] = lowpass_nirs(data['nirs_hbr'], data['nirs_fs'])

            eeg_wins, nirs_wins, labels = extract_windows(data, task='vf')
            all_data[subj] = {'eeg': eeg_wins, 'nirs': nirs_wins, 'labels': labels}
            print(f'  EEG windows shape: {eeg_wins.shape}')
            print(f'  NIRS windows shape: {nirs_wins.shape}')
            print(f"  Labels dist: VF={np.sum(labels==1)}, BL={np.sum(labels==0)}")
        except Exception as e:
            print(f'  ERROR preprocessing {subj}: {e}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(all_data, f)
    print(f'\nPreprocessing complete. Saved to {OUT_PATH}')
