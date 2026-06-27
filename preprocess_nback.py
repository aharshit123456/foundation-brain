"""
Preprocess Shin 2017 Dataset A (n-back task) for Foundation Brain cross-task evaluation.

Produces: data/processed/dataset_nback_windows.pkl
  Dict keyed by subject, each entry:
    eeg    (N, 30, 1000)  — 5s windows @ 200Hz
    nirs   (N, 72, 50)    — 5s windows @ 10Hz, HbO+HbR concatenated
    labels (N,)           — 0=0-back, 1=2-back, 2=3-back

Run from project root:
    python preprocess_nback.py
"""

import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

DATA_ROOT = Path('data/raw')
OUT_PATH  = Path('data/processed/dataset_nback_windows.pkl')

# n-back session marker rows in mrk.y
SESSION_ROWS   = {0: 5, 1: 6, 2: 7}   # label → row index in y matrix
CLASS_NAMES    = {0: '0-back', 1: '2-back', 2: '3-back'}
TASK_DUR_S     = 40     # seconds of task per trial (paper protocol)
WIN_S, STEP_S  = 10, 1  # window and step in seconds — matches VF window length for cross-task transfer


def bandpass_eeg(eeg, fs, low=1.0, high=40.0, order=6):
    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, eeg, axis=0)


def lowpass_nirs(nirs, fs, cutoff=0.2, order=6):
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, nirs, axis=0)


def load_subject_nback(subj, data_root=DATA_ROOT):
    eeg_dir  = data_root / subj / 'EEG'
    nirs_dir = data_root / subj / 'NIRS'

    # EEG
    eeg_mat = sio.loadmat(str(eeg_dir / 'cnt_nback.mat'),
                          squeeze_me=True, struct_as_record=False)
    mrk_mat = sio.loadmat(str(eeg_dir / 'mrk_nback.mat'),
                          squeeze_me=True, struct_as_record=False)
    cnt = eeg_mat['cnt_nback']
    mrk = mrk_mat['mrk_nback']
    eeg_data = cnt.x.astype(np.float32)    # (T, 30)
    fs_eeg   = float(cnt.fs)               # 200 Hz

    # NIRS
    nirs_mat = sio.loadmat(str(nirs_dir / 'cnt_nback.mat'),
                           squeeze_me=True, struct_as_record=False)
    cnt_n      = nirs_mat['cnt_nback']
    nirs_hbo   = cnt_n.oxy.x.astype(np.float32)    # (T, 36)
    nirs_hbr   = cnt_n.deoxy.x.astype(np.float32)  # (T, 36)
    fs_nirs    = float(cnt_n.oxy.fs)                # ~10 Hz

    # Extract session marker times (ms → sample indices)
    # mrk.y: (8, n_events), mrk.time: (n_events,) in ms
    session_times = {}
    for label, row in SESSION_ROWS.items():
        times_ms = mrk.time[mrk.y[row] == 1]
        session_times[label] = times_ms  # in milliseconds

    return eeg_data, nirs_hbo, nirs_hbr, fs_eeg, fs_nirs, session_times


def extract_windows_nback(eeg, nirs_hbo, nirs_hbr, fs_eeg, fs_nirs, session_times):
    """
    For each trial (session marker), extract sliding windows over the 40s task period.
    Returns eeg_wins (N,30,1000), nirs_wins (N,72,50), labels (N,)
    """
    win_eeg  = int(WIN_S  * fs_eeg)    # 1000
    step_eeg = int(STEP_S * fs_eeg)    # 200
    win_nirs  = int(WIN_S  * fs_nirs)  # 50
    step_nirs = int(STEP_S * fs_nirs)  # 10
    dur_eeg   = int(TASK_DUR_S * fs_eeg)
    dur_nirs  = int(TASK_DUR_S * fs_nirs)

    all_eeg, all_nirs, all_labels = [], [], []

    for label, times_ms in session_times.items():
        for t_ms in times_ms:
            # Convert ms to sample index
            t_eeg  = int(t_ms / 1000.0 * fs_eeg)
            t_nirs = int(t_ms / 1000.0 * fs_nirs)

            # Bounds check — skip if trial extends beyond signal
            if t_eeg + dur_eeg > eeg.shape[0]:
                continue
            if t_nirs + dur_nirs > nirs_hbo.shape[0]:
                continue

            eeg_trial  = eeg[t_eeg  : t_eeg  + dur_eeg]      # (8000, 30)
            hbo_trial  = nirs_hbo[t_nirs : t_nirs + dur_nirs] # (400, 36)
            hbr_trial  = nirs_hbr[t_nirs : t_nirs + dur_nirs] # (400, 36)
            nirs_trial = np.concatenate([hbo_trial, hbr_trial], axis=1)  # (400, 72)

            # Sliding windows
            t = 0
            while t + win_eeg <= eeg_trial.shape[0]:
                t_n = int(t / step_eeg) * step_nirs

                if t_n + win_nirs > nirs_trial.shape[0]:
                    break

                eeg_win  = eeg_trial[t : t + win_eeg].T        # (30, 1000)
                nirs_win = nirs_trial[t_n : t_n + win_nirs].T  # (72, 50)

                all_eeg.append(eeg_win)
                all_nirs.append(nirs_win)
                all_labels.append(label)
                t += step_eeg

    return (np.stack(all_eeg).astype(np.float32),
            np.stack(all_nirs).astype(np.float32),
            np.array(all_labels, dtype=np.int64))


def preprocess_subject(subj):
    print(f'  {subj}...', end=' ', flush=True)
    eeg, hbo, hbr, fs_eeg, fs_nirs, session_times = load_subject_nback(subj)

    # Filter
    eeg = bandpass_eeg(eeg, fs_eeg)
    hbo = lowpass_nirs(hbo, fs_nirs)
    hbr = lowpass_nirs(hbr, fs_nirs)

    eeg_wins, nirs_wins, labels = extract_windows_nback(
        eeg, hbo, hbr, fs_eeg, fs_nirs, session_times
    )

    counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(3)}
    print(f'eeg={eeg_wins.shape}  nirs={nirs_wins.shape}  labels={counts}')
    return {'eeg': eeg_wins, 'nirs': nirs_wins, 'labels': labels}


if __name__ == '__main__':
    # Find subjects that have nback data
    subjects = sorted([
        p.name for p in DATA_ROOT.iterdir()
        if (p / 'EEG' / 'cnt_nback.mat').exists()
        and (p / 'NIRS' / 'cnt_nback.mat').exists()
    ])
    print(f'Found {len(subjects)} subjects with n-back data: {subjects}')

    dataset = {}
    for subj in subjects:
        try:
            dataset[subj] = preprocess_subject(subj)
        except Exception as e:
            print(f'  FAILED: {e}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'\nSaved {len(dataset)} subjects to {OUT_PATH}')
