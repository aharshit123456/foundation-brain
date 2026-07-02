"""Script to create the 02_preprocessing.ipynb notebook."""
import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Title and imports
cells.append(nbf.v4.new_markdown_cell(
    "# 02 -- Preprocessing Pipeline: Shin 2017 EEG+NIRS Dataset\n\n"
    "Bandpass EEG (1-40 Hz), lowpass NIRS (0.2 Hz), extract sliding windows, save processed data."
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import scipy.io\n"
    "from scipy.signal import butter, sosfiltfilt\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import pickle\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "print('Imports OK')\n"
))

# Cell 2: load_subject function
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading"))

cells.append(nbf.v4.new_code_cell(
    "def load_subject(subj, task, data_root):\n"
    '    """\n'
    "    Load EEG, NIRS, and markers for one subject and task.\n"
    "\n"
    "    Returns dict with:\n"
    "      'eeg': np.ndarray (T_eeg, 30) -- EEG in uV at 200 Hz\n"
    "      'nirs_hbo': np.ndarray (T_nirs, 36) -- HbO at 10 Hz\n"
    "      'nirs_hbr': np.ndarray (T_nirs, 36) -- HbR at 10 Hz\n"
    "      'eeg_fs': float -- EEG sampling rate (should be 200)\n"
    "      'nirs_fs': float -- NIRS sampling rate (should be 10)\n"
    "      'eeg_mrk_times_ms': np.ndarray -- trial onset times in ms (EEG clock)\n"
    "      'nirs_mrk_times_ms': np.ndarray -- trial onset times in ms (NIRS clock)\n"
    "      'labels': np.ndarray of int -- 1=task (VF), 0=baseline (BL)\n"
    "      'class_names': list of str\n"
    '    """\n'
    "    data_root = Path(data_root)\n"
    "    subj_dir = data_root / subj\n"
    "\n"
    "    var = f'cnt_{task}'\n"
    "    mrk_var = f'mrk_{task}'\n"
    "\n"
    "    # Load EEG\n"
    "    eeg_path = subj_dir / 'EEG' / f'cnt_{task}.mat'\n"
    "    eeg_mat = scipy.io.loadmat(str(eeg_path), squeeze_me=True, struct_as_record=False)\n"
    "    cnt_eeg = eeg_mat[var]\n"
    "    eeg_data = cnt_eeg.x  # (T, C)\n"
    "    eeg_fs = float(cnt_eeg.fs)\n"
    "\n"
    "    # Load EEG markers\n"
    "    eeg_mrk_path = subj_dir / 'EEG' / f'mrk_{task}.mat'\n"
    "    eeg_mrk_mat = scipy.io.loadmat(str(eeg_mrk_path), squeeze_me=True, struct_as_record=False)\n"
    "    mrk_eeg = eeg_mrk_mat[mrk_var]\n"
    "    eeg_mrk_times = np.atleast_1d(mrk_eeg.time).astype(float)\n"
    "\n"
    "    # Load NIRS\n"
    "    nirs_path = subj_dir / 'NIRS' / f'cnt_{task}.mat'\n"
    "    nirs_mat = scipy.io.loadmat(str(nirs_path), squeeze_me=True, struct_as_record=False)\n"
    "    cnt_nirs = nirs_mat[var]\n"
    "    nirs_hbo = cnt_nirs.oxy.x    # (T, 36)\n"
    "    nirs_hbr = cnt_nirs.deoxy.x  # (T, 36)\n"
    "    nirs_fs = float(cnt_nirs.oxy.fs)\n"
    "\n"
    "    # Load NIRS markers\n"
    "    nirs_mrk_path = subj_dir / 'NIRS' / f'mrk_{task}.mat'\n"
    "    nirs_mrk_mat = scipy.io.loadmat(str(nirs_mrk_path), squeeze_me=True, struct_as_record=False)\n"
    "    mrk_nirs = nirs_mrk_mat[mrk_var]\n"
    "    nirs_mrk_times = np.atleast_1d(mrk_nirs.time).astype(float)\n"
    "\n"
    "    # Labels: mrk.y is 2D one-hot (2, n_trials), argmax gives 0=BL, 1=VF\n"
    "    y = mrk_eeg.y\n"
    "    if y.ndim == 1:\n"
    "        labels = np.atleast_1d(y).astype(int)\n"
    "    else:\n"
    "        labels = np.argmax(y, axis=0).astype(int)\n"
    "\n"
    "    # class names\n"
    "    class_names = list(mrk_eeg.className)\n"
    "\n"
    "    return {\n"
    "        'eeg': eeg_data,\n"
    "        'nirs_hbo': nirs_hbo,\n"
    "        'nirs_hbr': nirs_hbr,\n"
    "        'eeg_fs': eeg_fs,\n"
    "        'nirs_fs': nirs_fs,\n"
    "        'eeg_mrk_times_ms': eeg_mrk_times,\n"
    "        'nirs_mrk_times_ms': nirs_mrk_times,\n"
    "        'labels': labels,\n"
    "        'class_names': class_names,\n"
    "    }\n"
    "\n"
    "print('load_subject defined')\n"
))

# Cell 3: bandpass_eeg
cells.append(nbf.v4.new_markdown_cell("## 2. EEG Bandpass Filter (1-40 Hz)"))

cells.append(nbf.v4.new_code_cell(
    "def bandpass_eeg(eeg, fs, low=1.0, high=40.0, order=6):\n"
    '    """Apply zero-phase bandpass filter to EEG. Input: (T, C), Output: (T, C)"""\n'
    "    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')\n"
    "    try:\n"
    "        return sosfiltfilt(sos, eeg, axis=0)\n"
    "    except Exception as e:\n"
    "        print(f'  Warning: bandpass_eeg failed ({e}), applying per-channel')\n"
    "        out = np.zeros_like(eeg)\n"
    "        for c in range(eeg.shape[1]):\n"
    "            try:\n"
    "                out[:, c] = sosfiltfilt(sos, eeg[:, c])\n"
    "            except Exception as e2:\n"
    "                print(f'    Channel {c} skipped: {e2}')\n"
    "                out[:, c] = eeg[:, c]\n"
    "        return out\n"
    "\n"
    "print('bandpass_eeg defined')\n"
))

# Cell 4: lowpass_nirs
cells.append(nbf.v4.new_markdown_cell("## 3. NIRS Low-pass Filter (0.2 Hz)"))

cells.append(nbf.v4.new_code_cell(
    "def lowpass_nirs(nirs, fs, cutoff=0.2, order=6):\n"
    '    """Apply zero-phase low-pass filter to NIRS. Input: (T, C), Output: (T, C)"""\n'
    "    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')\n"
    "    try:\n"
    "        return sosfiltfilt(sos, nirs, axis=0)\n"
    "    except Exception as e:\n"
    "        print(f'  Warning: lowpass_nirs failed ({e}), applying per-channel')\n"
    "        out = np.zeros_like(nirs)\n"
    "        for c in range(nirs.shape[1]):\n"
    "            try:\n"
    "                out[:, c] = sosfiltfilt(sos, nirs[:, c])\n"
    "            except Exception as e2:\n"
    "                print(f'    Channel {c} skipped: {e2}')\n"
    "                out[:, c] = nirs[:, c]\n"
    "        return out\n"
    "\n"
    "print('lowpass_nirs defined')\n"
))

# Cell 5: extract_windows
cells.append(nbf.v4.new_markdown_cell("## 4. Sliding Window Extraction"))

cells.append(nbf.v4.new_code_cell(
    "def extract_windows(data, task='vf', window_sec=10, step_sec=1):\n"
    '    """\n'
    "    Extract sliding windows from each trial.\n"
    "\n"
    "    For VF task: task duration = 10s, rest = 13-15s\n"
    "    Positive pairs: (eeg_window, nirs_window) from same trial, same time offset\n"
    "\n"
    "    EEG window shape:  (window_sec * eeg_fs, 30)  = (2000, 30)\n"
    "    NIRS window shape: (window_sec * nirs_fs, 72)  = (100, 72)  [HbO + HbR concatenated]\n"
    "\n"
    "    Returns:\n"
    "      eeg_windows:  np.ndarray (N, 30, 2000)  -- transposed for model input (C, T)\n"
    "      nirs_windows: np.ndarray (N, 72, 100)   -- transposed for model input (C, T)\n"
    "      labels:       np.ndarray (N,)\n"
    '    """\n'
    "    eeg_fs = data['eeg_fs']\n"
    "    nirs_fs = data['nirs_fs']\n"
    "    eeg_win = int(window_sec * eeg_fs)    # 2000\n"
    "    nirs_win = int(window_sec * nirs_fs)  # 100\n"
    "    eeg_step = int(step_sec * eeg_fs)     # 200\n"
    "    nirs_step = int(step_sec * nirs_fs)   # 10\n"
    "\n"
    "    eeg_data = data['eeg']         # (T_eeg, 30)\n"
    "    nirs_hbo = data['nirs_hbo']   # (T_nirs, 36)\n"
    "    nirs_hbr = data['nirs_hbr']   # (T_nirs, 36)\n"
    "    nirs_data = np.concatenate([nirs_hbo, nirs_hbr], axis=1)  # (T_nirs, 72)\n"
    "\n"
    "    eeg_mrk_ms = data['eeg_mrk_times_ms']\n"
    "    nirs_mrk_ms = data['nirs_mrk_times_ms']\n"
    "    labels_raw = data['labels']  # 1=VF, 0=BL\n"
    "\n"
    "    # trial duration in samples\n"
    "    task_dur_eeg = int(10 * eeg_fs)    # 10s task period\n"
    "    task_dur_nirs = int(10 * nirs_fs)\n"
    "\n"
    "    eeg_wins, nirs_wins, lbls = [], [], []\n"
    "\n"
    "    for i, (eeg_t_ms, nirs_t_ms, lbl) in enumerate(zip(eeg_mrk_ms, nirs_mrk_ms, labels_raw)):\n"
    "        eeg_onset = int(eeg_t_ms * eeg_fs / 1000)\n"
    "        nirs_onset = int(nirs_t_ms * nirs_fs / 1000)\n"
    "\n"
    "        # slide window across trial\n"
    "        t_eeg = eeg_onset\n"
    "        t_nirs = nirs_onset\n"
    "        while (t_eeg + eeg_win <= eeg_onset + task_dur_eeg and\n"
    "               t_eeg + eeg_win <= eeg_data.shape[0] and\n"
    "               t_nirs + nirs_win <= nirs_onset + task_dur_nirs and\n"
    "               t_nirs + nirs_win <= nirs_data.shape[0]):\n"
    "            eeg_wins.append(eeg_data[t_eeg:t_eeg + eeg_win, :].T)     # (30, 2000)\n"
    "            nirs_wins.append(nirs_data[t_nirs:t_nirs + nirs_win, :].T) # (72, 100)\n"
    "            lbls.append(lbl)\n"
    "            t_eeg += eeg_step\n"
    "            t_nirs += nirs_step\n"
    "\n"
    "    return (np.array(eeg_wins, dtype=np.float32),\n"
    "            np.array(nirs_wins, dtype=np.float32),\n"
    "            np.array(lbls, dtype=np.int64))\n"
    "\n"
    "print('extract_windows defined')\n"
))

# Cell 6: Run pipeline
cells.append(nbf.v4.new_markdown_cell("## 5. Run Full Pipeline on VP001-VP005"))

cells.append(nbf.v4.new_code_cell(
    "os.makedirs('../data/processed', exist_ok=True)\n"
    "DATA_ROOT = Path('../data/raw')\n"
    "SUBJECTS = ['VP001', 'VP002', 'VP003', 'VP004', 'VP005']\n"
    "\n"
    "all_data = {}\n"
    "for subj in SUBJECTS:\n"
    "    print(f'\\n--- {subj} ---')\n"
    "    try:\n"
    "        data = load_subject(subj, 'vf', DATA_ROOT)\n"
    "        print(f\"  EEG shape: {data['eeg'].shape}, fs={data['eeg_fs']} Hz\")\n"
    "        print(f\"  NIRS HbO shape: {data['nirs_hbo'].shape}, fs={data['nirs_fs']} Hz\")\n"
    "        print(f\"  Trials: {len(data['labels'])}, class_names: {data['class_names']}\")\n"
    "        print(f\"  Label dist: VF={np.sum(data['labels']==1)}, BL={np.sum(data['labels']==0)}\")\n"
    "\n"
    "        data['eeg'] = bandpass_eeg(data['eeg'], data['eeg_fs'])\n"
    "        data['nirs_hbo'] = lowpass_nirs(data['nirs_hbo'], data['nirs_fs'])\n"
    "        data['nirs_hbr'] = lowpass_nirs(data['nirs_hbr'], data['nirs_fs'])\n"
    "\n"
    "        eeg_wins, nirs_wins, labels = extract_windows(data, task='vf')\n"
    "        all_data[subj] = {'eeg': eeg_wins, 'nirs': nirs_wins, 'labels': labels}\n"
    "        print(f'  EEG windows: {eeg_wins.shape}')\n"
    "        print(f'  NIRS windows: {nirs_wins.shape}')\n"
    "        print(f\"  Labels: VF={np.sum(labels==1)}, BL={np.sum(labels==0)}, total={len(labels)}\")\n"
    "    except Exception as e:\n"
    "        import traceback\n"
    "        print(f'  ERROR: {e}')\n"
    "        traceback.print_exc()\n"
    "\n"
    "print('\\nPipeline complete.')\n"
))

# Cell 7: Save processed data
cells.append(nbf.v4.new_markdown_cell("## 6. Save Processed Data"))

cells.append(nbf.v4.new_code_cell(
    "out_path = '../data/processed/dataset_vf_windows.pkl'\n"
    "with open(out_path, 'wb') as f:\n"
    "    pickle.dump(all_data, f)\n"
    "print(f'Saved to {out_path}')\n"
    "\n"
    "# Summary\n"
    "print('\\n=== SUMMARY ===')\n"
    "for subj, d in all_data.items():\n"
    "    print(f\"{subj}: EEG={d['eeg'].shape}, NIRS={d['nirs'].shape}, \"\n"
    "          f\"VF={np.sum(d['labels']==1)}, BL={np.sum(d['labels']==0)}, \"\n"
    "          f\"total={len(d['labels'])}\")\n"
))

# Cell 8: Verify shapes
cells.append(nbf.v4.new_markdown_cell("## 7. Verification -- Shapes and Label Balance"))

cells.append(nbf.v4.new_code_cell(
    "print('Shape verification:')\n"
    "for subj, d in all_data.items():\n"
    "    eeg_ok = d['eeg'].ndim == 3 and d['eeg'].shape[1] == 30 and d['eeg'].shape[2] == 2000\n"
    "    nirs_ok = d['nirs'].ndim == 3 and d['nirs'].shape[1] == 72 and d['nirs'].shape[2] == 100\n"
    "    status = 'OK' if (eeg_ok and nirs_ok) else 'FAIL'\n"
    "    vf_pct = 100 * np.sum(d['labels']==1) / len(d['labels'])\n"
    "    print(f\"  {subj} [{status}]: EEG={d['eeg'].shape}, NIRS={d['nirs'].shape}, \"\n"
    "          f\"VF%={vf_pct:.1f}%\")\n"
))

# Cell 9: Plot sample
cells.append(nbf.v4.new_markdown_cell("## 8. Sanity Check -- Plot Sample Windows from VP001"))

cells.append(nbf.v4.new_code_cell(
    "os.makedirs('../docs/plots', exist_ok=True)\n"
    "\n"
    "if 'VP001' in all_data:\n"
    "    d = all_data['VP001']\n"
    "\n"
    "    # Find first VF window\n"
    "    vf_idx = np.where(d['labels'] == 1)[0]\n"
    "    idx = vf_idx[0] if len(vf_idx) > 0 else 0\n"
    "\n"
    "    eeg_sample = d['eeg'][idx]    # (30, 2000)\n"
    "    nirs_sample = d['nirs'][idx]  # (72, 100)\n"
    "\n"
    "    fig, axes = plt.subplots(3, 1, figsize=(14, 12))\n"
    "    lbl_str = 'VF' if d['labels'][idx]==1 else 'BL'\n"
    "\n"
    "    # EEG: first 8 channels\n"
    "    ax = axes[0]\n"
    "    t_eeg = np.arange(eeg_sample.shape[1]) / 200.0\n"
    "    offset = np.arange(8) * 50  # uV offset per channel\n"
    "    for ch in range(8):\n"
    "        ax.plot(t_eeg, eeg_sample[ch] + offset[ch], linewidth=0.8)\n"
    "    ax.set_xlabel('Time (s)', fontsize=24)\n"
    "    ax.set_ylabel('Amplitude (uV + offset)', fontsize=24)\n"
    "    ax.tick_params(labelsize=21)\n"
    "    ax.set_xlim([0, 10])\n"
    "\n"
    "    # NIRS HbO: first 6 channels\n"
    "    ax = axes[1]\n"
    "    t_nirs = np.arange(nirs_sample.shape[1]) / 10.0\n"
    "    hbo_sample = nirs_sample[:36, :]  # first 36 = HbO\n"
    "    for ch in range(6):\n"
    "        ax.plot(t_nirs, hbo_sample[ch], linewidth=1.8, label=f'Ch{ch+1}')\n"
    "    ax.set_xlabel('Time (s)', fontsize=24)\n"
    "    ax.set_ylabel('HbO Concentration', fontsize=24)\n"
    "    ax.tick_params(labelsize=21)\n"
    "    ax.legend(loc='upper right', fontsize=17, ncol=3)\n"
    "    ax.set_xlim([0, 10])\n"
    "\n"
    "    # NIRS HbR: first 6 channels\n"
    "    ax = axes[2]\n"
    "    hbr_sample = nirs_sample[36:, :]  # last 36 = HbR\n"
    "    for ch in range(6):\n"
    "        ax.plot(t_nirs, hbr_sample[ch], linewidth=1.8, label=f'Ch{ch+1}')\n"
    "    ax.set_xlabel('Time (s)', fontsize=24)\n"
    "    ax.set_ylabel('HbR Concentration', fontsize=24)\n"
    "    ax.tick_params(labelsize=21)\n"
    "    ax.legend(loc='upper right', fontsize=17, ncol=3)\n"
    "    ax.set_xlim([0, 10])\n"
    "\n"
    "    plt.tight_layout()\n"
    "    out_fig = '../docs/plots/VP001_preprocessed_sample.png'\n"
    "    plt.savefig(out_fig, dpi=150, bbox_inches='tight')\n"
    "    plt.close()\n"
    "    print(f'Plot saved to {out_fig}')\n"
    "else:\n"
    "    print('VP001 not available in all_data')\n"
))

nb.cells = cells

# Write notebook with explicit UTF-8 encoding
out_path = r"C:\Users\ahars\Desktop\Funstuff\coding\neuro-work\foundation-brain\notebooks\02_preprocessing.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
