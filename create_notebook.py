import nbformat

cell1_code = r"""import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

DATA_ROOT = Path('../data/raw')
SUBJECT = 'VP001'
print('Imports OK')
print('DATA_ROOT:', DATA_ROOT.resolve())
"""

cell2_code = r"""# Cell 2 — Inspect EEG cnt_vf.mat structure
eeg_path = DATA_ROOT / SUBJECT / 'EEG' / 'cnt_vf.mat'
eeg_raw = scipy.io.loadmat(str(eeg_path), squeeze_me=True, struct_as_record=False)

print('Top-level keys:', [k for k in eeg_raw.keys() if not k.startswith('_')])
cnt_eeg = eeg_raw['cnt_vf']
print('\ncnt_vf fields:', cnt_eeg._fieldnames)
print('cnt_vf.x shape (samples x channels):', cnt_eeg.x.shape)
print('cnt_vf.fs (sampling rate):', cnt_eeg.fs)
print('cnt_vf.clab (first 10 channel labels):', list(cnt_eeg.clab[:10]))
print('cnt_vf.yUnit:', cnt_eeg.yUnit)
print('cnt_vf.title:', cnt_eeg.title)
"""

cell3_code = r"""# Cell 3 — Inspect NIRS cnt_vf.mat structure
nirs_path = DATA_ROOT / SUBJECT / 'NIRS' / 'cnt_vf.mat'
nirs_raw = scipy.io.loadmat(str(nirs_path), squeeze_me=True, struct_as_record=False)

print('Top-level keys:', [k for k in nirs_raw.keys() if not k.startswith('_')])
cnt_nirs = nirs_raw['cnt_vf']
print('\ncnt_vf fields:', cnt_nirs._fieldnames)

oxy = cnt_nirs.oxy
deoxy = cnt_nirs.deoxy
print('\noxy fields:', oxy._fieldnames)
print('oxy.x shape (samples x channels):', oxy.x.shape)
print('oxy.fs (sampling rate):', oxy.fs)
print('oxy.clab (first 5):', list(oxy.clab[:5]))
print('oxy.yUnit:', oxy.yUnit)
print('oxy.signal:', oxy.signal)
print('oxy.wavelengths:', oxy.wavelengths)

print('\ndeoxy.x shape:', deoxy.x.shape)
print('deoxy.fs:', deoxy.fs)
print('deoxy.clab (first 5):', list(deoxy.clab[:5]))
"""

cell4_code = r"""# Cell 4 — Inspect mrk_vf.mat (markers)
mrk_eeg_raw = scipy.io.loadmat(str(DATA_ROOT / SUBJECT / 'EEG' / 'mrk_vf.mat'), squeeze_me=True, struct_as_record=False)
mrk_eeg = mrk_eeg_raw['mrk_vf']
print('EEG mrk_vf fields:', mrk_eeg._fieldnames)
print('mrk.time (ms, first 10):', mrk_eeg.time[:10])
print('mrk.y shape:', mrk_eeg.y.shape)
print('mrk.className:', mrk_eeg.className)
print('mrk.event.desc (first 10):', mrk_eeg.event.desc[:10])

print()
mrk_nirs_raw = scipy.io.loadmat(str(DATA_ROOT / SUBJECT / 'NIRS' / 'mrk_vf.mat'), squeeze_me=True, struct_as_record=False)
mrk_nirs = mrk_nirs_raw['mrk_vf']
print('NIRS mrk_vf fields:', mrk_nirs._fieldnames)
print('mrk.time (ms, first 10):', mrk_nirs.time[:10])
print('mrk.y shape:', mrk_nirs.y.shape)
print('mrk.className:', mrk_nirs.className)
print('mrk.event.desc (first 10):', mrk_nirs.event.desc[:10])
"""

cell5_code = r"""# Cell 5 — Extract and print basic dataset stats
eeg_fs = cnt_eeg.fs
eeg_n_samples, eeg_n_channels = cnt_eeg.x.shape
eeg_duration_s = eeg_n_samples / eeg_fs

nirs_fs = oxy.fs
nirs_n_samples, nirs_n_oxy = oxy.x.shape
_, nirs_n_deoxy = deoxy.x.shape
nirs_total_channels = nirs_n_oxy + nirs_n_deoxy
nirs_duration_s = nirs_n_samples / nirs_fs

n_trials = mrk_eeg.time.shape[0]
trial_labels = mrk_eeg.event.desc

print(f'EEG Sampling Rate: {eeg_fs} Hz')
print(f'EEG Channels: {eeg_n_channels}')
print(f'EEG Duration: {eeg_duration_s:.1f} s ({eeg_duration_s/60:.1f} min)')
print()
print(f'NIRS Sampling Rate: {nirs_fs} Hz')
print(f'NIRS Channels: {nirs_n_oxy} HbO + {nirs_n_deoxy} HbR = {nirs_total_channels} total')
print(f'NIRS Duration: {nirs_duration_s:.1f} s ({nirs_duration_s/60:.1f} min)')
print()
print(f'Total trials in vf task: {n_trials}')
print(f'Trial class names: {list(mrk_eeg.className)}')
print(f'EEG event desc values (unique): {np.unique(trial_labels)}')
print(f'NIRS event desc values (unique): {np.unique(mrk_nirs.event.desc)}')
print(f'Label encoding — VF(word generation)=2, BL(baseline)=1  [NIRS mrk]')
print(f'Label encoding — VF(word generation)=32, BL(baseline)=16 [EEG mrk]')
"""

cell6_code = r"""# Cell 6 — Plot raw EEG signals (first 30 seconds, first 5 channels)
plot_dir = Path('../docs/plots')
plot_dir.mkdir(parents=True, exist_ok=True)

t_end = 30  # seconds
n_samples_plot = int(t_end * eeg_fs)
time_eeg = np.arange(n_samples_plot) / eeg_fs

fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for i in range(5):
    ch_label = cnt_eeg.clab[i]
    signal = cnt_eeg.x[:n_samples_plot, i]
    axes[i].plot(time_eeg, signal, lw=0.7, color='steelblue')
    axes[i].set_ylabel(f'{ch_label}\n({cnt_eeg.yUnit})', fontsize=8)
    axes[i].tick_params(labelsize=7)

axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'{SUBJECT} EEG — Raw Signal (first 30 s, first 5 channels)', fontsize=12)
plt.tight_layout()
out_path = plot_dir / 'VP001_eeg_raw.png'
plt.savefig(str(out_path), dpi=120)
plt.close()
print('Saved:', out_path.resolve())
"""

cell7_code = r"""# Cell 7 — Plot raw NIRS signals (first 60 seconds, first 2 channels)
t_end_nirs = 60  # seconds
n_samples_nirs = int(t_end_nirs * nirs_fs)
time_nirs = np.arange(n_samples_nirs) / nirs_fs

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for i in range(2):
    ch_label = oxy.clab[i]
    axes[0].plot(time_nirs, oxy.x[:n_samples_nirs, i], lw=0.8, label=f'{ch_label}')
    axes[1].plot(time_nirs, deoxy.x[:n_samples_nirs, i], lw=0.8, label=f'{ch_label}')

axes[0].set_ylabel(f'HbO ({oxy.yUnit})')
axes[0].legend(fontsize=8)
axes[0].set_title('HbO (oxygenated hemoglobin)')
axes[1].set_ylabel(f'HbR ({deoxy.yUnit})')
axes[1].legend(fontsize=8)
axes[1].set_title('HbR (deoxygenated hemoglobin)')
axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'{SUBJECT} NIRS — Raw Signal (first 60 s, first 2 channels)', fontsize=12)
plt.tight_layout()
out_path = plot_dir / 'VP001_nirs_raw.png'
plt.savefig(str(out_path), dpi=120)
plt.close()
print('Saved:', out_path.resolve())
"""

cell8_code = r"""# Cell 8 — Trial-locked NIRS response around first VF trial onset
# NIRS mrk: className = ['VF', 'BL'], event.desc: 2=VF, 1=BL
vf_trial_indices = np.where(mrk_nirs.event.desc == 2)[0]
first_vf_time_ms = mrk_nirs.time[vf_trial_indices[0]]
first_vf_sample = int(first_vf_time_ms / 1000 * nirs_fs)

pre_s, post_s = 5, 20
pre_samples = int(pre_s * nirs_fs)
post_samples = int(post_s * nirs_fs)

start = max(first_vf_sample - pre_samples, 0)
end = min(first_vf_sample + post_samples, oxy.x.shape[0])

time_trial = (np.arange(end - start) - pre_samples) / nirs_fs

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
# Plot all 36 channels faintly, then mean bold
for i in range(nirs_n_oxy):
    axes[0].plot(time_trial, oxy.x[start:end, i], lw=0.4, alpha=0.3, color='red')
    axes[1].plot(time_trial, deoxy.x[start:end, i], lw=0.4, alpha=0.3, color='blue')

axes[0].plot(time_trial, oxy.x[start:end, :].mean(axis=1), lw=2.5, color='darkred', label='mean HbO')
axes[1].plot(time_trial, deoxy.x[start:end, :].mean(axis=1), lw=2.5, color='darkblue', label='mean HbR')

for ax in axes:
    ax.axvline(0, color='k', ls='--', lw=1.5, label='Task onset')
    ax.legend(fontsize=22)
    ax.tick_params(labelsize=20)

axes[0].set_ylabel(f'HbO ({oxy.yUnit})', fontsize=23)
axes[1].set_ylabel(f'HbR ({deoxy.yUnit})', fontsize=23)
axes[-1].set_xlabel('Time relative to task onset (s)', fontsize=23)
plt.tight_layout()
out_path = plot_dir / 'VP001_nirs_trial_response.png'
plt.savefig(str(out_path), dpi=150)
plt.close()
print('Saved:', out_path.resolve())
print(f'First VF trial onset: {first_vf_time_ms} ms ({first_vf_time_ms/1000:.1f} s)')
print(f'Plotted window: {start/nirs_fs:.1f}s to {end/nirs_fs:.1f}s')
"""

cell9_md = """## Summary — Shin 2017 Simultaneous EEG+NIRS Dataset (VP001, verbal fluency task)

### Data Format
- **Format**: BBCI toolbox MATLAB `.mat` files, loaded with `scipy.io.loadmat(squeeze_me=True, struct_as_record=False)`
- **Variable names**: `cnt_vf`, `mrk_vf`, `mnt_vf` (task name appended: `_nback`, `_gonogo`, `_vf`)

### EEG Structure (`cnt_vf`)
- **Fields**: `clab`, `fs`, `title`, `file`, `x`, `T`, `yUnit`
- `cnt_vf.x`: shape `(samples, channels)` — EEG signal matrix
- `cnt_vf.fs`: **200 Hz** sampling rate
- `cnt_vf.clab`: 30 channel labels (standard 10-20 system)
- `cnt_vf.yUnit`: microvolts (uV)

### NIRS Structure (`cnt_vf`)
- **Fields**: `oxy`, `deoxy` (two sub-structs, NOT raw intensity)
- `oxy.x` / `deoxy.x`: shape `(samples, 36)` — already converted to HbO/HbR
- `oxy.fs` / `deoxy.fs`: **10 Hz** sampling rate
- 36 HbO + 36 HbR = **72 channels total**
- Signal type: **HbO/HbR concentrations** (not raw intensity); `yUnit` = mmol*mm

### Marker Structure (`mrk_vf`)
- **Fields**: `time`, `y`, `className`, `event`
- `mrk.time`: trial onset times in **milliseconds**
- `mrk.className`: `['VF', 'BL']` — Word Generation vs. Baseline
- `mrk.event.desc`: integer labels — **2 = VF (word generation), 1 = BL (baseline)** (NIRS mrk)
- EEG mrk uses different encoding: **32 = VF, 16 = BL**
- `mrk.y`: one-hot matrix `(n_classes, n_trials)`

### Confirmed Stats (VP001, vf task)
| Property | Value |
|---|---|
| EEG sampling rate | 200 Hz |
| EEG channels | 30 |
| EEG duration | ~1857 s (~31 min) |
| NIRS sampling rate | 10 Hz |
| NIRS channels | 36 HbO + 36 HbR = 72 |
| NIRS duration | ~2011 s (~33 min) |
| Total trials (vf) | 60 (30 VF + 30 BL) |

### Notable Findings
- NIRS data is already preprocessed to HbO/HbR concentrations (not raw optical intensity)
- EEG and NIRS markers use **different integer encodings** (32/16 vs 2/1) — always use `mrk.className` to disambiguate
- NIRS is stored as nested struct: `cnt_vf.oxy` and `cnt_vf.deoxy`, unlike EEG where `cnt_vf.x` is directly the signal
- The trial-locked NIRS response should show HbO increase and HbR decrease 6–10 s post-stimulus (canonical hemodynamic response)
"""

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_code_cell(cell1_code),
    nbformat.v4.new_code_cell(cell2_code),
    nbformat.v4.new_code_cell(cell3_code),
    nbformat.v4.new_code_cell(cell4_code),
    nbformat.v4.new_code_cell(cell5_code),
    nbformat.v4.new_code_cell(cell6_code),
    nbformat.v4.new_code_cell(cell7_code),
    nbformat.v4.new_code_cell(cell8_code),
    nbformat.v4.new_markdown_cell(cell9_md),
]

out_path = 'C:/Users/ahars/Desktop/Funstuff/coding/neuro-work/foundation-brain/notebooks/01_dataset_exploration.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook written to', out_path)
