"""Script to create notebooks/06_colab_extras.ipynb -- T4 GPU additions.

Runs the experiments that benefit from a faster/different GPU than this
project's local Apple Silicon (MPS) machine:
  1. No-pretraining control (random-init backbone + linear probe) -- the
     key de-risking control for the paper's central claim.
  2. Larger-batch-size NT-Xent pretraining (MPS-constrained batch sizes are
     small; a T4 has more VRAM headroom for harder in-batch negatives).
  3. Longer pretraining schedule (more epochs) on the same 23-subject VF
     dataset, to check whether the modest cross-task transfer result is
     pretraining-budget-limited rather than architecture-limited.

Clones the repo fresh, re-downloads the dataset (network/disk on Colab is
ephemeral), and writes all results to data/processed/*.json so they can be
pulled back down and merged into the manuscript locally.
"""
import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell("""\
# Foundation Brain — Colab T4 Extras

Runs the experiments that benefit from a faster/different GPU than the
project's local Apple Silicon (MPS) machine:

1. **No-pretraining control** — random-init backbone + linear probe. The key
   control for the paper's central claim: if this scores near chance while
   the NT-Xent-pretrained backbone scores meaningfully above it, that
   substantiates the pretraining is doing real work.
2. **Larger-batch-size pretraining** — NT-Xent quality scales with batch
   size (more in-batch negatives per anchor); MPS-constrained batch sizes
   were small (64), a T4 has headroom for 128–256.
3. **Longer pretraining schedule** — more epochs on the same 23-subject VF
   dataset, to check whether modest cross-task transfer is a pretraining-
   budget limitation rather than an architecture limitation.

**Runtime:** Set Runtime → Change runtime type → T4 GPU before running.
"""))

cells.append(nbformat.v4.new_code_cell("""\
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""))

cells.append(nbformat.v4.new_markdown_cell("## 1. Clone repo and install dependencies"))

cells.append(nbformat.v4.new_code_cell("""\
!git clone https://github.com/aharshit123456/foundation-brain.git
%cd foundation-brain
!pip install -q numpy scipy torch scikit-learn tqdm
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
## 2. Download dataset (all 23 subjects: VP001-011, VP014-025)

This re-downloads from the TU Berlin institutional repository — Colab's
disk is ephemeral, so we can't reuse the local Mac's already-downloaded
files. Expect ~15-20 minutes depending on Colab's network to the source
server.
"""))

cells.append(nbformat.v4.new_code_cell("""\
SUBJECTS = [f'VP{str(i).zfill(3)}' for i in range(1, 26) if i not in (12, 13)]
!python3 data/download_dataset.py {' '.join(SUBJECTS)}
"""))

cells.append(nbformat.v4.new_code_cell("""\
!python3 preprocess_vf.py
!python3 preprocess_nback.py
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
## 3. No-Pretraining Control (the key addition)

Same LOSO protocol as `run_foundation_loso.py`, but the backbone is never
contrastively pretrained — random init, straight to the frozen linear probe.
If this scores near chance (~0.50 for the binary VF task) while the
pretrained backbone scores meaningfully higher, that's direct evidence the
NT-Xent objective is contributing real structure, not just providing a
fixed random projection that any logistic regression could exploit equally
well.
"""))

cells.append(nbformat.v4.new_code_cell("""\
!python3 run_foundation_nopretrain.py
"""))

cells.append(nbformat.v4.new_code_cell("""\
import json
with open('data/processed/foundation_nopretrain_results.json') as f:
    nopretrain = json.load(f)
print(f"No-pretraining control: Acc={nopretrain['mean_acc']:.3f}  F1={nopretrain['mean_f1']:.3f}")
print(f"(Compare to chance=0.500 and the pretrained Foundation Brain result)")
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
## 4. Larger Batch Size Pretraining (128 vs. the local default of 64)

NT-Xent's negative set is every other sample in the batch — bigger batches
give the contrastive loss more (and harder) negatives per anchor, typically
improving representation quality. This was VRAM-constrained on the local
Mac's MPS backend; a T4 (16GB) has room for this.
"""))

cells.append(nbformat.v4.new_code_cell("""\
import pickle
import sys
sys.path.insert(0, '.')
from run_foundation_loso import run_loso

with open('data/processed/dataset_vf_windows.pkl', 'rb') as f:
    all_data = pickle.load(f)

results_bs128, mean_acc_bs128, mean_f1_bs128 = run_loso(
    all_data, n_pretrain_epochs=50, batch_size=128, lr=3e-4, temperature=0.07
)
print(f"Batch size 128: Acc={mean_acc_bs128:.3f}  F1={mean_f1_bs128:.3f}")
"""))

cells.append(nbformat.v4.new_code_cell("""\
import json, os

output_bs128 = {
    'model': 'Foundation Brain (batch_size=128 ablation)',
    'dataset': 'Shin 2017 Dataset C (VF task)',
    'n_subjects': len(all_data),
    'evaluation': 'LOSO -- larger batch size ablation',
    'batch_size': 128,
    'n_pretrain_epochs': 50,
    'mean_acc': round(mean_acc_bs128, 4),
    'mean_f1': round(mean_f1_bs128, 4),
    'per_subject': {s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
                    for s, v in results_bs128.items()}
}
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/foundation_results_bs128.json', 'w') as f:
    json.dump(output_bs128, f, indent=2)
print('Saved -> data/processed/foundation_results_bs128.json')
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
## 5. Longer Pretraining Schedule (150 epochs vs. the local default of 50)

Tests whether the modest cross-task transfer result (n-back probe only
slightly above chance) is a pretraining-budget limitation -- contrastive
objectives are notoriously epoch-hungry -- rather than an architecture
limitation.
"""))

cells.append(nbformat.v4.new_code_cell("""\
results_ep150, mean_acc_ep150, mean_f1_ep150 = run_loso(
    all_data, n_pretrain_epochs=150, batch_size=64, lr=3e-4, temperature=0.07
)
print(f"150 epochs: Acc={mean_acc_ep150:.3f}  F1={mean_f1_ep150:.3f}")
"""))

cells.append(nbformat.v4.new_code_cell("""\
output_ep150 = {
    'model': 'Foundation Brain (150-epoch pretraining ablation)',
    'dataset': 'Shin 2017 Dataset C (VF task)',
    'n_subjects': len(all_data),
    'evaluation': 'LOSO -- longer pretraining schedule ablation',
    'batch_size': 64,
    'n_pretrain_epochs': 150,
    'mean_acc': round(mean_acc_ep150, 4),
    'mean_f1': round(mean_f1_ep150, 4),
    'per_subject': {s: {'acc': round(v['acc'], 4), 'f1': round(v['f1'], 4)}
                    for s, v in results_ep150.items()}
}
with open('data/processed/foundation_results_ep150.json', 'w') as f:
    json.dump(output_ep150, f, indent=2)
print('Saved -> data/processed/foundation_results_ep150.json')
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
## 6. Download results to merge locally

Run this cell and download the zip, then unzip into `data/processed/` in
the local repo before re-running `generate_paper_tex.py`.
"""))

cells.append(nbformat.v4.new_code_cell("""\
!zip -j colab_extras_results.zip \\
    data/processed/foundation_nopretrain_results.json \\
    data/processed/foundation_results_bs128.json \\
    data/processed/foundation_results_ep150.json

from google.colab import files
files.download('colab_extras_results.zip')
"""))

nb['cells'] = cells

with open('notebooks/06_colab_extras.ipynb', 'w') as f:
    nbformat.write(nb, f)

print('Wrote notebooks/06_colab_extras.ipynb')
