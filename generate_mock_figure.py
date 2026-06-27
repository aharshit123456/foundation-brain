"""
Generates a clearly-labeled MOCK PLACEHOLDER figure for results that have not
finished running yet, so the manuscript can be assembled end-to-end before all
experiments complete. Re-run generate_paper_tex.py once the real results land
(via run_efnet_subjectdep.py / run_efnet_loso.py --modality ...) to swap this
out automatically -- the generator checks for the real JSON files first.

Run from project root: python generate_mock_figure.py
Outputs: figures/mock_pending_result.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def plot_mock(label, out_name='mock_pending_result.png'):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.axis('off')
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                         edgecolor='gray', linestyle='--', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.55, 'PLACEHOLDER', ha='center', va='center',
            fontsize=16, color='gray', fontweight='bold', alpha=0.7)
    ax.text(0.5, 0.35, label, ha='center', va='center',
            fontsize=9, color='dimgray', wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    plot_mock('Experiment pending completion.\nFigure will be generated automatically\nfrom run output once available.')
