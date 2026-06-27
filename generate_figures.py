"""
Generates the Foundation Brain pipeline diagram for the manuscript.

The dataset montage and task-timeline figures are NOT generated here --
we use the real, openly-licensed Figure 1 from Shin et al. 2018 (Scientific
Data, CC-BY) directly, fetched and stored as
figures/shin2018_fig1_montage_and_tasks.png. See also
figures/shin2018_fig2_eeg_erp.jpg and figures/shin2018_fig3_nirs_hrf.jpg for
EEG ERP/topomap and fNIRS hemodynamic-response figures from the same source.

Run from project root: python generate_figures.py
Outputs: figures/pipeline_diagram.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def plot_pipeline_diagram():
    """Foundation Brain pretraining + linear probe pipeline, schematic block diagram."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, text, color='#dde8f5'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                       facecolor=color, edgecolor='black', linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.5, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color='black', linewidth=1.2))

    box(0.2, 2.5, 1.6, 1.0, 'EEG window\n(30, 2000)', '#f5e6c8')
    box(0.2, 0.5, 1.6, 1.0, 'fNIRS window\n(72, 100)', '#f5e6c8')
    box(2.3, 2.5, 1.8, 1.0, 'EEG Encoder\n(ShallowConvNet)', '#dde8f5')
    box(2.3, 0.5, 1.8, 1.0, 'fNIRS Encoder\n(CNN)', '#dde8f5')
    box(4.7, 2.5, 1.6, 1.0, 'Proj. Head\n-> z_eeg', '#e3f0db')
    box(4.7, 0.5, 1.6, 1.0, 'Proj. Head\n-> z_nirs', '#e3f0db')
    box(6.8, 1.5, 1.9, 1.0, 'NT-Xent\n(symmetric InfoNCE)', '#f5d6d6')
    box(9.0 - 1.4, 3.6, 1.4, 0.35, 'Pretraining (no labels)', 'none')

    arrow(1.8, 3.0, 2.3, 3.0)
    arrow(1.8, 1.0, 2.3, 1.0)
    arrow(4.1, 3.0, 4.7, 3.0)
    arrow(4.1, 1.0, 4.7, 1.0)
    arrow(6.3, 3.0, 6.8, 2.2)
    arrow(6.3, 1.0, 6.8, 1.8)

    out_path = os.path.join(OUT_DIR, 'pipeline_diagram.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    plot_pipeline_diagram()
