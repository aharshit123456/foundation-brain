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

NOTE: this figure is placed as a two-column-spanning `figure*` in the IEEEtran
manuscript (see manuscript.tex), so it renders at close to full page width
(~7 in) rather than single-column width (~3.3 in). Font sizes below assume
that wide placement -- do not shrink the LaTeX \\includegraphics width without
also enlarging the boxes/fonts here, or text will clip again.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def plot_pipeline_diagram():
    """Foundation Brain pretraining + frozen linear-probe pipeline, detailed schematic."""
    fig, ax = plt.subplots(figsize=(15.2, 8.4))
    ax.axis('off')
    ax.set_xlim(0, 17.9)
    ax.set_ylim(0, 14.8)

    def box(x, y, w, h, text, color='#dde8f5', fontsize=11.5, weight='normal', ls='-'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                        facecolor=color, edgecolor='black',
                                        linewidth=1.2, linestyle=ls)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                 fontsize=fontsize, fontweight=weight)

    def arrow(x0, y0, x1, y1, style='-|>', lw=1.4, color='black'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle=style, color=color, linewidth=lw))

    def stage_label(x, y, text, fontsize=10.5):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                 style='italic', color='#444444')

    # ------------------------------------------------------------------
    # Section band headers
    # ------------------------------------------------------------------
    ax.add_patch(mpatches.FancyBboxPatch((0.15, 7.5), 17.6, 7.0, boxstyle='round,pad=0.05',
                                          facecolor='#fbfbf5', edgecolor='#999999', linewidth=1.0))
    ax.text(0.5, 14.15, 'A. Contrastive Pretraining (no task labels)', ha='left', va='center',
            fontsize=14.0, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((0.15, 0.2), 17.6, 6.9, boxstyle='round,pad=0.05',
                                          facecolor='#f7fafd', edgecolor='#999999', linewidth=1.0))
    ax.text(0.5, 6.75, 'B. Frozen Deployment (linear probe -- task labels used here only)',
            ha='left', va='center', fontsize=14.0, fontweight='bold')

    # ==================================================================
    # A. PRETRAINING (top band): raw window -> temporal conv -> spatial
    #    conv -> BN+square+log -> embedding -> proj. head -> L2-norm -> NT-Xent
    # ==================================================================
    top_y_eeg = 11.65
    top_y_nirs = 8.55
    h = 1.35

    x0 = 0.5
    w_raw = 1.55
    box(x0, top_y_eeg, w_raw, h, 'EEG window\n(30, 2000)', '#f5e6c8', fontsize=10.5)
    box(x0, top_y_nirs, w_raw, h, 'fNIRS window\n(72, 100)', '#f5e6c8', fontsize=10.5)

    x1 = x0 + w_raw + 0.4
    w_stage1 = 1.85
    box(x1, top_y_eeg, w_stage1, h, 'Temporal Conv\n(freq. filters)', '#dde8f5', fontsize=10.0)
    box(x1, top_y_nirs, w_stage1, h, 'Temporal Conv\n(NIRS samples)', '#dde8f5', fontsize=10.0)

    x2 = x1 + w_stage1 + 0.35
    w_stage2 = 1.95
    box(x2, top_y_eeg, w_stage2, h, 'Spatial Conv\n(30 electrodes)', '#dde8f5', fontsize=10.0)
    box(x2, top_y_nirs, w_stage2, h, 'Spatial Conv\n(72 optode ch.)', '#dde8f5', fontsize=10.0)

    x3 = x2 + w_stage2 + 0.35
    w_stage3 = 2.05
    box(x3, top_y_eeg, w_stage3, h, 'BatchNorm ->\nSquare -> AvgPool -> Log', '#dde8f5', fontsize=9.3)
    box(x3, top_y_nirs, w_stage3, h, 'BatchNorm ->\nSquare -> AvgPool -> Log', '#dde8f5', fontsize=9.3)

    x4 = x3 + w_stage3 + 0.35
    w_emb = 1.55
    box(x4, top_y_eeg, w_emb, h, r'$e_{\mathrm{eeg}} \in \mathbb{R}^{128}$', '#e8f2e2', fontsize=11.5)
    box(x4, top_y_nirs, w_emb, h, r'$e_{\mathrm{nirs}} \in \mathbb{R}^{128}$', '#e8f2e2', fontsize=11.5)

    x5 = x4 + w_emb + 0.35
    w_proj = 1.65
    box(x5, top_y_eeg, w_proj, h, 'Proj. Head\n(2-layer MLP)', '#e3f0db', fontsize=10.0)
    box(x5, top_y_nirs, w_proj, h, 'Proj. Head\n(2-layer MLP)', '#e3f0db', fontsize=10.0)

    x6 = x5 + w_proj + 0.35
    w_z = 1.35
    box(x6, top_y_eeg, w_z, h, r'$z_{\mathrm{eeg}}$' + '\nL2-norm', '#dcefe8', fontsize=10.5)
    box(x6, top_y_nirs, w_z, h, r'$z_{\mathrm{nirs}}$' + '\nL2-norm', '#dcefe8', fontsize=10.5)

    # NT-Xent block with small similarity-matrix glyph
    x7 = x6 + w_z + 0.45
    nce_w, nce_h = 2.55, 4.35
    nce_y = (top_y_nirs + top_y_eeg + h - nce_h) / 2 + 0.05
    box(x7, nce_y, nce_w, nce_h, '', '#f5d6d6')
    ax.text(x7 + nce_w / 2, nce_y + nce_h - 0.4, 'NT-Xent', ha='center', va='center',
            fontsize=12.5, fontweight='bold')
    ax.text(x7 + nce_w / 2, nce_y + nce_h - 0.78, '(symmetric InfoNCE)', ha='center', va='center',
            fontsize=9.3)
    # small NxN similarity matrix glyph with highlighted diagonal
    n_g = 5
    cell = 0.3
    gx0 = x7 + nce_w / 2 - (n_g * cell) / 2
    gy0 = nce_y + nce_h / 2 - (n_g * cell) / 2 + 0.35
    for i in range(n_g):
        for j in range(n_g):
            c = '#c23b3b' if i == j else '#f0caca'
            ax.add_patch(mpatches.Rectangle((gx0 + j * cell, gy0 + (n_g - 1 - i) * cell),
                                             cell, cell, facecolor=c, edgecolor='#883030', linewidth=0.5))
    ax.text(x7 + nce_w / 2, gy0 - 0.35, 'similarity matrix,\ndiagonal = positives',
            ha='center', va='center', fontsize=8.6, color='#444444')
    ax.text(x7 + nce_w / 2, nce_y + 0.45,
            r'$\mathcal{L}=\frac{1}{2}[\mathcal{L}_{e \to n}+\mathcal{L}_{n \to e}]$'
            '\n' r'$\tau=0.07$, no class label used',
            ha='center', va='center', fontsize=8.6)

    # arrows along pretraining rows
    xs = [x0, x1, x2, x3, x4, x5, x6]
    ws = [w_raw, w_stage1, w_stage2, w_stage3, w_emb, w_proj, w_z]
    for xi, wi in zip(xs, ws):
        arrow(xi + wi, top_y_eeg + h / 2, xi + wi + 0.35, top_y_eeg + h / 2)
        arrow(xi + wi, top_y_nirs + h / 2, xi + wi + 0.35, top_y_nirs + h / 2)
    arrow(x6 + w_z, top_y_eeg + h / 2, x7, nce_y + nce_h - 1.1)
    arrow(x6 + w_z, top_y_nirs + h / 2, x7, nce_y + 1.1)

    stage_label((x1 + x1 + w_stage1) / 2 + 1.4, top_y_eeg + h + 0.4,
                'shared encoder design, separate weights per modality', fontsize=10.8)

    # ==================================================================
    # B. FROZEN DEPLOYMENT (bottom band)
    # ==================================================================
    dep_y_eeg = 4.85
    dep_y_nirs = 2.15
    dh = 1.35

    box(x0, dep_y_eeg, w_raw, dh, 'EEG window\n(30, 2000)', '#f5e6c8', fontsize=10.5)
    box(x0, dep_y_nirs, w_raw, dh, 'fNIRS window\n(72, 100)', '#f5e6c8', fontsize=10.5)

    xf1 = x0 + w_raw + 0.6
    wf = 3.2
    box(xf1, dep_y_eeg, wf, dh, 'EEG Encoder (frozen)\nweights fixed after A', '#cfd9ea', fontsize=10.5, ls='--')
    box(xf1, dep_y_nirs, wf, dh, 'fNIRS Encoder (frozen)\nweights fixed after A', '#cfd9ea', fontsize=10.5, ls='--')

    xf2 = xf1 + wf + 0.55
    w_embf = 1.6
    box(xf2, dep_y_eeg, w_embf, dh, r'$e_{\mathrm{eeg}}$', '#e8f2e2', fontsize=12.5)
    box(xf2, dep_y_nirs, w_embf, dh, r'$e_{\mathrm{nirs}}$', '#e8f2e2', fontsize=12.5)

    ax.text(xf2 + w_embf + 0.85, (dep_y_eeg + dep_y_nirs + dh) / 2, 'proj. heads\ndiscarded',
            ha='center', va='center', fontsize=10.2, color='#883030', style='italic')

    xf3 = xf2 + w_embf + 1.75
    w_concat = 1.65
    concat_h = dep_y_eeg + dh - dep_y_nirs
    box(xf3, dep_y_nirs, w_concat, concat_h, 'concat\n' + r'$\in \mathbb{R}^{256}$', '#efe7d8', fontsize=11.8)

    xf4 = xf3 + w_concat + 0.6
    w_probe = 2.75
    box(xf4, dep_y_nirs + concat_h / 2 - 0.68, w_probe, 1.35, 'Linear Probe\n(logistic regression)', '#f5d6d6', fontsize=11.5)

    xf5 = xf4 + w_probe + 0.55
    w_pred = 2.15
    box(xf5, dep_y_nirs + concat_h / 2 - 0.68, w_pred, 1.35, 'Task label\nprediction', '#e3ddf0', fontsize=11.5)

    arrow(x0 + w_raw, dep_y_eeg + dh / 2, xf1, dep_y_eeg + dh / 2)
    arrow(x0 + w_raw, dep_y_nirs + dh / 2, xf1, dep_y_nirs + dh / 2)
    arrow(xf1 + wf, dep_y_eeg + dh / 2, xf2, dep_y_eeg + dh / 2)
    arrow(xf1 + wf, dep_y_nirs + dh / 2, xf2, dep_y_nirs + dh / 2)
    arrow(xf2 + w_embf, dep_y_eeg + dh / 2, xf3, dep_y_nirs + concat_h * 0.75)
    arrow(xf2 + w_embf, dep_y_nirs + dh / 2, xf3, dep_y_nirs + concat_h * 0.25)
    arrow(xf3 + w_concat, dep_y_nirs + concat_h / 2, xf4, dep_y_nirs + concat_h / 2)
    arrow(xf4 + w_probe, dep_y_nirs + concat_h / 2, xf5, dep_y_nirs + concat_h / 2)

    ax.text((xf4 + xf4 + w_probe) / 2, dep_y_nirs - 0.55,
            'only this layer is fit with task labels; encoders never see labels or gradients',
            ha='center', va='center', fontsize=9.8, color='#444444')

    # Legend-style bridge annotation between panels
    arrow(1.25, 7.35, 1.25, 6.95, style='-|>', lw=1.6, color='#555555')
    ax.text(1.65, 7.15, 'weights copied, heads dropped', ha='left', va='center',
            fontsize=9.8, color='#555555', style='italic')

    out_path = os.path.join(OUT_DIR, 'pipeline_diagram.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    plot_pipeline_diagram()
