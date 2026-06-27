import nbformat

nb = nbformat.v4.new_notebook()

cells = []

# ---- Cell 1: Title ----
cells.append(nbformat.v4.new_markdown_cell("""# Notebook 04: STA-Net Architectural Analysis

**Purpose**: Read-only architectural analysis of STA-Net (Liu et al., Information Fusion 2025) for Phase 2 backbone design insights.

**Why read-only?** STA-Net requires TensorFlow 2.10 + Python 3.9. Our environment runs Python 3.13 which is incompatible. No execution is attempted — all analysis is from source code inspection.

**Reference**: Liu, Mutian et al. "STA-Net: Spatial–temporal alignment network for hybrid EEG-fNIRS decoding." *Information Fusion* 2025. [Paper](https://www.sciencedirect.com/science/article/pii/S156625352500096X)
"""))

# ---- Cell 2: Repo structure ----
cells.append(nbformat.v4.new_markdown_cell("""## 1. Repository Structure

```
baselines/STA-Net/
├── README.md                        # Citation, dataset links, requirements
├── requirements_stanet.txt          # TF 2.10, Python 3.9
├── sta.py                           # Core model definition (all architecture here)
├── run_sta_net.py                   # Training loop (3-session leave-one-out CV)
└── preprocessing/
    ├── preprocessing_order.txt      # Pipeline order documentation
    ├── load_mat.py                  # Load raw .mat BBCI dataset files
    ├── preprocessing.py             # Bandpass EEG, ICA artefact removal, fNIRS bandpass
    ├── epoch.py                     # Epoch extraction (60 trials per subject)
    ├── to3d.py                      # Electrode/optode → 16×16 spatial image (cubic interpolation)
    └── window.py                    # Sliding windows + fNIRS lag assembly → model inputs
```

**Dataset**: BBCI simultaneous EEG-fNIRS BCI dataset (TU Berlin).
- EEG: 28 channels, 200 Hz (later reduced to 26 after re-referencing)
- fNIRS: 36 channels (HbO + HbR), 10 Hz
- 60 trials per subject, 3 sessions, binary classification task
"""))

# ---- Cell 3: Input shapes ----
cells.append(nbformat.v4.new_markdown_cell("""## 2. Model Input Shapes

From `sta.py` line 250-251:

```python
eeg_input   = keras.Input(shape=(16, 16, 600, 1),        name="eeg_input")
fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2),     name="fnirs_input")
```

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `eeg_input` | `(batch, 16, 16, 600, 1)` | 16×16 spatial image × 600 timepoints (3 s × 200 Hz) × 1 channel |
| `fnirs_input` | `(batch, 11, 16, 16, 30, 2)` | 11 lag windows × 16×16 spatial × 30 timepoints (3 s × 10 Hz) × 2 (HbO, HbR) |

**Key observation**: The temporal mismatch (200 Hz vs 10 Hz = 20:1 ratio within a 3-second window) is addressed architecturally, not by resampling. fNIRS is given 11 lag windows to cover a broader temporal range that compensates for hemodynamic response lag.
"""))

# ---- Cell 4: FGSA ----
cells.append(nbformat.v4.new_markdown_cell("""## 3. FGSA — Fine-Grained Spatial Alignment

### What it is
FGSA is not a standalone layer name in the code — it is implemented as the `fga` class (Fine-Grained Alignment) inside `conv_block`. It operates after separate EEG and fNIRS convolutional feature extractors.

### How spatial alignment works (from `to3d.py`)
Before the model even sees data, both EEG electrodes (28 channels) and fNIRS optodes (36 channels) are projected onto a **shared 16×16 scalp topographic grid** via cubic spline interpolation using standard 10-05 electrode coordinate system positions.

- EEG: 28 known positions → cubic interpolation fills 228 unknown positions → 16×16 image
- fNIRS: 36 known positions → cubic interpolation fills 220 unknown positions → 16×16 image (done separately for HbO and HbR)

This creates **spatial correspondence**: pixel `(row, col)` in the EEG image represents the same scalp location as pixel `(row, col)` in the fNIRS image.

### How FGSA aligns features (from `fga.call`, lines 126-156)
```python
def call(self, inputs):
    eeg_fusion, eeg, fnirs = inputs

    # Step 1: Pool fNIRS across channels with a 3D conv
    fnirs_attention = self.channel_pooling(fnirs)          # Conv3D → 1 filter

    # Step 2: Temporal average pooling → spatial attention map
    fnirs_attention_map = self.tap_fnirs(fnirs_attention)  # mean over time dim
    fnirs_attention_map = tf.math.reduce_mean(...)         # mean over lag dim

    # Step 3: Sigmoid → spatial weights in [0, 1]
    fnirs_attention_map_norm = keras.activations.sigmoid(fnirs_attention_map)

    # Step 4: Multiply EEG fusion features by fNIRS spatial weights
    eeg_fusion_guided = tf.math.multiply(eeg_fusion, fnirs_attention_map_norm)

    # Step 5: Learnable residual blend of original EEG + fNIRS-guided EEG
    eeg_add = residual_para * eeg + (1 - residual_para) * eeg_fusion
    fga_feature = eeg_fusion_guided + eeg_add
```

**Alignment auxiliary loss**: Pearson correlation between the EEG spatial map and the fNIRS attention map is computed and added as `1 - pearson_r` loss. This forces the fNIRS-derived spatial attention to be correlated with EEG spatial activity.

### Summary
FGSA uses fNIRS **spatial activation patterns** (after pooling across time and lag) as a soft mask to re-weight EEG features. It does NOT learn cross-channel correspondences dynamically — the spatial alignment is pre-baked via topographic interpolation, and FGSA refines which scalp regions to attend to based on fNIRS hemodynamic evidence.
"""))

# ---- Cell 5: EGTA ----
cells.append(nbformat.v4.new_markdown_cell("""## 4. EGTA — EEG-fNIRS Global Temporal Alignment

### What it is
EGTA is implemented as the `e_f_attention` class (lines 53-96). It is the cross-modal attention layer applied after both streams have been spatially compressed by two `conv_block` layers and global average pooled.

### How it handles temporal mismatch
The 200 Hz : 10 Hz sampling rate difference (20:1) is handled in **two complementary stages**:

1. **Preprocessing (window.py)**: Rather than resampling, fNIRS is assembled as 11 overlapping lag windows, each 3 s long (30 samples at 10 Hz). Window `w` of fNIRS corresponds to the same-time EEG window but fNIRS windows `w` through `w+10` span an 11-second range. This encodes the **hemodynamic response lag** (peak ~5-7 s after stimulus) into the input tensor shape.

2. **Architecture (Conv3D strides)**: The two `conv_block` layers apply strided 3D convolutions that downsample the temporal dimension independently per modality:
   - EEG block 1: stride (2,2,6) on 600 frames → ~100 frames
   - EEG block 2: stride (2,2,2) on ~100 frames → ~50 frames
   - fNIRS block 1: stride (2,2,2) on 30 frames → ~15 frames
   - fNIRS block 2: stride (2,2,2) on ~15 frames → ~8 frames
   - After Global Average Pooling (gap layer), both are collapsed to single vectors.

### The cross-attention mechanism (e_f_attention.call, lines 70-96)
```python
def call(self, inputs):
    eeg, fnirs = inputs

    # EEG → flattened → single query vector
    q_eeg = self.q_flat(eeg)                         # flatten spatial+temporal
    fusion_output = self.fusion_proj(q_eeg)           # save for fusion path

    q_eeg = self.q_proj(q_eeg)
    q_eeg = tf.expand_dims(q_eeg, axis=1)             # shape: (B, 1, emb_size)

    # fNIRS → reshape lag dim as sequence → keys & values
    k_fnirs = self.k_flat(fnirs)                      # shape: (B, 11, -1)
    k_fnirs = self.pos(k_fnirs)                       # add learnable positional embedding
    k_fnirs = self.k_proj(k_fnirs)                    # shape: (B, 11, emb_size)

    # EEG queries fNIRS lag windows
    fnirs_weighted, attn_weights = self.dot_product_attention(q_eeg, k_fnirs)

    # Auxiliary loss: EEG embedding and attended fNIRS embedding should be correlated
    ef_loss = pearson_r(q_eeg_mean, fnirs_weighted_mean)
    self.add_loss(1 - ef_loss)

    return fusion_output, fnirs_weighted, attn_weights
```

**Key insight**: The EEG representation acts as a single query that attends over the 11 fNIRS lag windows. This lets the model learn which temporal lag of hemodynamic response is most informative for the current cognitive state. The positional embedding on fNIRS encodes lag order, and the Pearson auxiliary loss forces temporal alignment by making the EEG and attended-fNIRS embeddings correlated.
"""))

# ---- Cell 6: Classification head ----
cells.append(nbformat.v4.new_markdown_cell("""## 5. Classification Head

From `sta_net()` lines 273-306:

The model produces **two outputs** and uses a **learned prediction weighting**:

```python
# Three parallel prediction heads
eeg_pred       = Dense(2)(eeg_feature)           # pure EEG branch
eegfusion_pred = Dense(2)(eegfusion_feature)     # EEG+fNIRS fusion branch
fnirs_pred     = Dense(2)(fnirs_feature)         # fNIRS branch (from cross-attention)

# Softmax on each
eeg_pred       = softmax → named 'eeg_output'    # auxiliary output for training
eegfusion_pred = softmax
fnirs_pred     = softmax

# Learned dynamic weighting between fusion and fNIRS predictions
fnirs_p_weight    = Dense(1)(fnirs_feature)
eegfusion_p_weight= Dense(1)(eegfusion_feature)
p_weight = softmax([eegfusion_p_weight, fnirs_p_weight])   # (B, 2) weights

# Weighted combination of fusion and fNIRS predictions
the_pred = sum( stack([eegfusion_pred, fnirs_pred]) * p_weight )
# final output: named 'class_output'

model outputs = [the_pred, eeg_pred]   # two supervised outputs
```

**Training**: 2-phase training
1. Phase 1: train with validation split, early stopping on `val_class_output_loss`
2. Phase 2: retrain on full dataset until training loss matches Phase 1 best val loss

**Loss**: categorical cross-entropy on both outputs + Pearson auxiliary losses from FGSA and EGTA layers (4 auxiliary losses total: `fgsa1_plcc`, `fgsa2_plcc`, `ef_plcc`, all formulated as `1 - pearson_r`).

**Dataset results (from paper)**: Binary mental workload classification on BBCI dataset. STA-Net reported state-of-the-art accuracy outperforming prior EEG-only, fNIRS-only, and naive fusion baselines.
"""))

# ---- Cell 7: Design insights ----
cells.append(nbformat.v4.new_markdown_cell("""## 6. Three Key Design Insights for Phase 2

---

### Insight 1: Topographic 2D projection as a universal spatial alignment strategy

**What STA-Net does**: Before any neural network computation, both EEG (28 channels) and fNIRS (36 channels) are independently projected onto a shared 16×16 scalp topographic grid using cubic spline interpolation with standard 10-05 coordinate positions. This creates pixel-level spatial correspondence — position (r, c) means the same scalp location for both modalities.

**Why it matters**: EEG electrodes and fNIRS optodes physically sample overlapping scalp regions but with different spatial densities and layouts. The 2D projection creates a common spatial vocabulary without requiring any learned alignment. The Conv3D operations can then exploit local spatial structure (adjacent activations in the 16×16 grid are anatomically adjacent on the scalp).

**Phase 2 implication**: Our contrastive backbone could use the same 2D projection as a preprocessing step, or alternatively use positional encodings based on 3D electrode coordinates (x, y, z in MNI space) fed into a spatial attention module. The 2D grid approach is simple and effective, but a coordinate-conditioned cross-attention may be more flexible. For the BBCI dataset we already have, we should verify whether the same 28 EEG / 36 fNIRS channel layout applies. The key decision: **shared spatial coordinate system** vs **separate modality-specific representations with learned cross-modal alignment**.

---

### Insight 2: EEG queries fNIRS lag windows via cross-attention (temporal alignment)

**What STA-Net does**: Rather than resampling fNIRS to match EEG's 200 Hz, the pipeline stores 11 consecutive 3-second fNIRS windows as a lag sequence (shape: 11 × 16 × 16 × 30 × 2). A single EEG query vector then attends over these 11 lag positions using multi-head cross-attention, learning which hemodynamic response lag is most correlated with the EEG cognitive state. Positional embeddings on the fNIRS lag sequence encode temporal order.

**Why it matters**: The hemodynamic response peaks 5-7 seconds after a stimulus, creating a fundamental temporal offset between neural (EEG) and vascular (fNIRS) signals. STA-Net learns this offset end-to-end rather than imposing a fixed delay correction.

**Phase 2 implication**: Our contrastive loss training pairs (EEG window, fNIRS window) must account for this lag. Options: (a) use STA-Net's approach of providing multiple lag windows as input; (b) pre-align windows during preprocessing with a fixed lag offset (simpler, may be sufficient for contrastive learning); (c) use cross-attention in the fusion module with both modalities as query and key. For a backbone that produces **separate** modality embeddings, option (b) is simpler and more appropriate — we can bake in a fixed 5s NIRS lag at the data loading stage.

---

### Insight 3: Auxiliary Pearson correlation losses as alignment supervision

**What STA-Net does**: In both the spatial (FGSA) and temporal (EGTA) alignment layers, STA-Net adds auxiliary losses computed as `1 - pearson_r(eeg_representation, fnirs_representation)`. These are added directly to the model's loss via `self.add_loss()`. The Pearson correlation is computed between EEG spatial activations and fNIRS-derived attention maps (FGSA), and between EEG embeddings and attended fNIRS embeddings (EGTA).

**Why it matters**: The auxiliary losses act as alignment regularizers — they explicitly pull the learned EEG and fNIRS representations toward each other in a modality-invariant direction. This is a form of **supervised cross-modal alignment** that sits alongside the classification objective.

**Phase 2 implication**: This is remarkably similar to what contrastive learning does, but STA-Net uses Pearson correlation while we plan to use InfoNCE / NT-Xent. The key insight is that **cross-modal alignment as an auxiliary objective works**. For our backbone: the contrastive loss between EEG embeddings and fNIRS embeddings of the same trial IS the alignment objective — we are designing this more principled than STA-Net's Pearson proxy. Additionally, STA-Net's Pearson losses cannot distinguish different trials (they operate batch-mean), whereas our contrastive loss uses negative pairs. This is a direct improvement.
"""))

# ---- Cell 8: Key limitation ----
cells.append(nbformat.v4.new_markdown_cell("""## 7. Key Limitation: STA-Net Cannot Produce Separate Modality Embeddings

This is the **fundamental architectural gap** our Foundation Brain backbone addresses.

### Evidence from the code

STA-Net fuses modalities at the very first layer. From `sta_net()` line 253-256:

```python
# conv_block receives eeg_input as BOTH eegfusion AND eeg:
eegfusion1, eeg1, fnirs1 = conv_block(...)(
    (eeg_input, eeg_input, fnirs_input)   # <-- eeg_input fills the "fusion" slot
)
```

Inside `conv_block.call` (lines 180-197), three separate convolutions run in parallel, but **the FGSA layer immediately fuses them**:
```python
eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature))
# returns a single tensor — EEG features guided by fNIRS spatial attention
```

After the second conv_block, cross-modal attention (`e_f_attention`) fuses again:
```python
eegfusion_feature, fnirs_feature, _ = e_f_attention(...)((eegfusion2, fnirs2))
# eegfusion_feature is ALREADY a cross-modal representation
```

The final classification (lines 283-304) weights between `eegfusion_pred` and `fnirs_pred` — but `eegfusion_feature` was computed FROM fNIRS (via EGTA cross-attention). There is no clean EEG-only embedding at inference time.

### Why this matters for our project

| Capability | STA-Net | Foundation Brain Backbone |
|------------|---------|--------------------------|
| Classify with both modalities | Yes | Yes |
| Classify with EEG only (missing fNIRS) | No — fNIRS attention required in FGSA | **Yes** — separate encoders |
| Produce EEG embedding for zero-shot transfer | No | **Yes** |
| Produce fNIRS embedding independently | No | **Yes** |
| Contrastive pre-training without labels | No | **Yes** |
| Missing modality robustness | Not designed for | **Core design goal** |

Our backbone trains two **independent encoders** (one per modality) connected only by a contrastive loss on the embeddings. This preserves modality-specific representations while learning cross-modal alignment, enabling the above capabilities that STA-Net fundamentally cannot support.
"""))

# ---- Cell 9: Design checklist ----
cells.append(nbformat.v4.new_markdown_cell("""## 8. STA-Net → Phase 2 Design Checklist

### Concrete architectural decisions for our backbone:

---

**Q1: Should we use spatial channel attention (topographic 2D projection)?**

**Decision: YES, as preprocessing — but with flexibility.**

STA-Net's cubic-interpolated 16×16 grid is effective for capturing spatial relationships. For Phase 2, we should implement the same 2D topographic projection for both EEG and fNIRS. However, our transformer-based encoder can also take raw channel sequences with coordinate positional encodings (x, y, z MNI positions), which may generalize better across datasets with different electrode layouts.

**Recommendation**: Default to 2D topographic projection for the BBCI dataset (matches STA-Net's preprocessing). Add a `use_topo_projection` config flag so raw-channel-with-coords can be tested in ablations.

---

**Q2: Should we use cross-attention between modalities or keep encoders fully separate?**

**Decision: SEPARATE encoders during pre-training; optional cross-attention fusion head for fine-tuning.**

STA-Net's cross-attention (EGTA) is powerful for supervised fusion, but it creates the entanglement problem. Our contrastive training must use **fully separate encoders** so each modality embedding is independently useful. Cross-attention should only appear in the downstream classification head, which is trained after the encoders are frozen (or fine-tuned with low LR).

**Recommendation**: EEG encoder: temporal conv + spatial attention (or ViT patch-based). fNIRS encoder: similar but with longer temporal receptive field. Contrastive loss directly between final embeddings. No cross-modal attention in the backbone.

---

**Q3: What temporal pooling strategy should the fNIRS encoder use?**

**Decision: Encode the hemodynamic lag explicitly — multiple lag windows OR fixed lag offset.**

STA-Net confirms that fNIRS lag matters (11 lag windows, learned by cross-attention). For our separate fNIRS encoder: the simplest approach is to **pre-align windows during data loading** using a fixed 5-second offset (matching hemodynamic response peak). The fNIRS encoder then sees a single window that is time-aligned to the EEG window.

If we want to be more flexible: provide the fNIRS encoder with a sequence of lag-shifted windows and use self-attention over them (no cross-modal dependency). The final fNIRS embedding is pooled across lags.

**Recommendation**: Start with fixed 5 s lag offset in preprocessing. Add multi-lag sequence input as a v2 experiment.

---

**Q4: What loss function for cross-modal alignment?**

**Decision: NT-Xent (contrastive) over STA-Net's Pearson auxiliary losses.**

STA-Net's `1 - pearson_r` loss is a batch-level correlation that doesn't distinguish between different trials. NT-Xent uses negative pairs from the same batch — this is strictly more informative and is the standard for contrastive representation learning.

**Recommendation**: NT-Xent as the primary pre-training objective. Optionally add a Pearson auxiliary loss (as STA-Net does) as a soft regularizer during fine-tuning.

---

**Q5: Additional learnings from STA-Net**

- **Dual-output training with auxiliary EEG head**: STA-Net trains with both `class_output` and `eeg_output` (line 306). The EEG head prevents the model from ignoring EEG entirely. In our framework, this maps to ensuring the EEG encoder produces meaningful embeddings independently — contrastive loss accomplishes this naturally, but we may add a unimodal classification auxiliary loss during fine-tuning.
- **Two-phase training**: Phase 1 (with validation) finds optimal stopping point; Phase 2 (full data) trains to matching loss. This pattern is useful for our fine-tuning stage — we should not over-commit to a fixed epoch count.
- **3D convolutions over (spatial_x, spatial_y, time)**: The Conv3D approach is natural for topographic data and works well. For our transformer-based backbone, patch-based tokenization (spatial patches × time) is the analogous approach.

---

### Final Summary Table

| Design Question | STA-Net Approach | Our Phase 2 Choice |
|----------------|-----------------|-------------------|
| Spatial representation | 16×16 topo grid | Same, + coord embeddings option |
| Temporal mismatch | 11 lag windows + cross-attn | Fixed 5s lag offset (default) |
| Cross-modal fusion | Baked into backbone | Separate encoders; fusion only in head |
| Alignment loss | Pearson auxiliary | NT-Xent contrastive |
| Missing modality | Not supported | Core requirement |
| Architecture | CNN (TF 2.10) | Transformer/CNN hybrid (PyTorch) |
"""))

nb.cells = cells

with open('notebooks/04_baseline_stanet.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook written successfully.")
