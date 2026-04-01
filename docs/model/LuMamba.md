## LuMamba

LuMamba is a self-supervised EEG foundation model that combines a **topology-agnostic** design and efficient **Mamba state-space** temporal modeling. It addresses **1)** the challenge of varying channel layouts with **LUNA** channel unification, projecting a given EEG channel layout to a fixed latent topology, and **2)** overcomes the quadratic complexity of transformers with **FEMBA**'s efficient bi-Mamba encoder.

---

### Default Input Assumptions

Unless otherwise specified, the assumptions are identical to LUNA:

-   **Channels**: Variable; pre-trained on 20, 22-channel recordings. Fine-tuned on tasks with 16, 22, 26, 60, 62 and 128 channels.
-   **Sampling Rate**: Variable; default is 256 Hz (after resampling).
-   **Segment Duration**: Variable; durations which are a multiple of the patch size are also supported. Default is 5 seconds.
-   **Patch Size**: 40 timestamps.

---

### Preprocessing

A standardized preprocessing pipeline is applied to all data:

1.  **Filtering**: Signals are bandpass filtered between 0.1 Hz and 75 Hz  and a notch filter (50Hz or 60Hz) is applied.
2.  **Resampling**: All signals are resampled to 256 Hz. Sampling rates that are lower than 256 Hz are left unchanged.
3.  **Montage**: Signals are converted to a bipolar montage for TUH datasets or processed in unipolar format for non-TUH datasets.
4.  **Normalization**: Per-channel **interquartile range (IQR) normalization** is applied.


---

### Architecture Overview

LuMamba combines architectural elements from **LUNA** and **FEMBA**.

1. **LUNA Patch Feature Extraction**
    Raw EEG signals ($B \times C \times T$) are segmented into non-overlapping temporal patches. These patches are embedded via two parallel pathways:
    -   **Temporal Embedding**: A 1D convolutional network encodes local temporal features.
    -   **Frequency Embedding**: An MLP projects the magnitude and phase from each patch's Fourier transform.
    The two embeddings are summed. Sinusoidal positional encodings are added to represent 3D electrode coordinates.

2. **LUNA Channel-Unification Module**
    This core module uses cross-attention to map variable-channel features into a fixed-dimension latent space, achieving topology invariance. A set of `Q` learned queries cross-attends to the patch features from all `C` channels, projecting them onto a fixed-size representation. This step's complexity scales linearly with the number of channels.

3. **FEMBA Bi-Mamba Encoder**  
   Utilizes a **bidirectional Mamba block**, enabling both forward and backward temporal processing of embeddings ($B \times L \times D$) where $L$ is the number of tokens (sequence length), and $D = Q \times E$ is the embedding dimension. Forward and time-reversed streams are processed in parallel and combined by summation, with residual connections.

4. **Classifier Heads**  
   - **LUNA cross-attention classifier** (1.2M parameters): A single aggregation query pools the encoder output into a single representation, which is passed to an MLP for classification.
   - **Basic Linear classifier** (770 parameters): small fully connected stack.
   - **Mamba-enhanced classifier** (536K parameters): adds one Mamba block before the linear head (≈0.7M parameters), improving temporal modeling for classification.

---

### Self-Supervised Learning (SSL) Objective

LuMamba is pre-trained with a combination of two objectives:

**1. Masked reconstruction**
-   A random subset of patch tokens is masked.
-   Loss: **Smooth L1 loss** is applied to both masked and visible patches to encourage accurate reconstruction.

**2. LeJEPA (Latent Euclidean Joint-Embedding Predictive Architecture)**
-  Global and local views are extracted from the raw signal across all channels. Global views capture a wider temporal context than local views.
-  Loss: **LeJEPA Prediction loss** (difference between local and global views in embedding space) + **Isotropic Gaussian regularization** (Epps-Pulley deviation to Gaussian along 1D slices of local embeddings)

---

### Classification Protocols

LuMamba supports multiple downstream EEG task designs:

- **BC – Binary Classification**  
  For each time window, if **any** channel contains an artifact, the window is labeled as *artifact* (1); if no channel contains an artifact, it is labeled as *background EEG* (0).

- **MCC – Multi-Class Classification**  
  Single-label classification per window from a subset of artifact categories, without channel-wise separation.

- **Multi-target regression**
  Multi-output continuous values predicted for each temporal window of the signal.

---

### Model Variants

The model currently exists in a Tiny Variant, with the following parameters: 

| Variant         | Parameters | FEMBA parameters            |LUNA parameters                     |
|-----------------|------------|-----------------------------|------------------------------------|
| LuMamba_tiny    | 4.1M       |(`num_blocks` = 2, `exp` = 2)|(`num_queries` = 6, `embed_dim` = 64)

Larger model sizes can be attained by increasing the number of bi-Mamba blocks `num_blocks` (e.g. 8 bi-Mamba blocks yields 15M parameters).

---

### Training Setup

- **Pre-training**
  - **Dataset:** TUEG corpus (>21,000 hours).
  - **Optimizer:** AdamW, lr = $1.25 \times 10^{-4}$, cosine decay.
  - **Losses:** Smooth L1 reconstruction loss and query specialization loss (part of original LUNA pre-training). Other variant include: **mixed LeJEPA-reconstruction** pre-training, and **LeJEPA-only** pre-training.
    - Mask Ratio: 60%.

- **Fine-tuning**  
  - The reconstruction decoder is replaced with a classification head.
  - The encoder + classifier are trained end-to-end (unfreezed encoder).
  - **Optimizer:** Adam, lr = \(5\times10^{-4}\), cosine decay  
  - **Loss:** Cross-Entropy or Binary Cross-Entropy for multi-class and binary classification respectively. MSE Loss for regression task.
    - Early stopping on validation loss with a patience of 10 epochs.
  - **Dataset splits:**
    - **TUAB:** official predefined train/val/test split
    - **TUAR, TUSL:** 80/10/10 train/val/test split
    - **SEED-V:** 5/5/5 trials split for each session into train/validation/test. 
    - **APAVA:** 15/4/4 subjects for train/validation/test
    - **TDBrain:** 34/8/8 subjects for train/validation/test
    - **MODMA:** 15 depressed patients and 15 controls for training, and 7 normal and 5 depressed for validation and testing.
    - **Mumtaz2016:** 24 depressed and 19 controls for training, 5 depressed and 4 controls for validation, 5 depressed and 5 controls for testing.
    - **MoBI:** 10min/5min/5min split of each walking session for train/validation /test. A stride of 50ms is applied to generate samples. Target values for a sample are averaged over the final 50ms of the sample.

- **Visualization logs**  
  - Logs labeled **t-SNE** of subsets of **TUAB** and **TUAR** at each validation step.
    - Embeddings are extracted from the FEMBA bi-Mamba block and mean-pooled across the sequence dimension.
    - Plots are saved as .npy files in a logging directory for further customization.

---

### Results Summary

**TUAB (Abnormal EEG Detection)**
- LuMamba-Tiny (mixed LeJEPA-reconstruction): 80.99 balanced accuracy, 0.883 AUROC, 0.892 AUPR.

**TUAR (Artifact Detection)**
- LuMamba-Tiny (reconstruction-only): 0.914 AUROC.

**TUSL (Slowing Classification)**
- LuMamba-Tiny (reconstruction-only): 0.708 AUROC.

**APAVA (Alzheimer's detection)**
- LuMamba-Tiny (mixed LeJEPA-reconstruction): 0.955 AUROC, 0.970 AUPR (state-of-art).

**TDBrain (Parkinson's detection)**
- LuMamba-Tiny (mixed LeJEPA-reconstruction): 0.961 AUROC, 0.960 AUPR.

**SEED-V (Emotion Recognition)**
- LuMamba-Tiny (reconstruction-only): 35% balanced accuracy.

**Mumtaz2016 (Depression detection)**
- LuMamba-Tiny (mixed LeJEPA-reconstruction): 72% balanced accuracy.

**MoBI (Gait Prediction)**
- LuMamba-Tiny (reconstruction-only): 0.11 $R^2$, 0.38 Pearson's correlation.

---
