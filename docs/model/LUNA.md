## LUNA

LUNA (Latent Unified Network Architecture) is a self-supervised foundation model for EEG signal analysis designed to be **topology-agnostic**. It addresses the challenge of heterogeneous electrode layouts across different EEG datasets by mapping multi-channel signals into a fixed-size latent space using a **learned query and cross-attention mechanism**. This approach decouples computational complexity from the channel count, enabling efficient and scalable processing.

---

### Default Input Assumptions

Unless otherwise specified:

-   **Channels**: Variable; pre-trained on 20, 22, and 29-channel recordings. Fine-tuned on tasks with 22 and 62 channels.
-   **Sampling Rate**: 256 Hz (after resampling).
-   **Segment Duration**: 5 seconds.
-   **Patch Size**: 40 timestamps.

---

### Preprocessing

A standardized preprocessing pipeline is applied to all data:

1.  **Filtering**: Signals are bandpass filtered between 0.1 Hz and 75 Hz  and a notch filter (50Hz or 60Hz) is applied.
2.  **Resampling**: All signals are resampled to 256 Hz.
3.  **Montage**: Signals are converted to a bipolar montage for TU datasets or processed in unipolar format for Siena and SEED-V datasets.
4.  **Normalization**: Per-channel z-score normalization is applied within each sample.

---

### Architecture Overview

LUNA uses an encoder-decoder architecture to transform heterogeneous EEG signals into a unified latent representation.

1.  **Patch Feature Extraction**
    Raw EEG signals ($B \times C \times T$) are segmented into non-overlapping temporal patches. These patches are embedded via two parallel pathways:
    -   **Temporal Embedding**: A 1D convolutional network encodes local temporal features.
    -   **Frequency Embedding**: An MLP projects the magnitude and phase from each patch's Fourier transform.
    The two embeddings are summed. Sinusoidal positional encodings are added to represent 3D electrode coordinates.

2.  **Channel-Unification Module**
    This core module uses cross-attention to map variable-channel features into a fixed-dimension latent space, achieving topology invariance. A set of `Q` learned queries cross-attends to the patch features from all `C` channels, projecting them onto a fixed-size representation. This step's complexity scales linearly with the number of channels.

3.  **Patch-wise Temporal Encoder**
    A stack of Transformer encoder blocks processes the unified latent representations as a temporal sequence. Since channel information is already unified, the effective sequence length is reduced from $S \times C$ to just $S$ (the number of patches), leading to significant computational savings. This module uses Rotary Positional Embeddings (ROPE) to capture temporal dependencies.

4.  **Decoder and Heads**
    LUNA supports two decoding strategies:
    -   **Reconstruction Head (Pre-training)**: Learned decoder queries attend to the encoder output to reconstruct the original masked patches.
    -   **Classification Head (Fine-tuning)**: A single aggregation query pools the encoder output into a single representation, which is passed to an MLP for classification.

---

### Self-Supervised Learning (SSL) Objectives

LUNA is pre-trained using two objectives:

-   **Masked Patch Reconstruction**
    -   A random subset of patch tokens is masked.
    -   Loss: **Smooth L1 loss** is applied to both masked and visible patches to encourage accurate reconstruction.

-   **Query Specialization Loss**
    -   An auxiliary loss that penalizes similarity between query-channel affinity matrices.
    -   This promotes diversity among the learned queries, improving model robustness.

---

### Downstream Classification Tasks

LUNA is evaluated on four diverse downstream tasks:

-   **Abnormality Detection (TUAB)**: Binary classification of EEG recordings as normal or abnormal.
-   **Artifact Detection (TUAR)**: Multi-class, single-label classification of 5 distinct artifact types.
-   **Slowing Classification (TUSL)**: 4-class classification of slowing events.
-   **Emotion Recognition (SEED-V)**: 5-class emotion classification on an unseen 62-channel montage.

---

### Model Variants

LUNA is available in three sizes, scaled by increasing the depth of the temporal encoder, embedding dimension, and number/size of queries.

| Variant | Parameters | (Layers, Queries, Q_size, Hidden_size) |
| :--- | :--- | :--- |
| LUNA-Base | 7M  | (8, 4, 64, 256)  |
| LUNA-Large | 43M  | (10, 6, 96, 576)  |
| LUNA-Huge | 311.4M  | (24, 8, 128, 1024)  |

---

### Training Setup

-   **Pre-training**
    -   Dataset: Combined TUEG and Siena corpora (>21,000 hours).
    -   Optimizer: AdamW, lr = $1.25 \times 10^{-4}$, cosine decay.
    -   Losses: Smooth L1 reconstruction loss and query specialization loss.
    -   Mask Ratio: 50%.

-   **Fine-tuning**
    -   The reconstruction decoder is replaced with a classification head.
    -   Loss: Cross-Entropy or Binary Cross-Entropy.
    -   Early stopping on validation loss with a patience of 10 epochs.
    -   Dataset splits:
        -   TUAB: Official train/test split.
        -   TUAR, TUSL: 80%/10%/10% randomized train/val/test split.
        -   SEED-V: Session trials split equally into train/val/test sets.

---

### Results Summary

**TUAB (Abnormal EEG Detection)**
-   LUNA-Huge: 81.57% balanced accuracy, 0.8957 AUROC.

**TUAR (Artifact Detection)**
-   LUNA-Huge: 0.921 AUROC (State-of-the-art).

**TUSL (Slowing Classification)**
-   LUNA-Huge: 0.802 AUROC (State-of-the-art).

**SEED-V (Emotion Recognition)**
-   LUNA-Large: 39.18% balanced accuracy.