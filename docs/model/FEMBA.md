## FEMBA

FEMBA is a deep learning architecture for EEG signal analysis that leverages a **bidirectional Mamba block** for enhanced temporal modeling.  
It converts multi-channel EEG time series into compact temporal–spatial embeddings, supporting both **self-supervised pretraining** (mask reconstruction) and **supervised fine-tuning** for classification.

---

### Default Input Assumptions

Unless otherwise specified:

- **Channels**: 22  
- **Sampling Rate**: 256 Hz  
- **Segment Duration**: 5 seconds (1280 samples)  
- **Data Format**: Preprocessed and stored in HDF5 for efficient loading

These defaults are compatible with multiple EEG corpora (e.g., TUAB, TUAR, TUEG, TUSL), but FEMBA supports alternative montages, sampling rates, window lengths, and patch configurations.

---

### Preprocessing

Per-channel **interquartile range (IQR) normalization** is applied:

\[
x_{\text{norm}} = \frac{x - q_{\text{lower}}}{(q_{\text{upper}} - q_{\text{lower}}) + 1\times10^{-8}}
\]

where \(q_{\text{lower}}\) and \(q_{\text{upper}}\) are the 25th and 75th percentiles of the channel signal.

---

### Architecture Overview

1. **Tokenizer / Patch Embedding**  
   EEG \((C \times T)\) is segmented into 2D patches of size \(p \times q\) (default: **2 × 16**), projected via a 2D convolution into an embedding space, and augmented with **learnable positional embeddings**.  
   - Default input (22 ch × 1280 samples) with \(2 × 16\) patches → sequence length = 80 tokens.

2. **Encoder**  
   Built around a **bidirectional Mamba block**, enabling both forward and backward temporal processing. Forward and time-reversed streams are processed in parallel and combined by summation, with residual connections.

3. **Decoder**  
   A lightweight convolutional decoder with kernel sizes matched to the patch dimensions, used only during pretraining for masked patch reconstruction.

4. **Classifier Heads**  
   - **Linear head**: small fully connected stack (≈0.5M parameters)  
   - **Mamba-enhanced head**: adds one Mamba block before the linear head (≈0.7M parameters), improving temporal modeling for classification.

---

### Self-Supervised Learning (SSL) Objective

- Randomly mask **60%** of patch tokens (set to zero).
- Pass masked sequence through encoder.
- Reconstruct only masked patches.
- Loss: **Smooth L1** (default), with options for L1 or L2.

---

### Classification Protocols

FEMBA supports multiple downstream EEG classification paradigms, particularly for artifact detection:

- **BC – Binary Classification**  
  For each time window \(\Delta T\), if **any** channel contains an artifact, the window is labeled as *artifact* (1); if no channel contains an artifact, it is labeled as *background EEG* (0).

- **MC – Multi-Label Classification**  
  Each channel is classified independently in a binary fashion: artifact (1) if that channel contains **any** artifact type in \(\Delta T\), otherwise background EEG (0).

- **MMC – Multi-Class Multi-Output Classification**  
  Extends MC by distinguishing the **specific artifact type** per channel (e.g., eye movement, muscle, electrode pop, etc.). Each channel in \(\Delta T\) receives one of multiple artifact labels (or 0 for background EEG).

- **MCC – Multi-Class Classification**  
  Single-label classification per window from a subset of artifact categories, without channel-wise separation.

---

### Model Variants

| Variant       | Parameters | (num_blocks, embed_dim) |
|---------------|------------|-------------------------|
| FEMBA_tiny    | 7.8M       | (2, 35)                  |
| FEMBA_base    | 47.7M      | (12, 35)                 |
| FEMBA_large   | 77.8M      | (4, 79)                  |
| FEMBA_huge    | 386M       | (20, 79)                 |

---

### Training Setup

- **Pretraining**  
  - Dataset: TUEG (filtered to remove subjects present in TUAB, TUAR, TUSL)  
  - Optimizer: Adam, lr = \(1\times10^{-4}\), cosine decay  
  - Layer-wise learning rate decay factor: **0.75**  
  - Loss: Smooth L1 on masked patches  
  - Early stopping on validation loss

- **Fine-tuning**  
  - Decoder is removed  
  - Encoder + classifier trained end-to-end  
  - Optimizer: Adam, lr = \(1\times10^{-4}\), cosine decay  
  - Loss: CrossEntropyLoss  
  - Early stopping on validation loss  
  - Dataset splits:
    - TUAB: predefined train/test split
    - TUAR, TUSL: 80/10/10 train/val/test split

---

### Results Summary

**TUAB**  
- FEMBA-Huge: 81.82% balanced accuracy, 0.892 AUROC

**TUAR (BC protocol)**  
- FEMBA-Base: 0.949 AUROC, 0.932 AUPR

**TUSL**  
- FEMBA-Base: 0.731 AUROC

---
