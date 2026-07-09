## PanLUNA

PanLUNA is a self-supervised **pan-modal** biosignal foundation model that jointly processes **EEG, ECG, and PPG** within a single shared encoder. Extending LUNA's channel-unification module, PanLUNA treats multimodal channels as entries in a **unified query set augmented with sensor-type embeddings**, enabling efficient cross-modal early fusion while remaining inherently **robust to missing modalities** at inference time. Despite its compact 5.4M-parameter footprint, PanLUNA matches or exceeds models up to 57× larger, and supports quantization-aware INT8 deployment on the GAP9 ultra-low-power RISC-V microcontroller for continuous wearable monitoring.

---

### Default Input Assumptions

All modalities are resampled to 256 Hz and segmented into non-overlapping 5-second windows with a patch size of 32 samples, unless a downstream task specifies otherwise (e.g., 10-second windows for ECG benchmarks, 30-second epochs for sleep staging).

| Modality | Channels | Native Sampling Rate |
|----------|----------|----------------------|
| EEG | 20–22 (pre-training); 29 (Siena) | 250–512 Hz |
| ECG | 12 (pre-training and cardiac benchmarks); 1 (HMC sleep staging) | 400–500 Hz |
| PPG | 1 (PulseDB) | 125 Hz |

Missing modalities are handled natively at inference without any architectural modification.

---

### Preprocessing

A standardized modality-specific preprocessing pipeline is applied to all data:

1. **Filtering**: Bandpass filtering with a 4th-order Butterworth filter, with modality-specific cutoffs: EEG 0.1–75 Hz; ECG 0.5–120 Hz; PPG 0.5–8 Hz. A notch filter (50 Hz or 60 Hz) is additionally applied.
2. **Resampling**: All signals resampled to 256 Hz. 
3. **Normalization**: Per-channel z-score normalization, to account for the large amplitude differences across modalities (e.g., EEG in µV vs. ECG in mV).
4. **Segmentation**: Non-overlapping 5-second windows during pre-training; task-specific windowing during fine-tuning (e.g., 10-second windows for PTB-XL/CSN, 30-second epochs for HMC sleep staging).

---

### Architecture Overview

PanLUNA extends LUNA to the multimodal setting by generalizing topology invariance to cross-modal fusion.

1. **Input Representation**
   Channels from all modalities are concatenated along the channel dimension before entering the model. **Sensor-type embeddings** are introduced via a modality-specific lookup table, added to channel features at the input stage to distinguish sensing modalities. Channel positional encodings are modality-specific:
   - **EEG**: Normalized 3D electrode coordinates encoded with sinusoidal embeddings (as in LUNA).
   - **ECG**: Lead-angle estimates derived from [anatomical measurements on 30 body scans](https://www.ijcai.org/proceedings/2021/0495.pdf), constructing a spatial encoding analogous to EEG electrode positioning.
   - **PPG**: Neutral coordinate (0, 0) assigned; the model relies on the sensor-type embedding for modality identification.

2. **Patch Feature Extraction**
   Signals are partitioned into short temporal patches and embedded via lightweight convolutional encoders combined with frequency features from the real-valued FFT. Patch-level features are augmented with positional encodings and sensor-type embeddings before entering the unification module.

3. **Channel–Modality Unification Module**
   Cross-attention aggregates information across both channels and modalities through a shared set of latent queries. This design removes the requirement for paired multimodal recordings during pre-training and enables training on large-scale unimodal corpora. 

4. **Temporal Transformer Encoder**
   The unified latent sequence is processed by a patch-wise temporal Transformer with **Rotary Positional Embeddings (RoPE)** to capture long-range temporal dependencies. Self-attention operates on the fixed-size latent representation, fully decoupled from electrode count and modality composition.

5. **Decoding and Classifier Heads**
   During pre-training, a reconstruction decoder attends to encoder outputs to recover masked signal patches in a channel-specific manner. During fine-tuning this decoder is discarded and replaced by a lightweight aggregation query that pools the encoder output into a single representation, fed to a classification head. Three adaptation strategies are supported:
   - **Full Fine-tuning (FF)**: All 5.4M parameters updated.
   - **Frozen Encoder (FE)**: Backbone fixed; only the classification head (~400k parameters) trained.
   - **LoRA**: Low-rank matrices (rank 16, ~180k parameters, ~580k total) injected into selected Transformer layers.

---

### Self-Supervised Learning (SSL) Objective

PanLUNA is pre-trained with a **masked signal reconstruction** objective. A random subset of patch tokens is masked, and the reconstruction decoder is trained to recover the original signal patches in a channel-specific manner.

---

### Classification Protocols

- **BC – Binary Classification**: Window-level binary label (e.g., normal vs. abnormal EEG on TUAB).
- **MCC – Multi-class Classification**: Single-label classification per window (e.g., 5-stage sleep scoring on HMC).
- **Multi-label Classification**: Multiple co-occurring labels per window (e.g., 19-label PTB-XL-Form ECG morphology).

---

### Model Variants

| Variant | Parameters |
|---------|------------|
| PanLUNA | 5.4M |

---

### Training Setup

- **Pre-training**
  - **Datasets**: ~40,000 hours of heterogeneous biosignal data across five corpora:

    | Dataset | Modality | Subjects | Channels | FS (Hz) | Window |
    |---------|----------|----------|----------|---------|--------|
    | TUEG | EEG | 14,987 | 20/22 | 250 | 5 s |
    | Siena | EEG | 14 | 29 | 512 | 5 s |
    | MIMIC-IV | ECG | 161,352 | 12 | 500 | 5 s |
    | CODE-15% | ECG | 233,700 | 12 | 400 | 5 s |
    | PulseDB | ECG, PPG | 5,361 | 2 | 125 | 5 s |

  - **Objective**: Masked signal reconstruction; each modality can be used independently (no paired multimodal data required).

- **Fine-tuning**
  - Reconstruction decoder replaced with aggregation query + classification head; three adaptation strategies available (FF, FE, LoRA).
  - **Loss**: Cross-Entropy for multi-class; BCE for multi-label classification.
  - **Dataset splits**:
    - **TUAB**: Official predefined train/val/test split.
    - **PTB-XL (Super/Sub/Form/Rhythm)** and **CSN**: MERL ICML 2024 protocol.
    - **HMC (sleep staging)**: Splits as in PhysioOmni.

- **Quantization**
  - Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) via Brevitas; evaluated at INT8, INT4, and INT2 weights. QAT runs for 15 fine-tuning epochs and recovers ≥96% of FP32 performance at INT8; INT2 weights achieve up to 16× storage reduction with graceful degradation.

---

### Edge Deployment (GAP9)

PanLUNA is deployed on the **GAP9 ultra-low-power RISC-V microcontroller** (9-core cluster at 370 MHz, 1.5 MB L2 SRAM) using the BioFoundation edge framework with automated operator tiling, double-buffered DMA, NE16 acceleration, and custom tiled kernels for cross-attention projections and sensor-type embedding lookup.

| Configuration | Channels | Window | MACs | Latency | Energy | Power |
|---------------|----------|--------|------|---------|--------|-------|
| ECG only | 12 | 10 s | 120.5 M | 325.6 ms | 18.8 mJ | 60.2 mW |
| EEG + ECG | 5 | 30 s | 446.2 M | 1.206 s | 68.65 mJ | 56.9 mW |

Streaming latency for ECG (patch-triggered): **450.6 ms** (125 ms acquisition + 325.6 ms compute). Estimated continuous monitoring battery life on a 300 mAh / 3.7 V wearable: **~24 days** (ECG-only), **~20 days** (multimodal sleep staging). This is, to our knowledge, the first deployment of a multimodal physiological FM on an ultra-low-power MCU.

---

### Results Summary

**TUAB (Abnormal EEG Detection)**
- PanLUNA (FF): **81.21%** balanced accuracy, 0.8999 AUC-PR, 0.8932 AUROC — outperforming LUNA-Base and LUNA-Large despite being 8–57× smaller.

**HMC (Multimodal Sleep Staging, 5-class)**

| Variant | Test Modality | Bal. Acc. | Cohen's κ | Weighted F1 |
|---------|--------------|-----------|-----------|-------------|
| PanLUNA (FF) | EEG | **0.7416** | **0.6946** | **0.7659** |
| PanLUNA (FF) | EEG + ECG | 0.7002 | 0.6561 | 0.7383 |
| PanLUNA (FF) | ECG only | 0.2977 | 0.1095 | 0.2876 |
| PanLUNA (QAT INT8) | EEG + ECG | 0.7347 | 0.6913 | 0.7273 |

State-of-the-art on HMC; surpasses PhysioOmni by +1.27% balanced accuracy.

**PTB-XL / CSN (Cardiac Benchmarks, LoRA FE, FP32)**

| Task | AUROC |
|------|-------|
| PTB-XL Super | 0.9083 |
| PTB-XL Sub | 0.8880 |
| PTB-XL Form | 0.8331 |
| PTB-XL Rhythm | 0.9641 |
| CSN | 0.9505 |

State-of-the-art on PTB-XL Super and CSN. QAT INT8 recovers ≥96% of FP32 AUROC across all tasks; INT2 weights achieve up to 16× storage reduction with graceful degradation.

### Additional Cross-Modal Results
- We provide additional evaluation on PPG-only and cross-modal ECG+PPG combination leveraging WESAD stress detection dataset in 2-class and 4-class classification setup according to definition in the [PulsePPG](https://arxiv.org/pdf/2502.01108).

| Model | Size | WESAD (2) AUROC ↑ | WESAD (2) AUC-PR ↑ | WESAD (4) AUROC ↑ | WESAD (4) AUC-PR ↑ |
|---|---:|---:|---:|---:|---:|
| *Supervised Models* ||||| |
| ResNet-26 | – | 0.3974 | 0.2590 | 0.4662 | 0.2641 |
| Random Forest | – | 0.5374 | 0.3173 | 0.7060 | 0.4811 |
| *General Time-series FMs* ||||| |
| Chronos | 200M | 0.8878 | 0.7530 | **0.8751** | **0.7443** |
| MOMENT | 385M | <u>0.9583</u> | <u>0.9196</u> | 0.7491 | 0.5304 |
| *PPG FMs* ||||| |
| Pulse-PPG | 28.5M | **0.9687** | **0.9410** | 0.7807 | 0.5671 |
| Light Pulse-PPG | 5.7M | 0.9504 | 0.8764 | 0.6655 | 0.4667 |
| PaPaGei | 5.7M | 0.8043 | 0.6465 | 0.7684 | 0.5205 |
| **PanLUNA (FE) - PPG** | **5.4M** | 0.8747 ± 0.0046 | 0.7914 ± 0.0073 | <u>0.7882 ± 0.0102</u> | <u>0.6314 ± 0.0233</u> |
| **PanLUNA (FE) - PPG+ECG** | **5.4M** | 0.8957 ± 0.0302 | 0.8616 ± 0.0394 | 0.7820 ± 0.0329 | 0.5682 ± 0.0372 | 

#### Key observations

- In the binary PPG-only setting, PanLUNA trails Pulse-PPG, but outperforms PaPaGei, a PPG-specific foundation model of comparable size.

- The gap to Pulse-PPG may reflect differences in pretraining data: Pulse-PPG is pretrained on wearable-field PPG, while PanLUNA and PaPaGei use clinical PPG signals.

- Adding ECG to PPG improves PanLUNA in the binary setup without increasing model size, raising AUROC from 0.8747 to 0.8957 and AUC-PR from 0.7914 to 0.8616.

- In the 4-class setting, PanLUNA achieves the strongest results among PPG-based foundation models under 10M parameters, with especially large gains over Light Pulse-PPG in AUC-PR.

- The PPG+ECG setting does not consistently improve over PPG-only in the 4-class case, suggesting dataset- and modality-specific effects.
