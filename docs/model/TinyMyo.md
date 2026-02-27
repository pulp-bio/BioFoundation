# **TinyMyo: A Tiny Foundation Model for EMG Signals**

**TinyMyo** is a lightweight **3.6M-parameter** Transformer-based foundation model (FM) for **surface EMG (sEMG)**. It is designed for **broad generalization** across datasets, sensor configurations, domains, and tasks, while remaining efficient enough for **ultra-low-power edge deployment** on microcontrollers.

TinyMyo is the **first EMG foundation model** demonstrated on a microcontroller (GAP9), achieving an inference time of **0.785 s**, energy of **44.91 mJ**and power envelope of **57.18 mW**.

---

## **1. Default Input Assumptions**

Unless otherwise specified, TinyMyo uses:

* **Channels**: 16
* **Sampling Rate**: 2000 Hz
* **Segment Length**: 1000 samples (0.5 s)
* **Windowing**: 50% overlap during pretraining
* **Preprocessing**:

  * 4th-order **20–450 Hz bandpass**
  * **Notch filter** at 50 Hz
  * Per-channel min–max normalization (pretraining)
  * Per-channel z-score normalization (downstream)

Datasets with fewer than 16 channels are *zero-padded* only during pretraining.

---

## **2. Pretraining Overview**

TinyMyo is pretrained using **masked reconstruction** across three heterogeneous large-scale EMG datasets:

| Dataset     | Subjects | fs      | Channels | Size    |
| ----------- | -------- | ------- | -------- | ------- |
| Ninapro DB6 | 10       | 2000 Hz | 14       | 20.3 GB |
| Ninapro DB7 | 22       | 2000 Hz | 12       | 30.9 GB |
| EMG2Pose    | 192      | 2000 Hz | 16       | 431 GB  |

### **Tokenization: Channel-Independent Patches**

Unlike 2D (channel-mixing) tokenizers in EEG FMs, TinyMyo uses **strictly per-channel patching**:

* Patch length: **20 samples**
* Patch stride: **20 samples**
* Tokens per channel: **50**
* Sequence length: **800 tokens** (16 x 50)
* Positional encoding: **RoPE** (Rotary Position Embeddings)

This preserves electrode-specific information while letting attention learn cross-channel relationships.

### **Transformer Encoder**

* **8 layers**
* **3 heads**
* Embedding dim: **192**
* Pre-LayerNorm
* Dropout & drop-path: **0.1**

### **Lightweight Decoder**

A simple **linear layer** (≈ **3.9k params**) reconstructs masked patches.
Following SimMIM philosophy, the minimal decoder forces the encoder to learn structured latent representations.

### **Masking Objective**

* **50% random masking** with a learnable [MASK] token
* Reconstruction loss = **Smooth L1**

$$
  \mathcal{L} = \mathcal{L}*{\text{masked}} + 0.1 \cdot \mathcal{L}*{\text{visible}}
$$

### **Training Setup**

* Optimizer: **AdamW** (β=(0.9, 0.98), wd=0.01)
* LR: **1x10⁻⁴**, cosine decay
* Batch size: **512** with gradient accumulation
* Epochs: **50** with 10-epoch warm-up
* Hardware: **4x NVIDIA GH200 GPUs**

---

## **3. Architecture Summary**

### **Model Variant**

| Variant     | Params   | (Layers, dim) |
| ----------- | -------- | ------------- |
| **TinyMyo** | **3.6M** | (8, 192)      |

### **Pipeline**

**Pretraining**

```
EMG -> Channel-indep. patching -> Masking -> Transformer Encoder -> Linear Decoder -> Patch reconstruction
```

**Downstream**

```
EMG -> Patching -> Transformer Encoder -> Channel fusion -> Temporal pooling -> Task-specific head
```

---

## **4. Downstream Tasks**

TinyMyo supports three major categories:

---

### **4.1 Hand Gesture Classification**

Evaluated on:

* **Ninapro DB5** (52 classes, 10 subjects, 200 Hz)
* **EPN-612** (5 classes, 612 subjects, 200 Hz)
* **UCI EMG** (6 classes, 36 subjects, 200 Hz)
* **Generic Neuromotor Interface** (Meta wristband; 9 gestures)
  * Repository: [MatteoFasulo/generic-neuromotor-interface](https://github.com/MatteoFasulo/generic-neuromotor-interface)

>Note: Additional details on generic non-invasive neuromotor interface dataset and instructions on how to run experiments can be found in the linked repository inside the `notebooks` folder.

**Pipeline**

* EMG filtering: **20–90 Hz** bandpass + 50 Hz notch
* Windows:

  * **1 sec** (best for DB5)
  * **5 sec** (best for EPN & UCI)
* Per-channel z-scoring
* Linear classification head

  * Input: **C x 192**
  * Params: typically **<40k**

**Performance (Fine-tuned)**

| Dataset                  | Metric   | Result            |
| ------------------------ | -------- | ----------------- |
| **Ninapro DB5 (1 sec)** | Accuracy | **89.41 ± 0.16%** |
| **EPN-612 (5 sec)**    | Accuracy | **96.74 ± 0.09%** |
| **UCI EMG (5 sec)**    | Accuracy | **97.56 ± 0.32%** |
| **Neuromotor Interface** | CLER     | **0.153 ± 0.006** |

TinyMyo achieves **state-of-the-art** on DB5, EPN-612, and UCI.

---

### **4.2 Hand Kinematic Regression**

Dataset: **Ninapro DB8**
Task: Regress **5 joint angles (DoA)**
Preprocessing: z-score only; windows of **100 ms** or **500 ms**

**Regression head (788k params)**

* Depthwise + pointwise convolutions
* Upsampling
* Global average pooling
* Linear projection to 5 outputs

**Performance (Fine-tuned)**

* **MAE = 8.77 ± 0.12°** (1000 ms window)

Although previous works achieve lower MAE (≈6.89°), those models are **subject-specific**, whereas TinyMyo trains **one model across all subjects**, a significantly harder problem.

---

### **4.3 Speech Production & Speech Recognition**

Dataset: **Gaddy Silent Speech**
(8 channels, 1000 Hz, face/neck EMG)
Repository: [MatteoFasulo/silent_speech](https://github.com/MatteoFasulo/silent_speech)
>Note: Additional details on Silent Speech dataset and instructions on how to run experiments can be found in the linked repository.

#### **Speech Production (EMG -> MFCC -> HiFi-GAN -> Audio)**

Pipeline:

1. Residual downsampling blocks
2. TinyMyo encoder
3. Linear projection to **26-dim MFCC**
4. HiFi-GAN vocoder (pretrained)

**WER (Fine-tuned):**

* **33.54 ± 1.12%**

Comparable to SoA (≈32%) with **>90% fewer parameters** in the transduction model.

#### **Speech Recognition (EMG -> Text)**

* Same encoder + residual front-end
* Linear projection to 37 characters
* **CTC loss**
* 4-gram LM + beam search

**WER:**

* **33.95 ± 0.97%**

Although not surpassing the multimodal MONA-LISA (12.2%), TinyMyo is vastly smaller and EMG-only.

---

## **5. Edge Deployment**

TinyMyo is deployed on **GAP9 (RISC-V, ultra-low power)**.

Key elements:

* **INT8 quantization**, including attention
* Hierarchical streaming:

  * L3 -> L2 (slab streaming)
  * L2 -> L1 (tile streaming)
* Integer softmax, integer LayerNorm, integer GELU
* Static liveness-based memory arena

**Runtime (EPN612 dataset):**

* **0.785 s inference time**
* **44.91 mJ energy**
* **57.18 mW average power**

This is the **first demonstration of an EMG FM on a microcontroller**.

---

## **6. Results Summary**

### **Pretraining**

* Smooth L1 reconstruction with high fidelity
* Total FLOPs: ~4.0G

### **Downstream SoA Highlights**

* **DB5:** 89.41%
* **EPN-612:** 96.74%
* **UCI EMG:** 97.56%
* **Neuromotor:** 0.153 CLER
* **DB8 Regression:** MAE 8.77°
* **Speech Production:** WER 33.54%
* **Speech Recognition:** WER 33.95%

Overall TinyMyo matches or exceeds state-of-the-art while being on par with or smaller than prior EMG foundation models.
