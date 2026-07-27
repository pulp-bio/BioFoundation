## TinyMyo

**TinyMyo** is a lightweight **3.6M-parameter** Transformer-based foundation model (FM) for **surface EMG (sEMG)**. It is designed for **broad generalization** across datasets, sensor configurations, domains, and tasks, while remaining efficient enough for **ultra-low-power edge deployment** on microcontrollers.

TinyMyo is designed for deployment on ultra-low-power microcontrollers such as GAP9.

---

### Default Input Assumptions

* **Channels**: 16
* **Sampling Rate**: 2000 Hz
* **Segment Length**: 1000 samples (0.5 s)
* **Windowing**: 50% overlap during pretraining

### Preprocessing

The standard preprocessing pipeline is:

* 4th-order **20–450 Hz bandpass**
* **Notch filter** at 50 Hz
* Per-channel min–max normalization (pretraining)
* Per-channel z-score normalization (downstream)

Datasets with fewer than 16 channels are *zero-padded* only during pretraining.
The model supports at most 16 runtime channels. When fewer channels are supplied,
the corresponding prefix of the 16 learned channel-slot embeddings is used.
The default model expects exactly 1000 temporal samples; inputs with a different
length must use a model configuration with a matching `img_size`.

---

### Architecture Overview

TinyMyo is pretrained using **masked reconstruction** across three heterogeneous large-scale EMG datasets:

| Dataset     | Subjects | fs      | Channels | Size    |
| ----------- | -------- | ------- | -------- | ------- |
| Ninapro DB6 | 10       | 2000 Hz | 14       | 20.3 GB |
| Ninapro DB7 | 22       | 2000 Hz | 12       | 30.9 GB |
| EMG2Pose    | 192      | 2000 Hz | 16       | 431 GB  |

#### Tokenization: Channel-Independent Patches

Unlike 2D (channel-mixing) tokenizers in EEG FMs, TinyMyo uses **strictly per-channel patching**:

* Patch length: **20 samples**
* Patch stride: **20 samples**
* Tokens per channel: **50**
* Sequence length: **800 tokens** (16 x 50)
* Positional encoding: **RoPE** (Rotary Position Embeddings)

Tokens are ordered channel-major: all temporal patches from channel 0 are followed
by all patches from channel 1, and so on. RoPE positions reset for each channel,
so the first patch of two different channels has the same temporal position. This
prevents the flattened token order from being interpreted as a physical temporal
gap between channels.

This preserves electrode-specific information while letting attention learn cross-channel relationships.

The learned `channel_embed` is a channel-slot embedding. It identifies the
channel index used by the input tensor; it is not an electrode-coordinate or
sensor-placement embedding. Consequently, heterogeneous sensor layouts remain
structurally supported, but channel ordering should be documented by each data
pipeline and is not automatically aligned by physical electrode location.

During pretraining, zero-padded channels are identified by `pad_mask_ch`. Any
masking assigned to those channels is cleared, and padded tokens are excluded
from the set of attention keys. The reconstruction target still follows the
pretraining task's configured masked and unmasked loss weighting.

#### Transformer Encoder

* **8 layers**
* **3 heads**
* Embedding dim: **192**
* Pre-LayerNorm
* Dropout & drop-path: **0.1**

#### Lightweight Decoder

A simple **linear layer** (≈ **3.9k params**) reconstructs masked patches.
Following SimMIM philosophy, the minimal decoder forces the encoder to learn structured latent representations.

### Self-Supervised Learning Objective

* **50% random masking** with a learnable [MASK] token
* Reconstruction loss = **Smooth L1**

$$
  \mathcal{L} = \mathcal{L}*{\text{masked}} + 0.1 \cdot \mathcal{L}*{\text{visible}}
$$

### Training Setup

The repository configurations define the following defaults:

* **Pretraining**: AdamW, learning rate `5e-4`, weight decay `1e-2`, 30 epochs,
  3 warm-up epochs, batch size 512, and `bf16-mixed` precision.
* **Fine-tuning**: AdamW, learning rate `5e-4`, weight decay `1e-2`, 50 epochs,
  and 5 warm-up epochs.
* Both schedules use cosine decay.

---

### Model Variants and Pipeline

#### Model Variants

| Variant     | Params   | (Layers, dim) |
| ----------- | -------- | ------------- |
| **TinyMyo** | **3.6M** | (8, 192)      |
| **TinyssimoMyo** | **1.9M** | (4, 192)      |

#### Pipeline

**Pretraining**

```
EMG -> Channel-indep. patching -> Masking -> Transformer Encoder -> Linear Decoder -> Patch reconstruction
```

**Downstream**

```
EMG -> Patching -> Transformer Encoder -> Channel fusion -> Temporal pooling -> Task-specific head
```

#### Implementation Contract

For the default configuration:

| Quantity | Value |
| --- | ---: |
| Input shape | `(B, 16, 1000)` |
| Temporal patch size | `20` samples |
| Patches per channel | `50` |
| Encoder tokens | `800` |
| Token dimension | `192` |
| Attention heads | `3` |
| Per-head dimension | `64` |

`embed_dim` must be divisible by `n_head`, and the per-head dimension must be
even because RoPE rotates pairs of features. `img_size` must be divisible by
`patch_size`, and runtime input length must match `img_size`.

The task heads return `(B, num_classes)` for classification and
`(B, img_size, num_classes)` for regression. In pretraining, the linear decoder
returns one reconstructed patch of length `patch_size` per encoder token.

---

### Downstream Tasks

TinyMyo supports three major categories:

---

#### Hand Gesture Classification

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

---

#### Hand Kinematic Regression

Dataset: **Ninapro DB8** (2000 Hz)
Task: Regress **5 joint angles (DoA)**
Preprocessing: z-score only; windows of **100 ms** or **500 ms**

**Regression head (788k params)**

* Depthwise + pointwise convolutions
* Upsampling
* Global average pooling
* Linear projection to 5 outputs

---

#### Speech Production and Speech Recognition

Dataset: **Gaddy Silent Speech**
(8 channels, 1000 Hz, face/neck EMG)
Repository: [MatteoFasulo/silent_speech](https://github.com/MatteoFasulo/silent_speech)
>Note: Additional details on Silent Speech dataset and instructions on how to run experiments can be found in the linked repository.

##### Speech Production (EMG -> MFCC -> HiFi-GAN -> Audio)

Pipeline:

1. Residual downsampling blocks
2. TinyMyo encoder
3. Linear projection to **26-dim MFCC**
4. HiFi-GAN vocoder (pretrained)

##### Speech Recognition (EMG -> Text)

* Same encoder + residual front-end
* Linear projection to 37 characters
* **CTC loss**
* 4-gram LM + beam search

---

### Edge Deployment

TinyMyo is deployed on **GAP9 (RISC-V, ultra-low power)**.

Key elements:

* **INT8 quantization**, including attention
* Hierarchical streaming:

  * L3 -> L2 (slab streaming)
  * L2 -> L1 (tile streaming)
* Integer softmax, integer LayerNorm, integer GELU
* Static liveness-based memory arena

For current deployment measurements and benchmark results, refer to the
[PulpBio/TinyMyo model card](https://huggingface.co/PulpBio/TinyMyo) and the
[TinyMyo paper](https://arxiv.org/abs/2512.15729).

---

### Pretrained Weights

The [PulpBio/TinyMyo Hugging Face repository](https://huggingface.co/PulpBio/TinyMyo)
provides the pretrained model card and downloadable checkpoints. The repository
configuration is the source of truth for model construction and fine-tuning entry points.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PulpBio/TinyMyo",
    local_dir="checkpoints/TinyMyo",
)
```

Run fine-tuning from the repository root:

```bash
python -u run_train.py +experiment=TinyMyo_finetune \
  pretrained_safetensors_path=/absolute/path/to/checkpoints/TinyMyo/UCI_EMG/base.safetensors
```

Related experiments remain in dedicated repositories:

- [Silent Speech](https://github.com/MatteoFasulo/silent_speech)
- [Generic Neuromotor Interface](https://github.com/MatteoFasulo/generic-neuromotor-interface)
