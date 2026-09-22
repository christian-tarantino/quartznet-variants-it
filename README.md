```text
                                   888                               888    
                                   888                               888    
                                   888                               888    
 .d88888 888  888  8888b.  888d888 888888 88888888 88888b.   .d88b.  888888 
d88" 888 888  888     "88b 888P"   888       d88P  888 "88b d8P  Y8b 888    
888  888 888  888 .d888888 888     888      d88P   888  888 88888888 888    
Y88b 888 Y88b 888 888  888 888     Y88b.   d88P    888  888 Y8b.     Y88b.  
 "Y88888  "Y88888 "Y888888 888      "Y888 88888888 888  888  "Y8888   "Y888 
     888                                                                    
     888                                                                    
     888                                                                                                                                      

```
## 📊 Main Results on Common Voice (Italian v26.0)

Evaluated on the test set using **Greedy Decoding** and **KenLM** integration.

### Greedy Decoding

| Method | Params | Test Loss | CER (%) | WER (%) |
| :--- | :---: | :---: | :---: | :---: |
| QuartzNet 5x5 | 6.7M | 0.4284 | 38.03 | 10.15 |
| QuartzNet_attn 5x5 | 9.3M | 0.4422 | 33.35 | 8.90 |
| QuartzNet 10x5 | 12.8M | 0.3647 | 33.65 | 9.03 |
| **QuartzNet_attn 10x5** | **12.7M** | **0.3508** | **28.01** | **7.43** |

### Language Model Decoding (KenLM)

| Method | Params | Test Loss | CER (%) | WER (%) |
| :--- | :---: | :---: | :---: | :---: |
| QuartzNet 5x5 | 6.7M | 0.4284 | 19.38 | 6.80 |
| QuartzNet_attn 5x5 | 9.3M | 0.4422 | 19.16 | 6.33 |
| QuartzNet 10x5 | 12.8M | 0.3647 | 17.04 | 6.07 |
| **QuartzNet_attn 10x5** | **12.7M** | **0.3508** | **15.80** | **5.29** |

---

## 📝 Methodological & Architectural Notes

### 1. Architectural Decisions & Rationale
To improve context aggregation without significantly increasing parameter overhead, the standard 1x1 Conv classification head was replaced with a **Self-Attention head**. This modification consistently yielded superior performance, notably lowering the Word Error Rate (WER) by enabling the model to capture long-range acoustic dependencies before projection.

### 2. Data Pipeline & Augmentation Strategy
To prevent acoustic memorization and enforce invariance across specific frequency channels and temporal spans, **SpecAugment** was integrated directly into the training pipeline:
* **Time Masking**: Randomly masks consecutive time steps (frames) to force the model to rely on surrounding temporal context for phoneme sequence prediction.
* **Frequency Masking**: Randomly masks consecutive mel-frequency channels to prevent over-reliance on narrow spectral bands or pitch-specific features.

---

### 🔍 Key Observations: Depth vs. Attention Mechanism

A comparison of the evaluation metrics reveals a clear trade-off between local receptive fields and context aggregation:

1. **Greedy Decoding Advantage:** Under standard greedy decoding, **QuartzNet_attn 5x5** (9.3M params) outperforms the deeper **QuartzNet 10x5** (12.8M params), reducing WER from **9.03%** to **8.90%**. This confirms that replacing the standard 1x1 Conv head with a Self-Attention mechanism allows the model to capture global acoustic context more effectively than merely stacking deeper 1D depthwise separable convolutions.
2. **Language Model Rescoring Dynamics:** However, when integrating **KenLM**, **QuartzNet 10x5** achieves a lower WER (**6.07%** vs **6.33%**). While the Language Model resolves phrase-level semantic ambiguities, the deeper convolutional backbone of the 10x5 model provides richer, fine-grained temporal representations. The lighter 5x5 backbone, despite its self-attention head, hits a structural capacity limit in feature extraction that the LM cannot fully compensate for.

---

### ⚙️ Hyperparameter Configuration

| Category | Hyperparameter | Value | Rationale / Note |
| :--- | :--- | :---: | :--- |
| **Optimization** | Optimizer | `AdamW` | Standard Adam with decoupled weight decay |
| | Base Learning Rate ($\eta_0$) | `1e-3` | Initial learning rate |
| | Weight Decay | `3e-5` | Applied for $L_2$ regularization |
| | Gradient Clipping | `1.0` | Prevents gradient explosion during CTC training |
| **Scheduling** | Scheduler | `CosineAnnealingLR` | Cosine decay down to $\eta_{\min} = 1e-5$ |
| **Setup** | Batch Size | `64` | Global batch size across training runs |
| **KenLM Decoding** | N-gram Order | `5` | Optimal trade-off between memory footprint and context length |
| | Alpha ($\alpha$) | `0.8` | LM weight empirical tuning |
| | Beta ($\beta$) | `2.0` | Word insertion penalty to favor longer complete words |
| | Beam Width | `128` | Balance between search depth and decoding latency |

---

## 🎙️ Acoustic Features & Audio Preprocessing

Audio signals are resampled and converted into Log-Mel Spectrograms during the preprocessing pipeline prior to feature extraction.

| Parameter | Value | Description / Configuration |
| :--- | :---: | :--- |
| **Sampling Rate** | `16,000 Hz` | Standard mono audio sample rate |
| **Window Length (win_length)** | `25 ms` | Frame length for STFT (400 samples) |
| **Hop Length (hop_length)** | `10 ms` | Frame shift / stride (160 samples) |
| **FFT Size (n_fft)** | `400` | Number of FFT components |
| **Mel Frequency Bins (n_mels)** | `80` | Number of Mel filterbank channels |
| **Audio Scaling** | `Decibel (dB)` | Converts Power Spectrogram to log-scale ($10 \cdot \log_{10}(S)$) to align with human auditory perception |
| **Feature Normalization** | `InstanceNorm1d` | Applied at the input layer across mel-channels to stabilize channel-wise feature distribution per sequence |

