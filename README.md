```text
                                      d8                                 d8   
 e88~-888 888  888   /~~~8e  888-~\ _d88__  ~~~d88P 888-~88e  e88~~8e  _d88__ 
d888  888 888  888       88b 888     888      d88P  888  888 d888  88b  888   
8888  888 888  888  e88~-888 888     888     d88P   888  888 8888__888  888   
Y888  888 888  888 C888  888 888     888    d88P    888  888 Y888    ,  888   
 "88_-888 "88_-888  "88_-888 888     "88_/ d88P___  888  888  "88___/   "88_/ 
      888                                                                     

```
### Main Results on Common Voice it v26 with greedy decode and KenLM

#with greedy decode

| Method | Params (M) | Test Loss | CER (%) | WER (%)
| :---: | :---: | :---: | :---: | :---: |
| QuartzNet 5x5| 6.7M | 0.4284 | 38.03 | 10.15 |
| QuartzNet_attn 5x5  | 9.3M | 0.4422 | 33.35 | 8.90 |
| QuartzNet 10x5 | 12.8M | 0.3647 | 33.65 | 9.03 |
| QuartzNet_attn 10x5 | 12.7M | **0.3508** | **28.01** | **7.43** |





#with KenLM
| Method | Params (M) | Test Loss | CER (%) | WER (%)
| :---: | :---: | :---: | :---: | :---: |
| QuartzNet 5x5 + KenLM | 6.7M | 0.4284 | 19.38 | 6.80 |
| QuartzNet_attn 5x5 + KenLM  | 9.3M | 0.4422 | 19.16 | 6.33 |
| QuartzNet 10x5 + KenLM | 12.8M | 0.3647 | 17.04 | 6.07 |
| QuartzNet_attn 10x5 +KenLM | 12.7M | **0.3508** | **15.80** | **5.29** |

                                                       
                                                          


## 📝 Methodological & Architectural Notes

### 1. Architectural Decisions & Rationale
* **SiLU (Swish) Activation**: chosen instead of ReLu for its improved gradient flow during backpropagation.
* **Parametric Efficiency**: Scaled down channels specifically for CIFAR-10, demonstrating that large parameter budgets are unnecessary for $32 \times 32$ image resolution
* **Weight Initialization**: 
  * *Convolutional Layers*: Kaiming (He) Normal. 
  * *Linear Layer (FC)*: Xavier (Glorot) Normal.
 
* **Data Loader**: Loaded the complete CIFAR-10 dataset directly into GPU memory (VRAM), bypassing CPU-GPU data transfer bottlenecks and eliminating traditional DataLoader overhead.

### 2. Data Pipeline & Augmentation Strategy
for overcome overfitting in the small CIFAR-10 scenario i did some augmentation :
* **MixUp**: combining two different images overlapping them into one "hybrid" image
* **CutMix**: combining two different images into one, picking pieces of both (like 60% cat and 40% dog)
* **Normalization**: normalizion based on the CIFAR-10 statistics  ($\mu = [0.4914, 0.4822, 0.4465]$, $\sigma = [0.2023, 0.1994, 0.2010]$).

---

### 3. Hyperparameters Summary

| Category | Hyperparameter | Value | Rationale / Note |
| :--- | :--- | :---: | :--- |
| **Optimization** | Optimizer | `AdamW` | better than SDG+momentum |
| | Base Learning Rate ($\eta_0$) | `1e-3` | not too big to get better convergence|
| | Weight Decay | `0.05` | Applied for strong $L_2$ regularization | 
| | Augmentation Probability | `1 or 0.5` | Empirically set to $p=1.0$ for heavy regularization, disabled during the final 15 epochs (epoch 135+ for 150e, epoch 45+ for 50e) for clean fine-tuning
| | Alpha  | `1` | High value selected to increase task difficulty and prevent network memorization |
| **Scheduling** | OneCycleLR | `CosineAnnealingLR` | cosine curve down to $\eta_{min}=1e-7$ |
| **Setup** | Batch Size | `128` | higher batch size with small datasets will overfit due to the model seeing almost all the dataset in one epoch|
| | Loss Function | `CrossEntropyLoss` | Label smoothing was omitted as MixUp/CutMix already provides soft target labels |


### 4. Epochs
with less epochs this resnet can still obtain good results:
| Model | Params (M) | Test Acc (%) | Time | Epochs
| :---: | :--- | :---: | :--- | :---: |
| ResNet-18 (He et al.) | 11.17 | 91.15 | 5m 7s | 50 |
| **Custom ResNet Medium** | **4.36** | **91.58** | **3m 24s** | 50 |
| **Custom ResNet Tiny** | **1.09** | **88.76** | **3m 10s** | 50 |
| **Custom ResNet Mini** | **0.27** | **85.36** | **2m 50s** | 50 |



## 📈 Training Dynamics & Convergence

Below are the test accuracy and loss trajectory comparisons with 150 epochs


ResNet (He et al.)
![Training Curves](images/curve_resnet.png)

ResNet Medium
![Training Curves](images/curve_resnet_medium.png)

ResNet Tiny
![Training Curves](images/curve_resnet_tiny.png)

ResNet Mini
![Training Curves](images/curve_resnet_mini.png)









