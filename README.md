# EfficientNet-B3 + CBAM — Breast Cancer Multiclass Classification

Fine-tuned EfficientNet-B3 with Convolutional Block Attention Modules (CBAM) for 8-class histopathological image classification on the BreakHis dataset (100× magnification).

---

## Results

### Test Set Performance (352 samples, 44 per class)

| Metric | Score |
|---|---|
| Accuracy | **93.75%** (330/352) |
| Weighted Precision | 0.9433 |
| Weighted Recall | 0.9375 |
| Weighted F1 | 0.9382 |
| Test Loss | 0.7374 |
| Macro-avg AUC | **0.9963** |

### Per-Class Classification Report

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| adenosis | 1.00 | 1.00 | 1.00 | 1.0000 |
| ductal_carcinoma | 0.76 | 0.93 | 0.84 | 0.9843 |
| fibroadenoma | 0.93 | 0.95 | 0.94 | 0.9973 |
| lobular_carcinoma | 0.95 | 0.80 | 0.86 | 0.9923 |
| mucinous_carcinoma | 0.95 | 0.95 | 0.95 | 0.9965 |
| papillary_carcinoma | 0.98 | 0.91 | 0.94 | 0.9998 |
| phyllodes_tumor | 1.00 | 0.95 | 0.97 | 0.9951 |
| tubular_adenoma | — | — | — | 1.0000 |

---

## Dataset

**BreakHis** — Breast Cancer Histopathological Database
Source: [Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset) · License: DbCL-1.0

Magnification used: **100×**

| Class | Images | Class Weight |
|---|---|---|
| adenosis | 113 | 2.3020 |
| ductal_carcinoma | 903 | 0.2881 |
| fibroadenoma | 260 | 1.0005 |
| lobular_carcinoma | 170 | 1.5301 |
| mucinous_carcinoma | 222 | 1.1717 |
| papillary_carcinoma | 142 | 1.8319 |
| phyllodes_tumor | 121 | 2.1498 |
| tubular_adenoma | 150 | 1.7342 |
| **Total** | **2081** | — |

**Split** (stratified, seed=42): Train 1383 · Val 346 · Test 352 (44/class)

---

## Model Architecture

- **Backbone**: EfficientNet-B3 (ImageNet pretrained)
- **Attention**: CBAM injected after blocks 6 (136ch) and 7 (384ch)
  - Channel Attention: dual avg/max-pool MLP with reduction ratio 16
  - Spatial Attention: 7×7 depthwise conv on concat(avg, max) feature maps
- **Head**: AdaptiveAvgPool → Dropout(0.4) → Linear(1536, 512) → BN → ReLU → Dropout(0.3) → Linear(512, 8)

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Input size | 300 × 300 |
| Batch size | 16 |
| Loss | CrossEntropyLoss (label_smoothing=0.1, class-weighted) |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=10) |

### Training Phases

**Warmup (5 epochs)** — backbone frozen, head + CBAM only
`lr=1e-3, weight_decay=1e-4`

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 2.6385 | 0.1931 | 2.2349 | 0.1474 |
| 3 | 2.1908 | 0.2719 | 1.9028 | 0.4335 |
| 5 | 2.0088 | 0.3601 | 1.7443 | 0.4220 |

**Full Fine-tuning (100 epochs)** — end-to-end
`lr=1e-4 → 1e-5 → 1e-6 (scheduled), weight_decay=1e-5`

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|---|---|---|---|---|
| 1 | 1.9144 | 1.6387 | 0.4451 | 0.4476 |
| 10 | 1.3085 | 1.1134 | 0.7486 | 0.7591 |
| 25 | 1.0489 | 0.9787 | 0.8642 | 0.8677 |
| 50 | 0.8689 | 0.9219 | 0.8988 | 0.8998 |
| 72 | 0.8187 | **0.8670** | 0.9220 | 0.9224 |
| 100 | 0.7988 | 0.8860 | 0.9220 | 0.9218 |

> Best checkpoint saved at epoch 95 (val_loss=0.8619)

---

## Data Augmentation

**Train**: RandomHorizontalFlip · RandomVerticalFlip · RandomRotation(20°) · ColorJitter(b=0.3, c=0.3, s=0.15, h=0.05) · RandomAffine(translate=0.05) · ImageNet normalize
**Val/Test**: ToTensor · ImageNet normalize

---

## Dependencies

```
torch · torchvision · torchinfo
scikit-learn · numpy · matplotlib · seaborn
tqdm
```

Install extras:
```bash
pip install torchinfo kaggle
```

---

## Usage

```python
# Load best checkpoint
model.load_state_dict(torch.load('best_efficientnet_b3_cbam_100X.pth'))

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(image_tensor.unsqueeze(0).to(device))
    pred = outputs.argmax(dim=1)
    print(class_names[pred.item()])
```