# 🧠 Breast Cancer Classification using EfficientNetB3 + CBAM

## 📌 What this project is about
This project is an attempt to build a deep learning model that can classify breast cancer histopathological images into multiple categories.

I used **EfficientNetB3** as the base model and improved it with an attention module (**CBAM**) so the model can focus on the most important parts of an image instead of treating everything equally.

The dataset used is **BreakHis**, which contains microscopic images of breast tumor tissues.

---

## 🎯 What I wanted to achieve
- Build a reliable multi-class classifier for medical images
- Improve performance using attention mechanisms (CBAM)
- Handle class imbalance properly (very common in medical datasets)
- Use data augmentation to avoid overfitting
- Evaluate the model properly instead of just relying on accuracy

---

## 📂 Dataset
- **Dataset:** BreakHis (Breast Cancer Histopathological Dataset)
- **Source:** Kaggle
- **Magnification used:** 100X

The dataset contains **8 different tumor classes**, making this a multi-class classification problem.

---

## ⚙️ Tech stack
- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Kaggle API

---

## 🧪 Data preprocessing
Before training, I applied:
- Resizing all images to **300 × 300**
- Data augmentation:
  - Horizontal & vertical flips
  - Random rotations
  - Color jitter

The dataset was split into:
- Training set
- Validation set
- Test set

---

## 🧠 Model approach

### Base model
I used a pretrained **EfficientNetB3**, since training from scratch on medical data usually doesn’t work well.

### What I added
I integrated **CBAM (Convolutional Block Attention Module)**:
- Channel attention → *what features matter*
- Spatial attention → *where to focus in the image*

### Training strategy
Instead of training everything at once:
1. **Warmup phase**
   - Froze the backbone
   - Trained only the classifier

2. **Fine-tuning phase**
   - Unfroze the full model
   - Trained everything together

---

## 📊 Training setup
- **Loss function:** CrossEntropy (with class weights + label smoothing)
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau

This setup helped stabilize training and deal with imbalance.

---

## 📈 How I evaluated the model
Instead of just accuracy, I used:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)

This is important because medical datasets are rarely balanced.

---

## 🧾 Results
The model was evaluated on the test set and the best version was saved as:
