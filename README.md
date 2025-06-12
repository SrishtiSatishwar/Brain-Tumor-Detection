# 🧠 Brain Tumor Detection: CNN vs. SVM

This project compares two approaches for classifying MRI images as **tumor** or **non-tumor**:

- A **Convolutional Neural Network (CNN)** using TensorFlow/Keras
- A **Support Vector Machine (SVM)** using scikit-learn

---

## 📁 Dataset

The dataset consists of MRI scans stored in two folders:

- `yes-tumor-meningioma-3/` — images with brain tumors (label **1**)
- `no-tumor3/` — images without tumors (label **0**)

All images are resized to **64×64** pixels before processing.

---

## 🔄 Preprocessing

For both models:
- Load and label images from folders
- Resize to 64x64
- Normalize pixel values (CNN only)
- Shuffle and split into training/testing sets (80%/20%)

---

## 🤖 Model 1: CNN (TensorFlow/Keras)

### 📐 Architecture

```text
Conv2D(32, 3x3) → ReLU  
MaxPooling2D(2x2)  
Flatten  
Dense(64) → ReLU  
Dense(1) → Sigmoid
```

### 🏋️ Training

- Loss: `binary_crossentropy`
- Optimizer: `Adam`
- Epochs: `40`
- Batch Size: `32`

### 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- ROC Curve

---

## 🤖 Model 2: SVM (scikit-learn)

### ⚙️ Configuration

- Kernel: `linear`
- Probability estimates: enabled

### 🧮 Training Process

- Images flattened into 1D arrays (`64x64x3 → 12288`)
- Fit a `SVC(kernel='linear', probability=True)` on training data

### 📊 Evaluation Metrics

Same metrics as CNN:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- ROC Curve

---

## 🔬 Sample Output (Both Models)

```
Accuracy:    0.XX  
Precision:   0.XX  
Recall:      0.XX  
F1 Score:    0.XX  
ROC AUC:     0.XX
```

---

## 📈 ROC Curve

Both models output an ROC Curve using:

```python
from sklearn.metrics import roc_curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

---

## 📦 Requirements

Install all dependencies with:

```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
```

---

## 🚀 How to Run

1. Place your image folders at:

```
/content/yes-tumor-meningioma-3/  
/content/no-tumor3/
```

2. Run the notebook/script for:
   - `cnn_model.py` or `cnn_notebook.ipynb`
   - `svm_model.py` or `svm_notebook.ipynb`

---

## 🧩 Future Improvements

- Add image augmentation
- Use pretrained CNNs (e.g. ResNet, MobileNet)
- Apply Grad-CAM for model explainability
- Try different SVM kernels (RBF, polynomial)
- Expand dataset for better generalization

---

## 📁 Repository Structure

```
.
├── cnn_model.py / .ipynb        # CNN pipeline
├── svm_model.py / .ipynb        # SVM pipeline
├── README_MERGED.md             # This file
└── data/
    ├── yes-tumor-meningioma-3/
    └── no-tumor3/
```
