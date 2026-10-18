# Project 03: Handwritten Digit Recognition (Multi-class Classification)

## 🎯 Goal
Classify handwritten digit images (0–9) from the **MNIST-like** dataset using traditional machine learning algorithms — without any deep learning — to understand how image data is represented and classified in raw pixel space.

---

## 📊 Dataset: Optical Handwritten Digits (Scikit-Learn `load_digits`)
A compact, offline MNIST-like dataset built into Scikit-Learn:
- **Total Samples**: 1,797 grayscale digit images
- **Image Size**: 8×8 pixels (64 features per image, flattened)
- **Pixel Values**: 0–16 (normalized grayscale intensity)
- **Classes**: 10 digits → `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`

> **Why not full MNIST (28×28, 70,000 samples)?**
> The sklearn built-in variant lets you train and evaluate in seconds without downloading gigabytes of data — ideal for learning core ML concepts before scaling up.

---

## 🧠 Machine Learning Concepts Covered

### 1. Image Data as Flat Feature Vectors
Each 8×8 image is **flattened** into a 1D array of 64 pixel values.
This is the simplest way to feed image data into traditional ML models.

### 2. Feature Scaling: Why It Matters for Images
- Pixel values range from 0 to 16. Algorithms sensitive to magnitude (SVM, KNN) need `StandardScaler` to normalize each pixel feature to zero mean and unit variance.

### 3. Multi-class Classification Algorithms Benchmarked
| Model | Strategy | Key Strength |
|---|---|---|
| **K-Nearest Neighbors (KNN)** | Majority vote among K closest training samples by Euclidean distance | Intuitive, high accuracy on pixel data |
| **Support Vector Machine (SVM)** | Learns an optimal hyperplane per class pair (One-vs-One) with RBF kernel | State-of-the-art on structured image tasks |
| **Logistic Regression** | Softmax multi-class via L2 regularization | Fast & interpretable |
| **Random Forest** | Ensemble of 200 decision trees on random pixel subsets | Robust, handles non-linearity |
| **Decision Tree** | Single greedy tree splitting on pixel thresholds | Baseline; highly interpretable |

### 4. Evaluation Metrics for Multi-class Problems
- **Accuracy**: Overall fraction of correctly classified digits
- **Confusion Matrix**: 10×10 matrix showing which digits are confused for which
- **Per-class Precision / Recall / F1**: Reveals which digits are hardest (e.g., `1` vs `7`, `3` vs `8`)

---

## 🚀 How to Run

### 1. Exploratory Data Analysis
Inspect class distribution, sample images, and pixel importance maps:
```bash
python3 eda.py
```

### 2. Train and Benchmark Models
Train 5 classifiers and show accuracy leaderboard + confusion matrix:
```bash
python3 train.py
```

### 3. Interactive Digit Predictor
Test the model on specific sample images or launch a batch evaluation:
```bash
python3 predict.py
```

Or specify a test sample index (0–1796):
```bash
python3 predict.py --index 42
```
