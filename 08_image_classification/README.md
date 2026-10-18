# Project 08: Image Classification with Deep Learning (CNN on CIFAR-10)

## 🎯 Goal
Build, train, and evaluate a **Deep Convolutional Neural Network (CNN)** from scratch in **PyTorch** to classify 32×32 color images into 10 real-world categories using the canonical **CIFAR-10** benchmark dataset.

The project explores fundamental computer vision concepts: spatial convolutions, hierarchical feature extraction, pooling, dropout regularization, and Apple Silicon GPU acceleration (`mps`).

---

## 🖼️ The 10 CIFAR-10 Image Classes

| Index | Class | Icon | Category Type |
|:-----:|:------|:----:|:--------------|
| **0** | Airplane | ✈️ | Vehicle |
| **1** | Automobile | 🚗 | Vehicle |
| **2** | Bird | 🐦 | Animal |
| **3** | Cat | 🐱 | Animal |
| **4** | Deer | 🦌 | Animal |
| **5** | Dog | 🐶 | Animal |
| **6** | Frog | 🐸 | Animal |
| **7** | Horse | 🐴 | Animal |
| **8** | Ship | 🚢 | Vehicle |
| **9** | Truck | 🚚 | Vehicle |

---

## 🧠 Convolutional Neural Network (CNN) Architecture

```
Input RGB Image (3 x 32 x 32)
    │
    ▼
[Conv Block 1]: Conv2D(3->32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32->32, 3x3) -> ReLU -> MaxPool2D(2x2) -> Dropout(0.2)
    │  (Output: 32 channels @ 16x16)
    ▼
[Conv Block 2]: Conv2D(32->64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64->64, 3x3) -> ReLU -> MaxPool2D(2x2) -> Dropout(0.3)
    │  (Output: 64 channels @ 8x8)
    ▼
[Conv Block 3]: Conv2D(64->128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.4)
    │  (Output: 128 channels @ 4x4)
    ▼
[Flatten]: 128 * 4 * 4 = 2,048 features
    │
    ▼
[Dense Block]: Linear(2048 -> 256) -> BatchNorm -> ReLU -> Dropout(0.5)
    │
    ▼
[Classifier]: Linear(256 -> 10 classes) -> Logits / Softmax Probabilities
```

---

## 📁 Project Structure

```
08_image_classification/
├── README.md               # Architecture documentation & training guide
├── data/                   # CIFAR-10 dataset
├── eda.py                  # Dataset inspection & ASCII color grid rendering
├── model.py                # PyTorch CNN module definition
├── train.py                # Training loop with validation & model checkpointing
├── predict.py              # Interactive image classifier CLI with probability bars
└── models/
    └── cifar10_cnn.pt      # Saved PyTorch model weights & evaluation metrics
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Inspect class balance, tensor dimensions, and ASCII image previews:
```bash
python3 08_image_classification/eda.py
```

### 2. Train the Convolutional Neural Network
Trains the CNN with Adam optimizer, cross-entropy loss, and Apple Silicon MPS/CPU acceleration:
```bash
python3 08_image_classification/train.py --epochs 10 --batch-size 64
```

### 3. Predict & Test Images
Run inference on individual test samples or custom images:
```bash
python3 08_image_classification/predict.py --sample-index 42
```

Or launch the interactive visual console:
```bash
python3 08_image_classification/predict.py
```

---

## 📊 Key Deep Learning Concepts Explored

- **Spatial Invariance & Feature Hierarchies**: Lower conv layers detect edges/colors; deeper layers combine them into complex object parts (wheels, wings, eyes).
- **Batch Normalization**: Stabilizes and accelerates gradient descent across deep layers.
- **Dropout Regularization**: Prevents co-adaptation of neurons and combats overfitting on small images.
- **Hardware Acceleration**: Automatically selects `mps` (Metal Performance Shaders on Apple Silicon) or `cuda` over `cpu`.
