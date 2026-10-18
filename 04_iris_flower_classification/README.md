# Project 04: Iris Flower Classification (Multi-class Classification)

## 🎯 Goal
Classify Iris flowers into three distinct species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) based on four biometric measurements of their sepals and petals.

Introduced by the British statistician and biologist Ronald Fisher in 1936, the Iris dataset is the canonical "Hello World" of multi-class supervised machine learning.

---

## 🌸 Botanical Features & Species

### Measurements (Features)
1. **Sepal Length** ($cm$): Length of the outer protective leaf
2. **Sepal Width** ($cm$): Width of the outer protective leaf
3. **Petal Length** ($cm$): Length of the inner flower petal
4. **Petal Width** ($cm$): Width of the inner flower petal

### Target Species (Classes)
- `0: Iris Setosa` — Distinctively smaller petals; linearly separable from the other two species.
- `1: Iris Versicolor` — Medium-sized petals and sepals; intermediate characteristics.
- `2: Iris Virginica` — Largest petals and sepals; slight boundary overlap with Versicolor.

---

## 📁 Project Structure

```
04_iris_flower_classification/
├── README.md               # Project documentation & benchmark guide
├── eda.py                  # Exploratory Data Analysis & feature distributions
├── train.py                # Model training, benchmarking & export pipeline
├── predict.py              # Interactive CLI & parameter-driven flower predictor
└── models/
    └── iris_classifier.joblib  # Saved trained model & scaler pipeline
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Examine feature summary statistics, class distributions, and pairwise separations:
```bash
python3 04_iris_flower_classification/eda.py
```

### 2. Train and Benchmark Classifiers
Trains KNN, Logistic Regression, SVM, Decision Tree, and Random Forest models with stratified evaluation:
```bash
python3 04_iris_flower_classification/train.py
```

### 3. Predict Flower Species
Run the interactive console menu:
```bash
python3 04_iris_flower_classification/predict.py
```

Or predict directly with command-line flags:
```bash
python3 04_iris_flower_classification/predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

---

## 📊 Key Machine Learning Concepts Explored

- **Multi-class Classification**: Softmax / multinomial classification across 3 discrete targets.
- **Linear Separability vs. Overlap**: Understanding why Setosa is 100% separable while Versicolor/Virginica require non-linear or soft-margin boundaries.
- **Feature Importance**: Discovering which physical measurements (Petal dimensions vs. Sepal dimensions) provide the strongest discriminative power.
- **Model Pipeline**: Standardizing features using `StandardScaler` inside a Scikit-Learn `Pipeline` to prevent data leakage.
