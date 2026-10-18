# Project 09: Titanic Survival Prediction (Tabular Classification)

## 🎯 Goal
Predict passenger survival aboard the **RMS Titanic** using supervised machine learning on the canonical **Kaggle Titanic** benchmark dataset.

The RMS Titanic disaster of April 15, 1912, resulted in 1,502 casualties among 2,224 passengers and crew. By modeling the intricate sociological interactions between ticket class, gender, age, family dynamics, and honorific titles, this project demonstrates end-to-end tabular classification with strict data leakage prevention and ensemble benchmarking.

---

## 🚢 Passenger Features & Manifest Attributes

### Raw Attributes
- `PassengerId`: Unique identifier for each traveler
- `Pclass`: Ticket class (`1 = 1st Class (Upper)`, `2 = 2nd Class (Middle)`, `3 = 3rd Class (Steerage)`)
- `Name`: Full passenger name including honorific titles (e.g., *Mr, Mrs, Miss, Master, Dr*)
- `Sex`: Gender (`male` or `female`)
- `Age`: Age in fractional or whole years (~20% missing, imputed via Title/Pclass grouping)
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Ticket`: Ticket number string
- `Fare`: Passenger fare paid in British Pounds (£)
- `Cabin`: Cabin room identifier (letter denotes deck level: A–G)
- `Embarked`: Port of embarkation (`C = Cherbourg`, `Q = Queenstown`, `S = Southampton`)

### Engineered Features
1. **Title Extraction**: Extracted from name and mapped to standardized cohorts (`Mr`, `Miss`, `Mrs`, `Master`, `Officer`, `Noble`, `Rare`). Separates young boys (`Master`, ~57.5% survival) from adult men (`Mr`, ~15.7% survival).
2. **FamilySize**: Total traveling unit (`SibSp + Parch + 1`).
3. **IsAlone**: Binary indicator (1 if traveling solo, 0 if traveling with family). Small families (2–4 persons) enjoyed ~55–72% survival, whereas solo travelers (~30%) and large families (>4 persons, ~16%) suffered high casualty rates.
4. **Deck**: Extracted first character from `Cabin` mapping to deck levels A–G or `U` (Unknown).
5. **Fare_Log**: $\log(1 + \text{Fare})$ transformation to handle extreme right skewness.

---

## 📁 Project Structure

```
09_titanic_survival_prediction/
├── README.md               # Project documentation & benchmark guide
├── features.py             # Feature engineering, imputation & data preparation
├── eda.py                  # Exploratory Data Analysis & demographic audits
├── train.py                # 5-fold stratified cross-validation & model benchmarking
├── predict.py              # CLI predictor, interactive simulation & Kaggle submission generator
├── data/
│   ├── train.csv           # 891 labeled historical training passengers
│   ├── test.csv            # 418 unlabelled Kaggle test passengers
│   └── submission.csv      # Kaggle-ready batch prediction output
└── models/
    └── titanic_classifier.joblib  # Trained champion model pipeline & fitted medians
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Analyze class distributions, missing value structures, and survival rates across gender, age brackets, and traveling party size:
```bash
python3 09_titanic_survival_prediction/eda.py
```

### 2. Train and Benchmark Classifiers
Evaluates 7 candidate classification architectures with 5-fold Stratified Cross-Validation:
```bash
python3 09_titanic_survival_prediction/train.py
```

### 3. Predict Passenger Survival (CLI Flags)
Test specific passenger profiles directly from the command line:

- **1st Class Young Female (e.g. Rose Bukater)**:
```bash
python3 09_titanic_survival_prediction/predict.py --pclass 1 --sex female --age 17 --fare 150 --embarked S
```

- **3rd Class Young Male (e.g. Jack Dawson)**:
```bash
python3 09_titanic_survival_prediction/predict.py --pclass 3 --sex male --age 20 --fare 8.05 --embarked S
```

### 4. Interactive Simulation Wizard
Launch the console menu to simulate historical passengers or build custom passenger profiles:
```bash
python3 09_titanic_survival_prediction/predict.py
```

### 5. Generate Kaggle Test Set Submissions
Batch predict all 418 passengers in `data/test.csv` and export `data/submission.csv`:
```bash
python3 09_titanic_survival_prediction/predict.py --batch 09_titanic_survival_prediction/data/test.csv
```

---

## 🏆 Model Benchmark & Evaluation Leaderboard

Evaluated on 891 passengers (80% train / 20% held-out test split, stratified by survival):

| Rank | Model Architecture | 5-Fold Stratified CV | Held-Out Test Acc | F1-Score | ROC-AUC | Key Characteristics |
|:----:|:-------------------|:--------------------:|:-----------------:|:--------:|:-------:|:--------------------|
| 🥇 | **Support Vector Machine (RBF)** | **82.59% (±1.60%)** | **84.36%** | **0.785** | **0.853** | Smooth non-linear decision boundary; top generalizer |
| 🥈 | **Logistic Regression (L2)** | 82.31% (±2.91%) | 83.80% | 0.782 | 0.867 | Interpretable log-odds; highest probability discrimination |
| 🥉 | **Hist Gradient Boosting** | 82.30% (±2.10%) | 83.24% | 0.769 | 0.858 | Fast histogram-binned tree ensemble |
| 4 | **Decision Tree (Pruned)** | 81.46% (±2.26%) | 81.56% | 0.756 | 0.847 | High interpretability with depth-limiting regularization |
| 5 | **Random Forest Classifier** | 82.59% (±2.22%) | 81.01% | 0.738 | 0.845 | Bagged ensemble of 200 de-correlated decision trees |
| 6 | **Gradient Boosting** | 81.18% (±1.90%) | 79.89% | 0.710 | 0.849 | Sequential residual boosting on shallow trees |
| 7 | **K-Nearest Neighbors (k=7)** | 81.33% (±2.05%) | 78.21% | 0.688 | 0.839 | Metric distance classification in scaled feature space |

---

## 📊 Top Discriminative Survival Factors

Permutation importance analysis on the champion pipeline reveals the primary factors governing passenger survival:

1. **Sex (40.6%)**: The single strongest factor — Women survived at ~74.2% versus Men at ~18.9%.
2. **Honorific Title (19.5%)**: Critical for separating young boys (`Master`, 57.5% survival) from adult men (`Mr`, 15.7% survival).
3. **Age (9.9%)**: Children under 10 had preferential lifeboat access (~61.3% survival).
4. **SibSp & Family Dynamics (7.0%)**: High sibling counts in steerage correlated with severe casualty risk.
5. **Fare Paid (6.7%)**: Proxy for upper-deck cabin locations and emergency access.
6. **Ticket Class (Pclass, 6.3%)**: 1st class passengers (63.0% survival) had 2.6x the survival odds of 3rd class (24.2%).

---

## 💡 Key Machine Learning Concepts Explored

- **Data Leakage Prevention**: Imputation medians (Age by Title, Fare by Class) are computed strictly on the training partition and applied uniformly to test and inference datasets.
- **Categorical Encoding**: Robust handling of unknown categories using Scikit-Learn `OneHotEncoder(handle_unknown='ignore')` in a `ColumnTransformer`.
- **Feature Engineering from Text**: Parsing unstructured names to extract social status and age signals via honorific titles.
- **Cross-Validation vs. Held-Out Generalization**: Utilizing Stratified K-Fold CV to ensure realistic variance estimates on moderate-sized tabular datasets.
