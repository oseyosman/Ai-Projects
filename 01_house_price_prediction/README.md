# Project 01: Michigan House Price Prediction (Regression)

## 🎯 Goal
Predict residential house prices across Michigan metropolitan and regional real estate markets (Ann Arbor, Birmingham, Northville, Novi, Detroit, Grand Rapids, Lansing, Troy, Traverse City, etc.) using supervised machine learning regression algorithms.

---

## 📊 Dataset: Michigan Real Estate
- **Sample Size**: 5,000 residential sales records (`data/michigan_housing.csv`).
- **Target Variable**: `price` (Sale price in USD).
- **Features**:
  1. `city`: Metro region (`Ann Arbor`, `Birmingham`, `Northville`, `Novi`, `Detroit`, `Grand Rapids`, `Lansing`, `Troy`, `Traverse City`, `Royal Oak`, `Kalamazoo`, `Rochester Hills`)
  2. `square_feet`: Finished interior living area (sqft)
  3. `bedrooms`: Number of bedrooms
  4. `bathrooms`: Number of bathrooms
  5. `lot_size_sqft`: Property lot area in square feet
  6. `year_built`: Year property was constructed (1920–2024)
  7. `garage_spaces`: Attached/detached garage capacity (0–3 cars)
  8. `has_basement`: 1 = Full basement (Michigan standard), 0 = Slab/crawlspace
  9. `has_lake_access`: 1 = Lake frontage or deeded lake access (Great Lakes state perk)
  10. `school_rating`: Local public school district rating (1.0 to 10.0)

---

## 🧠 Machine Learning Concepts Covered
1. **Regression with Mixed Data Types**:
   - Categorical feature encoding (`OneHotEncoder`) for city locations.
   - Numerical feature standardization (`StandardScaler`).
   - Combined via Scikit-Learn `ColumnTransformer` and `Pipeline`.
2. **Model Comparison & Benchmarking**:
   - **Linear Regression**: Baseline parametric model.
   - **Ridge Regression (L2)**: Regularized linear model.
   - **Decision Tree Regressor**: Non-linear tree partitions.
   - **Random Forest Regressor**: Ensemble of bagged decision trees.
   - **Gradient Boosting Regressor**: Sequentially optimized boosted decision trees.
3. **Model Evaluation Metrics**:
   - **$R^2$ Score**: Coefficient of determination ($R^2 > 0.90$).
   - **MAE ($)**: Mean Absolute Error in actual dollars.
   - **RMSE ($)**: Root Mean Squared Error.
4. **Model Persistence**:
   - Best model pipeline exported to `models/best_michigan_house_model.joblib`.

---

## 🚀 How to Run

### 1. Exploratory Data Analysis (EDA)
Inspect market distributions, city prices per square foot, and feature correlations:
```bash
python3 eda.py
```

### 2. Train and Benchmark Models
Train all 5 regression models, compare metrics on the test set, and save the winner:
```bash
python3 train.py
```

### 3. Interactive Predictions
Launch the interactive CLI estimator with preset Michigan homes or custom inputs:
```bash
python3 predict.py
```

Or pass direct flags via terminal:
```bash
python3 predict.py --city "Ann Arbor" --sqft 2500 --beds 4 --baths 2.5 --year 2004 --school 9.0
```
