# Project 06: Customer Segmentation (Unsupervised Clustering)

## 🎯 Goal
Segment retail mall customers into distinct behavioral cohorts using **Unsupervised Machine Learning** (K-Means, Hierarchical Clustering, and DBSCAN) based on demographic and purchasing behavior (Age, Annual Income, and Spending Score).

Unsupervised learning requires no ground-truth labels — instead, algorithms uncover hidden geometric patterns to empower data-driven marketing strategies.

---

## 👥 The 5 Behavioral Customer Personas

By analyzing **Annual Income** ($k\$$) versus **Spending Score** ($1–100$), the algorithm automatically isolates 5 distinct consumer groups:

| Cluster | Persona | Income Level | Spending Level | Key Demographic Traits | Recommended Marketing Action |
|:-------:|:--------|:------------:|:--------------:|:----------------------:|:----------------------------|
| **01** | 💎 **VIP / Target** | High ($>$ $70k) | High ($>$ 65) | Prime target, high disposable income | Premium loyalty programs, early access, VIP concierges |
| **02** | 💰 **Careful / Savers** | High ($>$ $70k) | Low ($<$ 40) | Financially disciplined, high wealth | Value-oriented promotions, quality guarantees, investment-grade items |
| **03** | 🛍️ **Trend / Impulsive** | Low ($<$ $40k) | High ($>$ 65) | Young, fashion-forward, trend followers | Installment financing (BNPL), discounts on trending streetwear/gadgets |
| **04** | 🪙 **Sensible / Budget** | Low ($<$ $40k) | Low ($<$ 40) | Price-sensitive, frugal shoppers | Essential coupons, bundle savings, clearance sales |
| **05** | ⚖️ **Standard / Balanced** | Mid ($40k–$70k) | Mid (40–60) | Average income and spending habits | Broad-appeal marketing, seasonal catalog mailers, membership points |

---

## 📁 Project Structure

```
06_customer_segmentation/
├── README.md               # Project documentation & persona playbooks
├── data/
│   └── Mall_Customers.csv  # Customer demographics and spending dataset
├── eda.py                  # Exploratory Data Analysis & visual cluster scatter
├── train.py                # Elbow method, Silhouette analysis & clustering pipeline
├── segment.py              # Interactive CLI & new customer segmentation profiler
└── models/
    └── customer_segmentation.joblib  # Saved K-Means pipeline, centroids & personas
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Examine customer distributions, gender spending comparisons, and feature correlations:
```bash
python3 06_customer_segmentation/eda.py
```

### 2. Determine Optimal Clusters & Train Models
Evaluate the Elbow method ($WCSS$), Silhouette scores ($k=2 \dots 10$), and benchmark K-Means against Hierarchical and DBSCAN clustering:
```bash
python3 06_customer_segmentation/train.py
```

### 3. Profile New Customers (Inference)
Classify a new customer with CLI flags:
```bash
python3 06_customer_segmentation/segment.py --income 85 --spending-score 82 --age 28
```

Or launch the interactive profiling wizard:
```bash
python3 06_customer_segmentation/segment.py
```

---

## 📊 Key Machine Learning Concepts Explored

- **Unsupervised Learning**: Discovering patterns and grouping data points without predefined target labels.
- **The Elbow Method**: Tracking Inertia / Within-Cluster Sum of Squares ($WCSS$) to find the inflection point of optimal $k$.
- **Silhouette Coefficient**: Evaluating intra-cluster tightness versus inter-cluster separation ($[-1, 1]$ scale).
- **Feature Standardization**: Scaling features to ensure high-range variables (Income) don't overpower standardized metrics (Spending Score).
