# Project 05: Movie Recommendation System (Collaborative Filtering)

## 🎯 Goal
Build an intelligent Movie Recommender System powered by **Collaborative Filtering** and **Matrix Factorization** (Singular Value Decomposition - SVD) using the gold-standard **MovieLens** dataset.

The system learns latent preferences from over 100,000 ratings to predict user enjoyment and discover similar titles.

---

## 🧠 Core Recommendation Paradigms Implemented

| Paradigm | How It Works | Strengths |
|----------|--------------|-----------|
| **Item-Based Collaborative Filtering** | Calculates Cosine Similarity between movie rating vectors across all users. | "Users who loved *The Matrix* also loved *Fight Club*." Stable and interpretable. |
| **Matrix Factorization (SVD)** | Decomposes the sparse User-Item matrix $R \approx U \times \Sigma \times V^T$ into $k$ latent preference factors (e.g. action intensity, mood, pacing). | Overcomes sparsity, predicts unrated titles, state-of-the-art accuracy. |
| **User Personalized Profiling** | Projects an active user's rating vector into the latent space to deliver personalized top-$N$ recommendations. | Highly tailored recommendations matching specific user taste. |
| **Content-Based Genre Matching** | Vectorizes genre metadata with TF-IDF / Multi-hot encoding for content fallback. | Solves cold-start when new movies have zero ratings. |

---

## 📁 Project Structure

```
05_movie_recommendation_system/
├── README.md               # Project documentation & algorithm guide
├── data/                   # MovieLens dataset (movies.csv, ratings.csv)
├── eda.py                  # Exploratory Data Analysis & sparsity metrics
├── train.py                # Matrix Factorization (SVD) training & RMSE benchmark
├── recommend.py            # Interactive CLI & movie recommender engine
└── models/
    └── recommender.joblib  # Saved latent embeddings, similarity index & metadata
```

---

## 🚀 Quick Start

### 1. Run Exploratory Data Analysis
Analyze dataset sparsity, rating distribution, and most-watched titles:
```bash
python3 05_movie_recommendation_system/eda.py
```

### 2. Train the Latent Factor Model
Decompose the interaction matrix, benchmark RMSE against a baseline mean model, and build the similarity index:
```bash
python3 05_movie_recommendation_system/train.py
```

### 3. Generate Recommendations
Find movies similar to your favorite title:
```bash
python3 05_movie_recommendation_system/recommend.py --movie "Toy Story"
```

Or launch the interactive discovery wizard (rate movies and get instant recommendations):
```bash
python3 05_movie_recommendation_system/recommend.py
```

---

## 📊 Key Machine Learning Concepts Explored

- **User-Item Interaction Matrix**: Transforming tabular logs `(user, item, rating)` into a high-dimensional sparse matrix.
- **Matrix Sparsity & Cold Start**: Analyzing why 98%+ of potential user-movie combinations are unrated and how collaborative filtering handles missing entries.
- **Latent Factor Representations**: Using SVD / TruncatedSVD to capture hidden concepts (themes, cinematic style) without manual tagging.
- **Evaluation with RMSE**: Measuring root mean squared error on held-out user ratings.
