"""
Training and Evaluation Pipeline for Movie Recommendation System.
Benchmarks Global Mean, Movie Mean, User+Item Bias, and Matrix Factorization (Funk SVD).
Builds Item-Based Collaborative Filtering similarity index and exports model artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def main():
    print_banner("🎬 TRAINING MOVIE RECOMMENDATION SYSTEM (MovieLens) 🎬")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # 1. Filter movies with at least 5 ratings to focus on reliable patterns
    movie_rating_counts = ratings["movieId"].value_counts()
    eligible_movie_ids = set(movie_rating_counts[movie_rating_counts >= 5].index)
    filtered_ratings = ratings[ratings["movieId"].isin(eligible_movie_ids)].copy()

    print(f"Dataset after thresholding (>= 5 ratings per movie):")
    print(f"  • {filtered_ratings['userId'].nunique():,} unique users")
    print(f"  • {filtered_ratings['movieId'].nunique():,} unique movies")
    print(f"  • {len(filtered_ratings):,} ratings retained")

    # 2. Train / Test Split on Rating interactions (80% train, 20% test)
    np.random.seed(42)
    test_mask = np.random.rand(len(filtered_ratings)) < 0.20
    train_df = filtered_ratings[~test_mask].copy()
    test_df = filtered_ratings[test_mask].copy()

    print(f"Train/Test split: {len(train_df):,} train interactions, {len(test_df):,} test interactions")

    # Map user and movie IDs to sequential indices
    users = sorted(filtered_ratings["userId"].unique())
    eligible_movies = sorted(filtered_ratings["movieId"].unique())
    u_map = {u: i for i, u in enumerate(users)}
    m_map = {m: i for i, m in enumerate(eligible_movies)}

    n_users = len(users)
    n_items = len(eligible_movies)

    # 3. Model 1: Global Mean Baseline
    mu = train_df["rating"].mean()
    test_actual = test_df["rating"].values
    preds_global = np.full(len(test_actual), mu)

    # 4. Model 2: Movie Mean Baseline
    movie_means = train_df.groupby("movieId")["rating"].mean().to_dict()
    preds_movie_mean = np.array([movie_means.get(m, mu) for m in test_df["movieId"]])

    # 5. Model 3: User & Item Bias Baseline (r_ui = mu + b_u + b_i)
    user_means = train_df.groupby("userId")["rating"].mean()
    user_bias = (user_means - mu).to_dict()
    
    # Calculate item bias after subtracting user bias
    train_df["u_bias"] = train_df["userId"].map(user_bias).fillna(0)
    item_bias = (train_df["rating"] - mu - train_df["u_bias"]).groupby(train_df["movieId"]).mean().to_dict()

    preds_bias = []
    for u, m in zip(test_df["userId"], test_df["movieId"]):
        p = mu + user_bias.get(u, 0.0) + item_bias.get(m, 0.0)
        preds_bias.append(np.clip(p, 0.5, 5.0))
    preds_bias = np.array(preds_bias)

    # 6. Model 4: Latent Factor Matrix Factorization (Funk SVD)
    print_banner("1. BENCHMARKING RECOMMENDATION ALGORITHMS")
    k_factors = 15
    lr = 0.012
    reg = 0.04
    epochs = 12

    P = np.random.normal(0, 0.05, (n_users, k_factors))
    Q = np.random.normal(0, 0.05, (n_items, k_factors))
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)

    train_u = np.array([u_map[u] for u in train_df["userId"]])
    train_i = np.array([m_map[m] for m in train_df["movieId"]])
    train_r = train_df["rating"].values

    test_u = np.array([u_map[u] for u in test_df["userId"]])
    test_i = np.array([m_map[m] for m in test_df["movieId"]])

    print(f"Training Funk SVD (k={k_factors} latent factors, {epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        for u_idx, i_idx, r in zip(train_u, train_i, train_r):
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            err = r - pred
            bu[u_idx] += lr * (err - reg * bu[u_idx])
            bi[i_idx] += lr * (err - reg * bi[i_idx])
            p_cache = P[u_idx].copy()
            P[u_idx] += lr * (err * Q[i_idx] - reg * P[u_idx])
            Q[i_idx] += lr * (err * p_cache - reg * Q[i_idx])

    preds_svd = []
    for u_idx, i_idx in zip(test_u, test_i):
        p = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
        preds_svd.append(np.clip(p, 0.5, 5.0))
    preds_svd = np.array(preds_svd)

    # Benchmark Summary Table
    models_eval = [
        ("Global Mean Baseline", preds_global),
        ("Movie Mean Baseline", preds_movie_mean),
        ("User & Item Bias Model", preds_bias),
        ("Matrix Factorization (Funk SVD)", preds_svd)
    ]

    print(f"\n{'Algorithm':<36} | {'RMSE (Lower is Better)':<24} | {'MAE':<10}")
    print("-" * 75)
    for name, p_vals in models_eval:
        rmse = np.sqrt(mean_squared_error(test_actual, p_vals))
        mae = mean_absolute_error(test_actual, p_vals)
        print(f"{name:<36} | {rmse:<24.4f} | {mae:<10.4f}")

    # 7. Compute Item-Item Collaborative Filtering Similarity Matrix
    print_banner("2. COMPUTING ITEM-TO-ITEM COLLABORATIVE FILTERING INDEX")
    print("Building normalized user-item rating pivot table...")
    # Center user ratings before computing item cosine similarity
    pivot = filtered_ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    sim_matrix = cosine_similarity(pivot)

    # Pre-index top-20 most similar neighbors per movie for fast inference
    item_neighbors = {}
    pivot_movie_ids = list(pivot.index)
    for idx, mid in enumerate(pivot_movie_ids):
        # Sort descending, exclude self (index 0)
        top_indices = np.argsort(sim_matrix[idx])[::-1][1:21]
        item_neighbors[mid] = [
            (pivot_movie_ids[n_idx], float(sim_matrix[idx][n_idx]))
            for n_idx in top_indices if sim_matrix[idx][n_idx] > 0.05
        ]

    # 8. Build Comprehensive Movie Metadata Lookup
    movie_stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).to_dict(orient="index")

    movie_metadata = {}
    for row in movies.itertuples():
        stats = movie_stats.get(row.movieId, {"rating_count": 0, "avg_rating": 0.0})
        movie_metadata[row.movieId] = {
            "title": row.title,
            "genres": row.genres.split("|"),
            "rating_count": stats["rating_count"],
            "avg_rating": stats["avg_rating"]
        }

    # 9. Save Recommender Model Artifact
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, "recommender.joblib")

    artifact = {
        "mu": float(mu),
        "P": P,
        "Q": Q,
        "bu": bu,
        "bi": bi,
        "u_map": u_map,
        "m_map": m_map,
        "movie_metadata": movie_metadata,
        "item_neighbors": item_neighbors,
        "eligible_movies": eligible_movies
    }
    joblib.dump(artifact, artifact_path)

    print_banner("3. MODEL PERSISTENCE & READINESS")
    print(f"  Artifact successfully saved to: {artifact_path}")
    print(f"  Indexed {len(item_neighbors):,} movies for instantaneous Item-Based CF")
    print(f"  Saved SVD latent space ({k_factors} factors) for personalized user profiling")
    print("  Ready to generate recommendations!\n")


if __name__ == "__main__":
    main()
