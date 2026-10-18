"""
Interactive Movie Recommendation Engine (MovieLens).
Supports:
1. Item-Based Collaborative Filtering (Find similar movies)
2. Latent Factor SVD Personalization (Rate movies or pick a User ID)
3. Genre-based curated recommendations
"""

import os
import sys
import argparse
import joblib
import numpy as np


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "recommender.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'.")
        print("Please train the model first by running:\n  python3 05_movie_recommendation_system/train.py")
        sys.exit(1)
    return joblib.load(model_path)


def format_stars(rating: float) -> str:
    full_stars = int(rating)
    half = "½" if (rating - full_stars) >= 0.3 else ""
    return f"{'★' * full_stars}{half}".ljust(6)


def format_similarity_bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {score * 100:>5.1f}% match"


def find_movies_by_title(query: str, metadata: dict, limit: int = 5):
    query_lower = query.strip().lower()
    matches = []
    for mid, info in metadata.items():
        if query_lower in info["title"].lower():
            matches.append((mid, info))
    # Sort by rating count (favor popular titles)
    matches.sort(key=lambda x: x[1]["rating_count"], reverse=True)
    return matches[:limit]


def recommend_similar_movies(target_mid: int, model_artifact: dict, top_n: int = 8):
    metadata = model_artifact["movie_metadata"]
    neighbors = model_artifact["item_neighbors"]

    target_info = metadata.get(target_mid)
    if not target_info:
        print("Movie metadata not found.")
        return

    print_banner(f"🎬 MOVIES SIMILAR TO: {target_info['title'].upper()}")
    print(f"  • Genres         : {', '.join(target_info['genres'])}")
    print(f"  • Community Score: {format_stars(target_info['avg_rating'])} ({target_info['avg_rating']:.2f}/5.0 from {target_info['rating_count']:,} reviews)")
    print("-" * 75)

    movie_neighbors = neighbors.get(target_mid, [])
    if not movie_neighbors:
        print("  Not enough collaborative ratings to compute reliable neighbors.")
        return

    for rank, (mid, sim_score) in enumerate(movie_neighbors[:top_n], 1):
        info = metadata.get(mid, {})
        title = info.get("title", f"Movie #{mid}")
        genres = ", ".join(info.get("genres", []))[:28]
        avg_r = info.get("avg_rating", 0.0)
        sim_bar = format_similarity_bar(sim_score)

        print(f"  {rank:>2}. {title:<42} {sim_bar}")
        print(f"      Genres: {genres:<28} | Rating: {format_stars(avg_r)} ({avg_r:.2f} ★)")


def recommend_by_genre(genre_query: str, model_artifact: dict, top_n: int = 10):
    metadata = model_artifact["movie_metadata"]
    g_lower = genre_query.strip().lower()

    matches = []
    for mid, info in metadata.items():
        movie_genres = [g.lower() for g in info["genres"]]
        if any(g_lower in g for g in movie_genres) and info["rating_count"] >= 25:
            matches.append((mid, info))

    matches.sort(key=lambda x: (x[1]["avg_rating"], x[1]["rating_count"]), reverse=True)

    print_banner(f"🌟 TOP RATED '{genre_query.upper()}' MOVIES (Min 25 Ratings)")
    if not matches:
        print(f"No movies found for genre '{genre_query}'.")
        return

    for rank, (mid, info) in enumerate(matches[:top_n], 1):
        genres = ", ".join(info["genres"])[:26]
        print(f"  {rank:>2}. {info['title']:<44} {format_stars(info['avg_rating'])} {info['avg_rating']:.2f} ★ ({info['rating_count']} reviews)")
        print(f"      Genres: {genres}")


def recommend_personalized(user_ratings_dict: dict, model_artifact: dict, top_n: int = 8):
    """
    Given a dict of {movieId: rating}, folds user into the SVD space
    and predicts ratings for all other movies in the catalog.
    """
    mu = model_artifact["mu"]
    Q = model_artifact["Q"]
    bi = model_artifact["bi"]
    m_map = model_artifact["m_map"]
    metadata = model_artifact["movie_metadata"]

    # Compute active user bias
    user_bias = np.mean(list(user_ratings_dict.values())) - mu

    # Solve for user latent vector p_u via ridge regression / gradient step
    rated_indices = [m_map[mid] for mid in user_ratings_dict if mid in m_map]
    if not rated_indices:
        print("None of the rated movies are in the eligible latent index.")
        return

    # User latent factor optimization
    k_factors = Q.shape[1]
    Q_rated = Q[rated_indices]
    r_residuals = np.array([
        user_ratings_dict[mid] - (mu + user_bias + bi[m_map[mid]])
        for mid in user_ratings_dict if mid in m_map
    ])

    reg = 0.05
    # Closed form: p_u = (Q^T Q + reg*I)^(-1) Q^T r
    p_u = np.linalg.solve(np.dot(Q_rated.T, Q_rated) + reg * np.eye(k_factors), np.dot(Q_rated.T, r_residuals))

    # Predict for all movies
    predictions = []
    rated_set = set(user_ratings_dict.keys())

    for mid, info in metadata.items():
        if mid in rated_set or mid not in m_map or info["rating_count"] < 10:
            continue
        i_idx = m_map[mid]
        pred_rating = mu + user_bias + bi[i_idx] + np.dot(p_u, Q[i_idx])
        pred_rating = np.clip(pred_rating, 0.5, 5.0)
        predictions.append((mid, pred_rating, info))

    predictions.sort(key=lambda x: x[1], reverse=True)

    print_banner("✨ PERSONALIZED RECOMMENDATIONS BASED ON YOUR TASTE ✨")
    for rank, (mid, score, info) in enumerate(predictions[:top_n], 1):
        genres = ", ".join(info["genres"])[:28]
        print(f"  {rank:>2}. {info['title']:<44} Predicted: {format_stars(score)} {score:.2f} ★")
        print(f"      Genres: {genres:<28} | Community: {info['avg_rating']:.2f} ★ ({info['rating_count']} reviews)")


def run_interactive(model_artifact):
    metadata = model_artifact["movie_metadata"]

    while True:
        print("\n" + "=" * 65)
        print(" 🎬 MOVIE RECOMMENDATION SYSTEM - MAIN MENU 🎬")
        print("=" * 65)
        print("  1) Find Movies Similar to a Title (Item-Based CF)")
        print("  2) Rate Movies & Get Personalized Recommendations (SVD)")
        print("  3) Explore Top Rated Movies by Genre")
        print("  4) Exit")
        print("-" * 65)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            query = input("\nEnter movie title (or part of it, e.g. 'Matrix', 'Inception'): ").strip()
            if not query:
                continue
            matches = find_movies_by_title(query, metadata, limit=5)
            if not matches:
                print(f"No movie found matching '{query}'.")
                continue

            print("\nSelect a movie from matches:")
            for idx, (mid, info) in enumerate(matches, 1):
                print(f"  {idx}) {info['title']} ({', '.join(info['genres'][:3])})")

            sel = input(f"Select (1-{len(matches)}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(matches):
                chosen_mid = matches[int(sel) - 1][0]
                recommend_similar_movies(chosen_mid, model_artifact)
            else:
                print("Invalid selection.")

        elif choice == "2":
            sample_popular = [
                (1, "Toy Story (1995)"),
                (296, "Pulp Fiction (1994)"),
                (2571, "The Matrix (1999)"),
                (260, "Star Wars: Episode IV (1977)"),
                (356, "Forrest Gump (1994)"),
                (593, "The Silence of the Lambs (1991)")
            ]
            print("\nRate any of the following movies from 1.0 to 5.0 (Press Enter to skip):")
            user_ratings = {}
            for mid, title in sample_popular:
                score_str = input(f"  Rate '{title}': ").strip()
                if score_str:
                    try:
                        val = float(score_str)
                        if 0.5 <= val <= 5.0:
                            user_ratings[mid] = val
                    except ValueError:
                        pass

            if not user_ratings:
                print("No ratings entered. Try giving at least 1-2 ratings.")
            else:
                recommend_personalized(user_ratings, model_artifact)

        elif choice == "3":
            print("\nPopular genres: Action, Comedy, Drama, Sci-Fi, Animation, Thriller, Romance, Horror")
            genre = input("Enter genre to explore: ").strip()
            if genre:
                recommend_by_genre(genre, model_artifact)

        elif choice == "4":
            print("Enjoy your movies! Exiting recommender.\n")
            break
        else:
            print("Invalid option. Please enter 1 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation Engine using Collaborative Filtering.")
    parser.add_argument("--movie", type=str, help="Search a movie title and get similar recommendations")
    parser.add_argument("--genre", type=str, help="Get top rated movies in a given genre")

    args = parser.parse_args()
    model_artifact = load_model()

    if args.movie:
        matches = find_movies_by_title(args.movie, model_artifact["movie_metadata"], limit=1)
        if matches:
            chosen_mid = matches[0][0]
            recommend_similar_movies(chosen_mid, model_artifact)
        else:
            print(f"No movie found matching '{args.movie}'.")
    elif args.genre:
        recommend_by_genre(args.genre, model_artifact)
    else:
        run_interactive(model_artifact)


if __name__ == "__main__":
    main()
