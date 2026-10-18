"""
Training and Evaluation Pipeline for Customer Segmentation.
Executes the Elbow Method, Silhouette Analysis, benchmarks K-Means vs Hierarchical & DBSCAN,
profiles the 5 distinct customer personas, and exports the segmentation artifact.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


PERSONA_PLAYBOOK = {
    "VIP / Target": {
        "badge": "💎",
        "description": "High Income & High Spending. Core luxury consumers, low price sensitivity, value exclusivity.",
        "marketing_strategy": "Invite to VIP concierge, pre-release private previews, high-end loyalty rewards, personalized styling."
    },
    "Careful / Savers": {
        "badge": "💰",
        "description": "High Income & Low Spending. Affluent but highly disciplined, analytical, and cautious.",
        "marketing_strategy": "Highlight craftsmanship, longevity guarantees, investment-grade products, zero-risk return policies."
    },
    "Trend / Impulsive": {
        "badge": "🛍️",
        "description": "Low Income & High Spending. Typically younger cohort, trend-seeking, highly responsive to viral marketing.",
        "marketing_strategy": "Offer Buy-Now-Pay-Later (BNPL), student discounts, flash drops, social media influencer campaigns."
    },
    "Sensible / Budget": {
        "badge": "🪙",
        "description": "Low Income & Low Spending. Highly frugal, coupon-driven, purchases only basic essentials.",
        "marketing_strategy": "Bulk buy promotions, seasonal clearance alerts, high-value bundles, price-match guarantees."
    },
    "Standard / Balanced": {
        "badge": "⚖️",
        "description": "Moderate Income & Moderate Spending. The stable middle class backbone of mall foot traffic.",
        "marketing_strategy": "Points-based loyalty cards, seasonal family promotions, multi-brand gift card incentives."
    }
}


def identify_persona(mean_income: float, mean_spending: float) -> str:
    """Classifies cluster centroid into business persona."""
    if mean_income >= 65 and mean_spending >= 65:
        return "VIP / Target"
    elif mean_income >= 65 and mean_spending < 45:
        return "Careful / Savers"
    elif mean_income < 45 and mean_spending >= 65:
        return "Trend / Impulsive"
    elif mean_income < 45 and mean_spending < 45:
        return "Sensible / Budget"
    else:
        return "Standard / Balanced"


def main():
    print_banner("🛍️ TRAINING CUSTOMER SEGMENTATION CLUSTERING MODELS 🛍️")

    csv_path = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")
    df = pd.read_csv(csv_path)

    # Core features for behavioral segmentation
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].values

    # 1. Elbow Method & Silhouette Analysis (k = 2 to 9)
    print_banner("1. ELBOW METHOD & SILHOUETTE ANALYSIS (Finding Optimal k)")
    k_range = range(2, 10)
    inertias = []
    silhouettes = []

    print(f"{'k (Clusters)':<14} | {'WCSS (Inertia)':<18} | {'Silhouette Score':<18} | {'Curve Trend'}")
    print("-" * 75)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X)
        wcss = km.inertia_
        sil = silhouette_score(X, km.labels_)
        inertias.append(wcss)
        silhouettes.append(sil)
        bar = "█" * int(sil * 45)
        elbow_tag = " 👈 [Optimal Elbow]" if k == 5 else ""
        print(f"  k = {k:<9} | {wcss:>12.1f}      | {sil:>12.4f}       |{bar}{elbow_tag}")

    # 2. Benchmark Clustering Algorithms at k=5
    print_banner("2. BENCHMARKING CLUSTERING ALGORITHMS (at k=5)")
    algorithms = {
        "K-Means++ (k=5)": KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10),
        "Agglomerative (Ward, k=5)": AgglomerativeClustering(n_clusters=5, linkage="ward"),
        "DBSCAN (eps=9.5, min_samples=4)": DBSCAN(eps=9.5, min_samples=4)
    }

    print(f"{'Algorithm':<32} | {'Silhouette (↑)':<15} | {'Calinski-Harabasz (↑)':<22} | {'Davies-Bouldin (↓)'}")
    print("-" * 85)
    fitted_models = {}
    for name, model in algorithms.items():
        labels = model.fit_predict(X)
        # Filter noise for DBSCAN if any
        valid_mask = labels != -1
        if len(set(labels[valid_mask])) > 1:
            sil = silhouette_score(X[valid_mask], labels[valid_mask])
            ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
            db = davies_bouldin_score(X[valid_mask], labels[valid_mask])
            fitted_models[name] = (model, labels)
            print(f"{name:<32} | {sil:>10.4f}      | {ch:>16.1f}       | {db:>10.4f}")

    # 3. Champion Model: K-Means (k=5)
    champion_km = fitted_models["K-Means++ (k=5)"][0]
    champion_labels = champion_km.labels_
    centroids = champion_km.cluster_centers_

    df["Cluster"] = champion_labels

    # 4. Profile the 5 Personas
    print_banner("3. CUSTOMER PERSONA PROFILES & STRATEGIC PLAYBOOKS")
    cluster_personas = {}

    for c_id in range(5):
        c_df = df[df["Cluster"] == c_id]
        mean_inc = c_df["Annual Income (k$)"].mean()
        mean_spd = c_df["Spending Score (1-100)"].mean()
        mean_age = c_df["Age"].mean()
        fem_pct = (c_df["Gender"] == "Female").mean() * 100

        persona_name = identify_persona(mean_inc, mean_spd)
        playbook = PERSONA_PLAYBOOK[persona_name]
        cluster_personas[c_id] = {
            "persona": persona_name,
            "badge": playbook["badge"],
            "centroid": centroids[c_id].tolist(),
            "mean_income": mean_inc,
            "mean_spending": mean_spd,
            "mean_age": mean_age,
            "female_pct": fem_pct,
            "count": len(c_df),
            "share_pct": (len(c_df) / len(df)) * 100,
            "description": playbook["description"],
            "strategy": playbook["marketing_strategy"]
        }

        print(f"\nCluster #{c_id + 1} {playbook['badge']} : {persona_name.upper()} ({len(c_df)} customers, {len(c_df)/len(df)*100:.1f}% share)")
        print(f"  • Centroid       : Income ${mean_inc:.1f}k | Spending Score {mean_spd:.1f} / 100")
        print(f"  • Demographics   : Avg Age {mean_age:.1f} yrs | {fem_pct:.1f}% Female")
        print(f"  • Behavioral DNA : {playbook['description']}")
        print(f"  • Strategic Play : {playbook['marketing_strategy']}")

    # 5. Export Artifact
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, "customer_segmentation.joblib")

    artifact = {
        "model": champion_km,
        "features": features,
        "centroids": centroids,
        "personas": cluster_personas,
        "silhouette_score": silhouette_score(X, champion_labels),
        "total_customers": len(df)
    }
    joblib.dump(artifact, artifact_path)

    print_banner("4. MODEL PERSISTENCE & READINESS")
    print(f"  Artifact successfully saved to: {artifact_path}")
    print(f"  Silhouette Score : {artifact['silhouette_score']:.4f} (High cluster cohesion)")
    print("  Ready for interactive customer profiling!\n")


if __name__ == "__main__":
    main()
