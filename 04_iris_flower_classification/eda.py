"""
Exploratory Data Analysis (EDA) for Iris Flower Classification.
Analyzes feature distributions, correlations, and class separability across Iris species.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def print_banner(text: str, char: str = "=", width: int = 70):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def main():
    print_banner("🌸 PROJECT 04: IRIS DATASET EXPLORATORY DATA ANALYSIS 🌸")

    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # Rename target column for clarity
    df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    print(f"Dataset Shape: {df.shape[0]} samples, {len(feature_names)} features, 1 target")
    print(f"Target Species: {', '.join(target_names)}")

    # 1. Class Distribution
    print_banner("1. CLASS BALANCE")
    counts = df["species"].value_counts()
    for species, count in counts.items():
        bar = "█" * int(count / 2)
        print(f"  {species:<12} : {count:>3} samples (100.0% balanced)  |{bar}")

    # 2. Summary Statistics by Feature & Species
    print_banner("2. MEASUREMENT STATISTICS BY SPECIES (Mean ± Std)")
    print(f"{'Feature':<20} | {'Setosa':<14} | {'Versicolor':<14} | {'Virginica':<14}")
    print("-" * 70)
    for feat in feature_names:
        stats = []
        for sp in target_names:
            sub = df[df["species"] == sp][feat]
            stats.append(f"{sub.mean():.2f} ± {sub.std():.2f}")
        print(f"{feat:<20} | {stats[0]:<14} | {stats[1]:<14} | {stats[2]:<14}")

    # 3. ASCII Distribution of Petal Dimensions (Key Separator)
    print_banner("3. PETAL LENGTH DISTRIBUTION (cm) - KEY DISCRIMINATOR")
    print("Ranges from ~1.0cm to ~6.9cm:\n")

    bins = [(1.0, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 6.0), (6.0, 7.0)]
    for low, high in bins:
        label = f"[{low:.1f} - {high:.1f} cm]"
        setosa_cnt = len(df[(df["species"] == "setosa") & (df["petal length (cm)"] >= low) & (df["petal length (cm)"] < high)])
        versi_cnt  = len(df[(df["species"] == "versicolor") & (df["petal length (cm)"] >= low) & (df["petal length (cm)"] < high)])
        virg_cnt   = len(df[(df["species"] == "virginica") & (df["petal length (cm)"] >= low) & (df["petal length (cm)"] <= high)])

        bar_s = "S" * setosa_cnt
        bar_v = "V" * versi_cnt
        bar_g = "G" * virg_cnt
        visual = f"{bar_s}{bar_v}{bar_g}"
        print(f"  {label:<15} | {visual:<45} (S:{setosa_cnt:2d}, V:{versi_cnt:2d}, G:{virg_cnt:2d})")

    print("\n  Legend: S = Setosa (1.0-1.9cm), V = Versicolor (3.0-5.1cm), G = Virginica (4.5-6.9cm)")

    # 4. Correlation Matrix
    print_banner("4. FEATURE CORRELATION MATRIX")
    corr = df[feature_names].corr()
    header = f"{'Feature':<22}" + "".join([f"{f[:8]:>11}" for f in feature_names])
    print(header)
    print("-" * len(header))
    for feat in feature_names:
        row_str = f"{feat:<22}" + "".join([f"{corr.loc[feat, c]:>11.3f}" for c in feature_names])
        print(row_str)

    # 5. Key Takeaways
    print_banner("5. KEY OBSERVATIONS & ML IMPLICATIONS")
    print("  • Petal Length & Petal Width exhibit high positive correlation (+0.963).")
    print("  • Iris Setosa has distinctively tiny petals (length < 2.0cm, width < 0.6cm) and is")
    print("    100% linearly separable from the other two species with zero error.")
    print("  • Iris Versicolor and Iris Virginica overlap slightly in the 4.5 - 5.1cm petal zone,")
    print("    meaning algorithms that form non-linear or soft-margin boundaries (e.g. SVM, Random Forest)")
    print("    will excel at resolving the remaining edge cases.")


if __name__ == "__main__":
    main()
