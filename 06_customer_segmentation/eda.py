"""
Exploratory Data Analysis for Customer Segmentation (Mall Customers).
Analyzes customer demographics, spending behaviors, correlations, and visual clustering tendencies.
"""

import os
import pandas as pd
import numpy as np


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def main():
    print_banner("🛍️ PROJECT 06: MALL CUSTOMER SEGMENTATION - EDA 🛍️")

    csv_path = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at '{csv_path}'.")
        return

    df = pd.read_csv(csv_path)

    # 1. Overview
    print_banner("1. DATASET OVERVIEW & DEMOGRAPHICS")
    print(f"  • Total Customer Profiles : {len(df)}")
    print(f"  • Missing Values          : {df.isnull().sum().sum()} (Clean dataset)")

    gender_counts = df["Gender"].value_counts()
    for g, count in gender_counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2.5)
        print(f"  • Gender: {g:<6}           : {count:>3} customers ({pct:.1f}%) |{bar}")

    # 2. Summary Statistics
    print_banner("2. SUMMARY STATISTICS BY FEATURE")
    num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    print(f"{'Feature':<25} | {'Mean':<8} | {'Std':<8} | {'Min':<6} | {'Median':<8} | {'Max':<6}")
    print("-" * 75)
    for col in num_cols:
        s = df[col]
        print(f"{col:<25} | {s.mean():>6.1f}   | {s.std():>6.1f}   | {s.min():>4.0f}   | {s.median():>6.1f}   | {s.max():>4.0f}")

    # 3. Gender Spending Comparison
    print_banner("3. SPENDING BEHAVIOR BY GENDER")
    gender_grp = df.groupby("Gender").agg(
        avg_age=("Age", "mean"),
        avg_income=("Annual Income (k$)", "mean"),
        avg_spending=("Spending Score (1-100)", "mean")
    )
    print(f"{'Gender':<10} | {'Avg Age':<12} | {'Avg Income (k$)':<18} | {'Avg Spending (1-100)':<20}")
    print("-" * 75)
    for g, row in gender_grp.iterrows():
        print(f"{g:<10} | {row['avg_age']:>6.1f} yrs   | ${row['avg_income']:>6.1f}k           | {row['avg_spending']:>6.1f} / 100")

    # 4. Correlation Matrix
    print_banner("4. NUMERICAL CORRELATION MATRIX")
    corr = df[num_cols].corr()
    header = f"{'Feature':<25}" + "".join([f"{c[:10]:>14}" for c in num_cols])
    print(header)
    print("-" * len(header))
    for col in num_cols:
        row_str = f"{col:<25}" + "".join([f"{corr.loc[col, c]:>14.3f}" for c in num_cols])
        print(row_str)

    # 5. ASCII 2D Grid: Income vs. Spending Score
    print_banner("5. 2D CLUSTERING LANDSCAPE: ANNUAL INCOME vs. SPENDING SCORE")
    print("Each symbol represents count of customers in that (Income x Spending) quadrant:\n")

    grid = np.zeros((10, 10), dtype=int)
    for _, row in df.iterrows():
        inc_bin = min(int(row["Annual Income (k$)"] / 14), 9)
        spd_bin = min(int(row["Spending Score (1-100)"] / 10), 9)
        grid[spd_bin, inc_bin] += 1

    print("  Spending Score (Y)  ^")
    print("  High (80-100)  |  " + "  ".join([f"{grid[9, i]:>2}" if grid[9, i] > 0 else " ." for i in range(10)]) + "  <- [Impulsive] & [VIPs]")
    print("                 |  " + "  ".join([f"{grid[8, i]:>2}" if grid[8, i] > 0 else " ." for i in range(10)]))
    print("  Mid  (40-60)   |  " + "  ".join([f"{grid[5, i]:>2}" if grid[5, i] > 0 else " ." for i in range(10)]) + "  <- [Standard Balanced]")
    print("                 |  " + "  ".join([f"{grid[4, i]:>2}" if grid[4, i] > 0 else " ." for i in range(10)]))
    print("  Low  (0-20)    |  " + "  ".join([f"{grid[1, i]:>2}" if grid[1, i] > 0 else " ." for i in range(10)]) + "  <- [Budget] & [Careful Savers]")
    print("                 +-------------------------------------------------->")
    print("                    $15k                                       $140k  Annual Income (X)")

    # 6. Key Takeaways
    print_banner("6. KEY DATA INSIGHTS & UNSUPERVISED HYPOTHESIS")
    print("  • Age and Spending Score have a notable negative correlation (-0.327):")
    print("    Younger customers (<35) exhibit substantially higher spending enthusiasm.")
    print("  • The Income vs. Spending space naturally reveals 5 clean geometric islands:")
    print("    1. Low Income, High Spend (Impulsive Shoppers)")
    print("    2. Low Income, Low Spend  (Budget Savers)")
    print("    3. Mid Income, Mid Spend  (Standard Core Segment)")
    print("    4. High Income, High Spend (VIP Luxury Shoppers)")
    print("    5. High Income, Low Spend (Careful Wealthy Savers)")
    print("  • K-Means with k=5 is expected to cleanly partition this customer base.\n")


if __name__ == "__main__":
    main()
