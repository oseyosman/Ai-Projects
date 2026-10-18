"""
Exploratory Data Analysis (EDA) for Michigan House Price Prediction.
"""

import os
import pandas as pd
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "michigan_housing.csv")


def run_eda():
    print("=" * 68)
    print(" 🌲 Michigan Real Estate & House Prices - Exploratory Analysis")
    print("=" * 68)

    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Generating now...")
        import generate_michigan_data
        generate_michigan_data.main()

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1/5] Total Properties Analyzed: {len(df):,}")
    print(f"Features: {list(df.columns)}")

    # 1. Price overview
    print("\n[2/5] Michigan Housing Price Statistics:")
    print(f"  Mean Price:    ${df['price'].mean():,.2f}")
    print(f"  Median Price:  ${df['price'].median():,.2f}")
    print(f"  Min Price:     ${df['price'].min():,.2f}")
    print(f"  Max Price:     ${df['price'].max():,.2f}")
    print(f"  Std Deviation: ${df['price'].std():,.2f}")

    # 2. City-by-City Breakdown
    print("\n[3/5] Average Price & Price-per-SqFt by Michigan Metro / City:")
    df["price_per_sqft"] = df["price"] / df["square_feet"]
    city_stats = df.groupby("city").agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        avg_ppsf=("price_per_sqft", "mean"),
        count=("price", "count")
    ).sort_values(by="avg_price", ascending=False)

    for city, row in city_stats.iterrows():
        print(f"  {city:<16} | Mean: ${row['avg_price']:>9,.0f} | Median: ${row['median_price']:>9,.0f} | ${row['avg_ppsf']:>5.0f}/sqft ({int(row['count'])} sales)")

    # 3. Michigan-Specific Features Impact
    print("\n[4/5] Michigan Specific Amenities & Impact on Valuation:")
    # Basement impact
    basement_diff = df.groupby("has_basement")["price"].mean()
    if 1 in basement_diff and 0 in basement_diff:
        b_prem = basement_diff[1] - basement_diff[0]
        print(f"  • Full Basement vs Slab: +${b_prem:,.0f} premium")

    # Lake Access
    lake_diff = df.groupby("has_lake_access")["price"].mean()
    if 1 in lake_diff and 0 in lake_diff:
        lake_prem = lake_diff[1] - lake_diff[0]
        print(f"  • Lake Access / Frontage: +${lake_prem:,.0f} premium")

    # School Rating impact
    corr_school = df["school_rating"].corr(df["price"])
    print(f"  • School Rating Correlation: {corr_school:+.3f}")

    # 4. Overall Numerical Correlations
    print("\n[5/5] Feature Correlation with Home Price:")
    num_cols = df.select_dtypes(include=[np.number]).drop(columns=["price", "price_per_sqft"], errors="ignore")
    corrs = num_cols.apply(lambda col: col.corr(df["price"])).sort_values(ascending=False)

    for feat, corr in corrs.items():
        bar = "█" * int(abs(corr) * 25)
        sign = "+" if corr >= 0 else "-"
        print(f"  {feat:<16} : {corr:+.4f} | {bar} ({sign})")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    run_eda()
