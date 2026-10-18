"""
Generator for Michigan Real Estate & House Price Dataset.
Synthesizes 5,000 realistic residential records based on Michigan regional market trends:
- Top cities: Ann Arbor, Birmingham, Detroit, Grand Rapids, Lansing, Troy,
  Traverse City, Royal Oak, Kalamazoo, Rochester Hills.
- Features: Square footage, bedrooms, bathrooms, lot size, year built, garage,
  basement (Michigan standard), lake access (Great Lakes perk), school rating.
"""

import os
import numpy as np
import pandas as pd


# Real-world approximate baseline price per sqft & school rating benchmarks for MI regions
CITY_PROFILES = {
    "Ann Arbor": {"base_ppsf": 290, "avg_schools": 8.5, "lot_scale": 9000, "base_cost": 75000},
    "Birmingham": {"base_ppsf": 360, "avg_schools": 9.2, "lot_scale": 11000, "base_cost": 150000},
    "Northville": {"base_ppsf": 305, "avg_schools": 9.6, "lot_scale": 13000, "base_cost": 95000},
    "Novi": {"base_ppsf": 265, "avg_schools": 9.2, "lot_scale": 11500, "base_cost": 80000},
    "Detroit": {"base_ppsf": 110, "avg_schools": 4.2, "lot_scale": 6000, "base_cost": 30000},
    "Grand Rapids": {"base_ppsf": 210, "avg_schools": 7.4, "lot_scale": 10000, "base_cost": 50000},
    "Lansing": {"base_ppsf": 135, "avg_schools": 5.8, "lot_scale": 8500, "base_cost": 40000},
    "Troy": {"base_ppsf": 235, "avg_schools": 8.8, "lot_scale": 12000, "base_cost": 70000},
    "Traverse City": {"base_ppsf": 260, "avg_schools": 7.8, "lot_scale": 20000, "base_cost": 80000},
    "Royal Oak": {"base_ppsf": 255, "avg_schools": 7.9, "lot_scale": 7000, "base_cost": 65000},
    "Kalamazoo": {"base_ppsf": 145, "avg_schools": 6.2, "lot_scale": 10500, "base_cost": 42000},
    "Rochester Hills": {"base_ppsf": 240, "avg_schools": 8.9, "lot_scale": 14000, "base_cost": 75000},
}


def generate_dataset(num_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    cities = list(CITY_PROFILES.keys())

    # Assign city distribution with realistic market volume
    city_weights = [0.10, 0.07, 0.06, 0.06, 0.14, 0.12, 0.09, 0.09, 0.07, 0.08, 0.06, 0.06]
    assigned_cities = np.random.choice(cities, size=num_samples, p=city_weights)

    records = []

    for city in assigned_cities:
        profile = CITY_PROFILES[city]

        # Home specs
        sqft = int(np.random.normal(loc=1950, scale=650))
        sqft = max(750, min(5500, sqft))

        # Bedrooms & Bathrooms tied naturally to sqft
        if sqft < 1100:
            beds = np.random.choice([1, 2], p=[0.3, 0.7])
            baths = 1.0
        elif sqft < 1800:
            beds = np.random.choice([2, 3], p=[0.3, 0.7])
            baths = np.random.choice([1.0, 1.5, 2.0], p=[0.2, 0.3, 0.5])
        elif sqft < 2800:
            beds = np.random.choice([3, 4], p=[0.45, 0.55])
            baths = np.random.choice([2.0, 2.5, 3.0], p=[0.3, 0.5, 0.2])
        else:
            beds = np.random.choice([4, 5, 6], p=[0.5, 0.35, 0.15])
            baths = np.random.choice([2.5, 3.0, 3.5, 4.0, 4.5], p=[0.1, 0.3, 0.3, 0.2, 0.1])

        # Lot size in sqft
        lot_size = int(np.random.gamma(shape=3.5, scale=profile["lot_scale"] / 3.5))
        lot_size = max(3500, min(80000, lot_size))

        # Year built (1920 to 2024)
        year_built = int(np.random.choice(
            [
                np.random.randint(1920, 1960),
                np.random.randint(1960, 1995),
                np.random.randint(1995, 2024)
            ],
            p=[0.25, 0.40, 0.35]
        ))

        # Garage (0 to 3 cars)
        garage_spaces = int(np.random.choice([0, 1, 2, 3], p=[0.08, 0.22, 0.55, 0.15]))

        # Basement: ~88% of Michigan homes have basements due to frost line
        has_basement = int(np.random.choice([1, 0], p=[0.88, 0.12]))

        # Lake access (Michigan lake privilege / frontage)
        lake_prob = 0.20 if city == "Traverse City" else 0.05
        has_lake_access = int(np.random.choice([1, 0], p=[lake_prob, 1 - lake_prob]))

        # School rating (1 to 10)
        school_rating = float(np.clip(
            np.random.normal(loc=profile["avg_schools"], scale=0.8), 1.0, 10.0
        ))
        school_rating = round(school_rating, 1)

        # Realistic Price Calculation
        age = 2026 - year_built
        age_depreciation = max(-0.15, - (age * 0.0018))  # vintage homes stabilize
        renovation_bonus = 0.08 if (age > 40 and np.random.rand() > 0.6) else 0.0

        price = (
            profile["base_cost"]
            + (sqft * profile["base_ppsf"])
            + (lot_size * 2.8)
            + (beds * 8000)
            + (baths * 15000)
            + (garage_spaces * 12000)
            + (has_basement * 28000)
            + (has_lake_access * 75000)
            + ((school_rating - 5.0) * 16000)
        )

        price *= (1.0 + age_depreciation + renovation_bonus)
        # Market stochastic fluctuation (± 7%)
        noise = np.random.normal(1.0, 0.065)
        price = round(float(price * noise), -2)  # round to nearest $100
        price = max(45000.0, price)

        records.append({
            "city": city,
            "square_feet": sqft,
            "bedrooms": beds,
            "bathrooms": baths,
            "lot_size_sqft": lot_size,
            "year_built": year_built,
            "garage_spaces": garage_spaces,
            "has_basement": has_basement,
            "has_lake_access": has_lake_access,
            "school_rating": school_rating,
            "price": price
        })

    return pd.DataFrame(records)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_file = os.path.join(data_dir, "michigan_housing.csv")

    print(f"Generating 5,000 Michigan housing transaction records...")
    df = generate_dataset(5000)
    df.to_csv(out_file, index=False)
    print(f"Saved dataset to: {out_file}")
    print("\nDataset Preview:")
    print(df.head(5).to_string())
    print("\nAverage Price by Michigan City:")
    print(df.groupby("city")["price"].mean().sort_values(ascending=False).map("${:,.0f}".format))


if __name__ == "__main__":
    main()
