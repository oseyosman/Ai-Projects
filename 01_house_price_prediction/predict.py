"""
Interactive Michigan House Price Predictor.
Uses trained Scikit-learn model to estimate property values across Michigan markets.
"""

import os
import argparse
import joblib
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_michigan_house_model.joblib")

PRESET_PROPERTIES = {
    "1": {
        "title": "Ann Arbor Family Home (Near UM Campus / Top Schools)",
        "city": "Ann Arbor",
        "square_feet": 2400,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "lot_size_sqft": 9500,
        "year_built": 1998,
        "garage_spaces": 2,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 9.2
    },
    "2": {
        "title": "Birmingham Luxury Estate (Oakland County)",
        "city": "Birmingham",
        "square_feet": 3800,
        "bedrooms": 5,
        "bathrooms": 4.0,
        "lot_size_sqft": 14000,
        "year_built": 2018,
        "garage_spaces": 3,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 9.8
    },
    "3": {
        "title": "Detroit Historic Brick Bungalow",
        "city": "Detroit",
        "square_feet": 1500,
        "bedrooms": 3,
        "bathrooms": 1.5,
        "lot_size_sqft": 5200,
        "year_built": 1948,
        "garage_spaces": 1,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 4.5
    },
    "4": {
        "title": "Traverse City Lakefront Cottage",
        "city": "Traverse City",
        "square_feet": 2100,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "lot_size_sqft": 25000,
        "year_built": 1985,
        "garage_spaces": 2,
        "has_basement": 1,
        "has_lake_access": 1,
        "school_rating": 7.8
    },
    "5": {
        "title": "Grand Rapids Suburban Contemporary",
        "city": "Grand Rapids",
        "square_feet": 2050,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "lot_size_sqft": 11000,
        "year_built": 2005,
        "garage_spaces": 2,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 7.5
    },
    "6": {
        "title": "Northville Historic Downtown / Luxury Sub",
        "city": "Northville",
        "square_feet": 2800,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "lot_size_sqft": 13500,
        "year_built": 2012,
        "garage_spaces": 3,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 9.6
    },
    "7": {
        "title": "Novi Meadowbrook Lake Area Colonial",
        "city": "Novi",
        "square_feet": 2550,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "lot_size_sqft": 11500,
        "year_built": 2008,
        "garage_spaces": 2,
        "has_basement": 1,
        "has_lake_access": 0,
        "school_rating": 9.3
    }
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please run: python3 train.py first to train and benchmark models."
        )
    return joblib.load(MODEL_PATH)


def estimate_price(artifact, features: dict):
    pipeline = artifact["pipeline"]
    df_input = pd.DataFrame([features])
    predicted_val = pipeline.predict(df_input)[0]
    mae = artifact["metrics"]["mae"]

    low_bound = max(30000, predicted_val - mae)
    high_bound = predicted_val + mae
    ppsf = predicted_val / features["square_feet"]

    return predicted_val, low_bound, high_bound, ppsf


def interactive_menu(artifact):
    print("\n" + "=" * 68)
    print(" 🌲 Michigan House Price Predictor")
    print(f" Loaded Model: {artifact['model_name']} | Accuracy R²: {artifact['metrics']['r2']:.3f}")
    print("=" * 68)

    print("\nSelect an option to test:")
    for key, p in PRESET_PROPERTIES.items():
        print(f" [{key}] {p['title']} ({p['city']})")
    print(f" [{len(PRESET_PROPERTIES) + 1}] Enter custom property specifications")
    print(" [q] Quit")

    choice = input(f"\nYour choice [1-{len(PRESET_PROPERTIES) + 1}, q]: ").strip()

    if choice in PRESET_PROPERTIES:
        preset = PRESET_PROPERTIES[choice]
        print(f"\nEvaluating: {preset['title']}")
        print(f"• City: {preset['city']} | Size: {preset['square_feet']:,} sqft | {preset['bedrooms']} bed / {preset['bathrooms']} bath")
        print(f"• Built: {preset['year_built']} | Lot: {preset['lot_size_sqft']:,} sqft | School Rating: {preset['school_rating']}/10")
        print(f"• Full Basement: {'Yes' if preset['has_basement'] else 'No'} | Lake Access: {'Yes' if preset['has_lake_access'] else 'No'}")

        features = {k: v for k, v in preset.items() if k != "title"}
        pred, low, high, ppsf = estimate_price(artifact, features)

        print("\n" + "-" * 56)
        print(f" 💵 ESTIMATED MARKET VALUE:  ${pred:,.0f}")
        print(f" 📊 Expected Range (±MAE):    ${low:,.0f} - ${high:,.0f}")
        print(f" 📐 Price per Square Foot:    ${ppsf:,.0f} / sqft")
        print("-" * 56)

    elif choice == str(len(PRESET_PROPERTIES) + 1):
        cities = artifact["known_cities"]
        print("\nSelect Michigan City:")
        for i, c in enumerate(cities, 1):
            print(f"  {i}. {c}")
        city_idx = input(f"Choose city [1-{len(cities)}]: ").strip()
        try:
            city = cities[int(city_idx) - 1]
        except (ValueError, IndexError):
            city = "Ann Arbor"
        print(f"Selected: {city}")

        try:
            sqft = int(input("Square Feet (e.g., 2100): ") or 2100)
            beds = int(input("Bedrooms (e.g., 3): ") or 3)
            baths = float(input("Bathrooms (e.g., 2.5): ") or 2.5)
            lot = int(input("Lot Size in sqft (e.g., 9000): ") or 9000)
            year = int(input("Year Built (e.g., 1995): ") or 1995)
            garage = int(input("Garage Spaces [0-3] (e.g., 2): ") or 2)
            basement = 1 if input("Has Basement? [Y/n]: ").strip().lower() != 'n' else 0
            lake = 1 if input("Has Lake Access / Frontage? [y/N]: ").strip().lower() == 'y' else 0
            schools = float(input("School District Rating [1-10] (e.g., 8.0): ") or 8.0)

            features = {
                "city": city,
                "square_feet": sqft,
                "bedrooms": beds,
                "bathrooms": baths,
                "lot_size_sqft": lot,
                "year_built": year,
                "garage_spaces": garage,
                "has_basement": basement,
                "has_lake_access": lake,
                "school_rating": schools
            }
            pred, low, high, ppsf = estimate_price(artifact, features)

            print("\n" + "-" * 56)
            print(f" 💵 ESTIMATED MARKET VALUE:  ${pred:,.0f}")
            print(f" 📊 Expected Range (±MAE):    ${low:,.0f} - ${high:,.0f}")
            print(f" 📐 Price per Square Foot:    ${ppsf:,.0f} / sqft")
            print("-" * 56)
        except ValueError as e:
            print(f"Invalid input: {e}")
    elif choice.lower() == "q":
        print("Done!")
    else:
        print("Option not recognized.")


def main():
    parser = argparse.ArgumentParser(description="Estimate Michigan House Prices")
    parser.add_argument("--city", type=str, help="Michigan City name (e.g., 'Ann Arbor')")
    parser.add_argument("--sqft", type=int, help="Square footage (e.g., 2400)")
    parser.add_argument("--beds", type=int, default=3, help="Bedrooms")
    parser.add_argument("--baths", type=float, default=2.0, help="Bathrooms")
    parser.add_argument("--lot", type=int, default=8500, help="Lot size in sqft")
    parser.add_argument("--year", type=int, default=2000, help="Year built")
    parser.add_argument("--garage", type=int, default=2, help="Garage capacity")
    parser.add_argument("--basement", type=int, default=1, help="Has basement (1 or 0)")
    parser.add_argument("--lake", type=int, default=0, help="Has lake access (1 or 0)")
    parser.add_argument("--school", type=float, default=8.0, help="School rating (1-10)")

    args = parser.parse_args()
    artifact = load_model()

    if args.city and args.sqft:
        features = {
            "city": args.city,
            "square_feet": args.sqft,
            "bedrooms": args.beds,
            "bathrooms": args.baths,
            "lot_size_sqft": args.lot,
            "year_built": args.year,
            "garage_spaces": args.garage,
            "has_basement": args.basement,
            "has_lake_access": args.lake,
            "school_rating": args.school
        }
        pred, low, high, ppsf = estimate_price(artifact, features)
        print("=" * 56)
        print(f" Property: {args.city} | {args.sqft:,} sqft | {args.beds} bed / {args.baths} bath")
        print(f" Estimated Price:       ${pred:,.0f}")
        print(f" Value Range:           ${low:,.0f} - ${high:,.0f}")
        print(f" Price per SqFt:        ${ppsf:,.0f} / sqft")
        print("=" * 56)
    else:
        interactive_menu(artifact)


if __name__ == "__main__":
    main()
