"""
Interactive Passenger Survival Predictor & Kaggle Batch Submission Generator.
Supports direct CLI flags, interactive passenger simulation wizard,
historical passenger recreations, and Kaggle test set batch prediction.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Ensure local module import works from any execution directory
sys.path.insert(0, os.path.dirname(__file__))
from features import (
    engineer_features,
    build_passenger_record,
    extract_title_from_name,
    FEATURE_COLUMNS,
)


MODEL_FILE = os.path.join(os.path.dirname(__file__), "models", "titanic_classifier.joblib")
TEST_CSV_FILE = os.path.join(os.path.dirname(__file__), "data", "test.csv")
DEFAULT_SUBMISSION_FILE = os.path.join(os.path.dirname(__file__), "data", "submission.csv")


HISTORICAL_PASSENGERS = [
    {
        "name": "Jack Dawson (3rd Class Steerage Artist)",
        "pclass": 3,
        "sex": "male",
        "age": 20.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S",
        "title": "Mr",
        "cabin": None,
        "historical_note": "Fictional steerage traveler who won his ticket in a poker game. Perished in the freezing North Atlantic waters.",
        "expected": "Perished"
    },
    {
        "name": "Rose DeWitt Bukater (1st Class Socialite)",
        "pclass": 1,
        "sex": "female",
        "age": 17.0,
        "sibsp": 0,
        "parch": 1,
        "fare": 150.0,
        "embarked": "S",
        "title": "Miss",
        "cabin": "B52",
        "historical_note": "1st class upper-deck cabin with immediate access to starboard lifeboats. Boarded collapsible boat.",
        "expected": "Survived"
    },
    {
        "name": "Margaret 'The Unsinkable Molly' Brown",
        "pclass": 1,
        "sex": "female",
        "age": 44.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 27.72,
        "embarked": "C",
        "title": "Mrs",
        "cabin": "C101",
        "historical_note": "American socialite and activist who famously urged Lifeboat No. 6 to turn back to look for survivors.",
        "expected": "Survived"
    },
    {
        "name": "Master Thomas Becker (2nd Class Child)",
        "pclass": 2,
        "sex": "male",
        "age": 4.0,
        "sibsp": 1,
        "parch": 1,
        "fare": 26.25,
        "embarked": "S",
        "title": "Master",
        "cabin": None,
        "historical_note": "Young boy traveling with his mother and sister. Benefited from 'women and children first' maritime protocol.",
        "expected": "Survived"
    },
    {
        "name": "Captain Edward J. Smith (Ship Commander)",
        "pclass": 1,
        "sex": "male",
        "age": 62.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 0.0,
        "embarked": "S",
        "title": "Officer",
        "cabin": "A1",
        "historical_note": "Veteran White Star Line captain. Remained on duty and perished with his ship as maritime tradition dictated.",
        "expected": "Perished"
    },
    {
        "name": "Anders Johan Andersson (3rd Class Father)",
        "pclass": 3,
        "sex": "male",
        "age": 39.0,
        "sibsp": 1,
        "parch": 4,
        "fare": 31.27,
        "embarked": "S",
        "title": "Mr",
        "cabin": None,
        "historical_note": "Swedish farmer emigrating with his wife and 5 children. The entire family of 7 tragically perished in steerage.",
        "expected": "Perished"
    }
]


def load_model_artifact():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model artifact not found at {MODEL_FILE}.")
        print("Please train the model first by running:\n  python3 09_titanic_survival_prediction/train.py")
        sys.exit(1)
    return joblib.load(MODEL_FILE)


def format_probability_gauge(prob: float, width: int = 26) -> str:
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {prob * 100:>5.1f}%"


def explain_passenger_risk_factors(pclass: int, sex: str, age: float, family_size: int, title: str) -> list[str]:
    reasons = []
    # Gender factor
    if sex.lower() == "female":
        reasons.append("✅ Protocol Priority: Female passengers received top boarding priority ('Women and children first').")
    else:
        if title == "Master" or (age is not None and age < 12):
            reasons.append("✅ Child Exemption: Young male child ('Master') prioritized over adult male passengers.")
        else:
            reasons.append("⚠️ Strict Gender Restriction: Adult male passengers faced strict lifeboat denial by 2nd Officer Lightoller.")

    # Class factor
    if pclass == 1:
        reasons.append("✅ Boat Deck Proximity: 1st class cabins had direct stairwell access to the top boat deck.")
    elif pclass == 2:
        reasons.append("⚖️ Intermediate Access: 2nd class passengers were escorted to lifeboats once 1st class boarding commenced.")
    else:
        reasons.append("⚠️ Steerage Obstacles: 3rd class passengers were deep in lower decks, facing locked gates and maze-like corridors.")

    # Family factor
    if 2 <= family_size <= 4:
        reasons.append("✅ Optimal Traveling Party: Small families (2-4 pax) supported each other without overwhelming lifeboat capacity.")
    elif family_size > 4:
        reasons.append("⚠️ Large Family Burden: Large families (5+ pax) struggled to stay together, leading to severe casualty rates.")
    else:
        reasons.append("⚖️ Solo Traveler: Lacked assistance during middle-of-the-night emergency muster, lower survival on average.")

    return reasons


def predict_passenger(
    model_artifact: dict,
    pclass: int,
    sex: str,
    age: float = None,
    sibsp: int = 0,
    parch: int = 0,
    fare: float = None,
    embarked: str = "S",
    title: str = None,
    name: str = None,
    cabin: str = None,
    passenger_label: str = None
):
    pipeline = model_artifact["pipeline"]
    trained_medians = model_artifact["trained_medians"]

    # Assemble raw record
    raw_df = build_passenger_record(
        pclass=pclass,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        embarked=embarked,
        title=title,
        name=name,
        cabin=cabin
    )

    # Feature engineer using fitted medians
    X_feat, _ = engineer_features(raw_df, trained_medians=trained_medians)

    # Run inference
    pred = pipeline.predict(X_feat)[0]
    prob = pipeline.predict_proba(X_feat)[0][1] if hasattr(pipeline, "predict_proba") else (1.0 if pred == 1 else 0.0)

    survived = bool(pred == 1)
    status_icon = "🟢 SURVIVED" if survived else "🔴 PERISHED"
    resolved_age = float(X_feat["Age"].iloc[0])
    resolved_fare = float(np.expm1(X_feat["Fare_Log"].iloc[0]))
    resolved_title = str(X_feat["Title"].iloc[0])
    family_size = int(X_feat["FamilySize"].iloc[0])
    deck = str(X_feat["Deck"].iloc[0])
    port_names = {"S": "Southampton (UK)", "C": "Cherbourg (France)", "Q": "Queenstown (Ireland)"}
    port_display = port_names.get(embarked.upper(), embarked)

    print("\n" + "=" * 68)
    title_header = f" 🚢 PREDICTION: {status_icon}"
    if passenger_label:
        title_header += f" — {passenger_label}"
    print(title_header)
    print("=" * 68)

    print(f"  • Classification Result : {'SURVIVOR (Predicted Safe)' if survived else 'CASUALTY (Predicted Perished)'}")
    print(f"  • Survival Probability  : {format_probability_gauge(prob)}")
    print(f"  • Casualty Probability  : {format_probability_gauge(1.0 - prob)}")
    print("-" * 68)
    print("  Passenger Profile Details:")
    class_desc = {1: "1st Class (Upper Deck)", 2: "2nd Class (Middle)", 3: "3rd Class (Steerage)"}.get(pclass, f"{pclass} Class")
    print(f"    • Ticket Class : {class_desc}")
    print(f"    • Gender & Age : {sex.title()}, {resolved_age:.1f} years old (Title: '{resolved_title}')")
    print(f"    • Traveling As : {family_size} person(s) aboard ({sibsp} sibling/spouse, {parch} parent/child)")
    print(f"    • Ticket Fare  : £{resolved_fare:.2f} (Deck: '{deck}')")
    print(f"    • Departure    : {port_display}")

    print("\n  Sociological Risk & Survival Drivers:")
    reasons = explain_passenger_risk_factors(pclass, sex, resolved_age, family_size, resolved_title)
    for r in reasons:
        print(f"    {r}")
    print("=" * 68)

    return survived, prob


def run_batch_prediction(model_artifact: dict, input_csv: str, output_csv: str):
    print(f"\nProcessing batch inference on: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' does not exist.")
        return

    test_df = pd.read_csv(input_csv)
    print(f"Loaded {len(test_df)} passenger rows.")

    pipeline = model_artifact["pipeline"]
    trained_medians = model_artifact["trained_medians"]

    # Feature engineer using stored medians
    X_test_feat, _ = engineer_features(test_df, trained_medians=trained_medians)

    # Predict
    preds = pipeline.predict(X_test_feat)
    probs = pipeline.predict_proba(X_test_feat)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Save Kaggle submission format
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    submission_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": preds
    })
    submission_df.to_csv(output_csv, index=False)

    survived_count = int(preds.sum())
    total_count = len(preds)
    surv_rate = (survived_count / total_count) * 100

    print("\n" + "=" * 66)
    print(" 📊 KAGGLE BATCH PREDICTION SUMMARY")
    print("=" * 66)
    print(f"  • Total Manifest Evaluated   : {total_count} passengers")
    print(f"  • Predicted Survivors (1)    : {survived_count} passengers ({surv_rate:.1f}%)")
    print(f"  • Predicted Casualties (0)   : {total_count - survived_count} passengers ({100 - surv_rate:.1f}%)")
    print(f"  • Exported Kaggle Submission : {output_csv}")
    print("  • Verification (First 5 Rows):")
    print(submission_df.head(5).to_string(index=False))

    # Cross-tab summary by sex if available
    if "Sex" in test_df.columns:
        test_df["Predicted_Survived"] = preds
        grouped = test_df.groupby("Sex")["Predicted_Survived"].agg(["count", "sum", "mean"])
        print("\n  Predicted Survival Rate by Gender:")
        for sex_val, row in grouped.iterrows():
            print(f"    • {sex_val.title():<8} : {int(row['sum'])} / {int(row['count'])} survived ({row['mean']*100:.1f}%)")
    print("=" * 66 + "\n")


def run_interactive_menu(model_artifact: dict):
    while True:
        print("\n" + "=" * 66)
        print(" 🚢 TITANIC SURVIVAL PREDICTOR — INTERACTIVE SIMULATION 🚢")
        print("=" * 66)
        print("  1) Simulate Famous Historical Titanic Passengers (Jack, Rose, Molly...)")
        print("  2) Custom Passenger Survival Simulator (Enter your custom specs)")
        print("  3) Batch Predict Kaggle Test Set (`data/test.csv` -> `submission.csv`)")
        print("  4) View Model Architecture, Cross-Validation & Benchmark Scores")
        print("  5) Exit")
        print("-" * 66)

        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            print("\nSelect a historical passenger to simulate:")
            for idx, p in enumerate(HISTORICAL_PASSENGERS, 1):
                pclass_str = {1: "1st", 2: "2nd", 3: "3rd"}[p["pclass"]]
                print(f"  {idx}) {p['name']}")
                print(f"     Class: {pclass_str} | {p['sex'].title()}, {p['age']:.0f}yo | Fare: £{p['fare']:.2f} | Expected: {p['expected']}")

            p_choice = input(f"\nSelect passenger (1-{len(HISTORICAL_PASSENGERS)}): ").strip()
            if p_choice.isdigit() and 1 <= int(p_choice) <= len(HISTORICAL_PASSENGERS):
                selected = HISTORICAL_PASSENGERS[int(p_choice) - 1]
                predict_passenger(
                    model_artifact,
                    pclass=selected["pclass"],
                    sex=selected["sex"],
                    age=selected["age"],
                    sibsp=selected["sibsp"],
                    parch=selected["parch"],
                    fare=selected["fare"],
                    embarked=selected["embarked"],
                    title=selected["title"],
                    cabin=selected.get("cabin"),
                    passenger_label=selected["name"]
                )
                print(f"  Historical Reality: {selected['historical_note']}\n")
            else:
                print("Invalid passenger selection.")

        elif choice == "2":
            print("\n--- Custom Passenger Simulation Wizard ---")
            try:
                pclass_in = input("  Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd) [default: 3]: ").strip()
                pclass = int(pclass_in) if pclass_in in ["1", "2", "3"] else 3

                sex_in = input("  Gender (male / female) [default: male]: ").strip().lower()
                sex = sex_in if sex_in in ["male", "female"] else "male"

                age_in = input("  Age in years (e.g. 25, press Enter to impute): ").strip()
                age = float(age_in) if age_in else None

                sibsp_in = input("  Number of siblings / spouses aboard [default: 0]: ").strip()
                sibsp = int(sibsp_in) if sibsp_in.isdigit() else 0

                parch_in = input("  Number of parents / children aboard [default: 0]: ").strip()
                parch = int(parch_in) if parch_in.isdigit() else 0

                fare_in = input("  Ticket Fare in British Pounds (press Enter for class median): ").strip()
                fare = float(fare_in) if fare_in else None

                emb_in = input("  Port of Embarkation (S = Southampton, C = Cherbourg, Q = Queenstown) [default: S]: ").strip().upper()
                embarked = emb_in if emb_in in ["S", "C", "Q"] else "S"

                predict_passenger(
                    model_artifact,
                    pclass=pclass,
                    sex=sex,
                    age=age,
                    sibsp=sibsp,
                    parch=parch,
                    fare=fare,
                    embarked=embarked,
                    passenger_label="Custom Passenger Simulation"
                )
            except ValueError:
                print("Invalid input value entered. Please enter numeric values for age, family, and fare.")

        elif choice == "3":
            run_batch_prediction(model_artifact, TEST_CSV_FILE, DEFAULT_SUBMISSION_FILE)

        elif choice == "4":
            print("\n" + "=" * 66)
            print(" 🏆 TRAINED MODEL SPECIFICATIONS & BENCHMARK")
            print("=" * 66)
            print(f"  • Champion Model Architecture : {model_artifact['model_name']}")
            print(f"  • Generalization Test Accuracy: {model_artifact['metrics']['Test Acc'] * 100:.2f}%")
            print(f"  • 5-Fold Stratified CV Mean   : {model_artifact['metrics']['CV Mean Acc'] * 100:.2f}% (±{model_artifact['metrics']['CV Std'] * 100:.2f}%)")
            print(f"  • F1-Score / ROC-AUC          : {model_artifact['metrics']['F1-Score']:.3f} / {model_artifact['metrics']['ROC-AUC']:.3f}")
            print("\n  Benchmark Leaderboard Comparison:")
            print(f"    {'Model Name':<28} | {'CV Accuracy':<15} | {'Test Accuracy':<12}")
            print("    " + "-" * 58)
            for r in model_artifact.get("benchmark_summary", []):
                cv_str = f"{r['CV Mean Acc'] * 100:.2f}%"
                t_str = f"{r['Test Acc'] * 100:.2f}%"
                print(f"    {r['Model']:<28} | {cv_str:<15} | {t_str:<12}")
            print("=" * 66)

        elif choice == "5":
            print("\nExiting Titanic Survival Simulator. Fair winds and following seas!\n")
            break
        else:
            print("Invalid option. Please choose 1 to 5.")


def main():
    parser = argparse.ArgumentParser(description="Predict RMS Titanic passenger survival outcome.")
    parser.add_argument("--pclass", type=int, choices=[1, 2, 3], help="Passenger Ticket Class (1, 2, or 3)")
    parser.add_argument("--sex", type=str, choices=["male", "female"], help="Gender ('male' or 'female')")
    parser.add_argument("--age", type=float, help="Age in years (e.g. 29.0)")
    parser.add_argument("--sibsp", type=int, default=0, help="Number of siblings or spouses aboard (default: 0)")
    parser.add_argument("--parch", type=int, default=0, help="Number of parents or children aboard (default: 0)")
    parser.add_argument("--fare", type=float, help="Ticket fare paid in GBP (optional)")
    parser.add_argument("--embarked", type=str, choices=["S", "C", "Q"], default="S", help="Port of embarkation ('S', 'C', or 'Q')")
    parser.add_argument("--title", type=str, help="Honorific title (Mr, Miss, Mrs, Master, Officer, Noble)")
    parser.add_argument("--cabin", type=str, help="Cabin room number (e.g. C85, B52)")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch predictions (e.g. data/test.csv)")
    parser.add_argument("--output", type=str, default=DEFAULT_SUBMISSION_FILE, help="Output path for batch predictions CSV")

    args = parser.parse_args()
    model_artifact = load_model_artifact()

    # Batch prediction mode
    if args.batch:
        run_batch_prediction(model_artifact, args.batch, args.output)
        return

    # Direct CLI flag prediction
    if args.pclass is not None and args.sex is not None:
        predict_passenger(
            model_artifact=model_artifact,
            pclass=args.pclass,
            sex=args.sex,
            age=args.age,
            sibsp=args.sibsp,
            parch=args.parch,
            fare=args.fare,
            embarked=args.embarked,
            title=args.title,
            cabin=args.cabin,
            passenger_label="CLI Direct Parameter Inference"
        )
    else:
        run_interactive_menu(model_artifact)


if __name__ == "__main__":
    main()
