"""
Interactive Iris Flower Species Predictor.
Accepts custom botanical measurements via CLI arguments or interactive menu,
computes class probabilities, and outputs species classification.
"""

import os
import sys
import argparse
import joblib
import numpy as np


SPECIES_PROFILES = {
    "setosa": {
        "common_name": "Bristle-pointed Iris (Iris setosa)",
        "emoji": "🌸",
        "description": "Known for small petals and wide sepals. Native to Alaska, Canada, and northeastern Asia. 100% distinct from other iris varieties.",
        "typical_petal": "1.0 - 1.9 cm long, 0.1 - 0.6 cm wide"
    },
    "versicolor": {
        "common_name": "Harlequin Blueflag (Iris versicolor)",
        "emoji": "🌺",
        "description": "Medium-sized petals and sepals. Native to eastern North America. Moderate petal length and slender form.",
        "typical_petal": "3.0 - 5.1 cm long, 1.0 - 1.8 cm wide"
    },
    "virginica": {
        "common_name": "Virginia Iris (Iris virginica)",
        "emoji": "🪻",
        "description": "Largest petals and sepals among the three. Native to southeastern North America with robust purple-blue blossoms.",
        "typical_petal": "4.5 - 6.9 cm long, 1.4 - 2.5 cm wide"
    }
}

PRESET_SPECIMENS = [
    {
        "name": "Specimen A (Small petals, high sepal width)",
        "values": [5.1, 3.5, 1.4, 0.2],
        "expected": "setosa"
    },
    {
        "name": "Specimen B (Medium petals, moderate proportions)",
        "values": [5.7, 2.8, 4.1, 1.3],
        "expected": "versicolor"
    },
    {
        "name": "Specimen C (Long petals, wide petals)",
        "values": [6.7, 3.1, 5.6, 2.4],
        "expected": "virginica"
    },
    {
        "name": "Specimen D (Borderline transition case: 4.8cm petal)",
        "values": [6.0, 2.7, 4.8, 1.6],
        "expected": "versicolor / virginica borderline"
    }
]


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "iris_classifier.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'.")
        print("Please train the model first by running:\n  python3 04_iris_flower_classification/train.py")
        sys.exit(1)
    return joblib.load(model_path)


def format_probability_bar(prob: float, width: int = 25) -> str:
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {prob * 100:>5.1f}%"


def predict_specimen(pipeline, feature_names, target_names, sepal_len, sepal_w, petal_len, petal_w):
    features = np.array([[sepal_len, sepal_w, petal_len, petal_w]])
    pred_idx = pipeline.predict(features)[0]
    predicted_species = target_names[pred_idx]

    # Get probabilities if available
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(features)[0]

    profile = SPECIES_PROFILES.get(predicted_species, {})

    print("\n" + "=" * 65)
    print(f" 🌸 PREDICTION: IRIS {predicted_species.upper()} {profile.get('emoji', '')}")
    print("=" * 65)
    print(f"  • Common Name : {profile.get('common_name', predicted_species.title())}")
    print(f"  • Measurements: Sepal {sepal_len}x{sepal_w} cm | Petal {petal_len}x{petal_w} cm")
    print(f"  • Botanical Note: {profile.get('description', '')}")

    if probs is not None:
        print("\n  Class Probabilities:")
        for idx, sp in enumerate(target_names):
            bar = format_probability_bar(probs[idx])
            marker = " 👈 (Top Match)" if idx == pred_idx else ""
            print(f"    Iris {sp:<11} {bar}{marker}")

    print("=" * 65)
    return predicted_species


def run_interactive(model_artifact):
    pipeline = model_artifact["pipeline"]
    feature_names = model_artifact["feature_names"]
    target_names = model_artifact["target_names"]

    while True:
        print("\n" + "=" * 60)
        print(" 🌸 IRIS FLOWER SPECIES CLASSIFIER - INTERACTIVE MENU 🌸")
        print("=" * 60)
        print("  1) Predict with Pre-loaded Botanical Specimen")
        print("  2) Enter Custom Flower Measurements (Sepal & Petal)")
        print("  3) View Model Information & Features")
        print("  4) Exit")
        print("-" * 60)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("\nSelect a botanical specimen:")
            for i, spec in enumerate(PRESET_SPECIMENS, 1):
                vals = f"Sepal: {spec['values'][0]}x{spec['values'][1]} | Petal: {spec['values'][2]}x{spec['values'][3]}"
                print(f"  {i}) {spec['name']}")
                print(f"     ({vals} | Expected: {spec['expected']})")

            spec_choice = input(f"\nSelect specimen (1-{len(PRESET_SPECIMENS)}): ").strip()
            if spec_choice.isdigit() and 1 <= int(spec_choice) <= len(PRESET_SPECIMENS):
                selected = PRESET_SPECIMENS[int(spec_choice) - 1]
                vals = selected["values"]
                predict_specimen(pipeline, feature_names, target_names, *vals)
            else:
                print("Invalid selection.")

        elif choice == "2":
            print("\nEnter flower measurements in centimeters (cm):")
            try:
                sl = float(input("  Sepal Length (typical 4.3 - 7.9): ").strip())
                sw = float(input("  Sepal Width  (typical 2.0 - 4.4): ").strip())
                pl = float(input("  Petal Length (typical 1.0 - 6.9): ").strip())
                pw = float(input("  Petal Width  (typical 0.1 - 2.5): ").strip())
                predict_specimen(pipeline, feature_names, target_names, sl, sw, pl, pw)
            except ValueError:
                print("Invalid numeric measurement entered.")

        elif choice == "3":
            print("\nModel Artifact Info:")
            print(f"  • Trained Model Architecture : {model_artifact['model_name']}")
            print(f"  • Held-out Test Accuracy     : {model_artifact['accuracy'] * 100:.2f}%")
            print(f"  • Features Required          : {', '.join(feature_names)}")
            print(f"  • Target Species             : {', '.join(target_names)}")

        elif choice == "4":
            print("Exiting Iris Classifier. Happy botanical modeling!\n")
            break
        else:
            print("Invalid option. Please choose 1 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Predict Iris flower species from 4 biometric measurements.")
    parser.add_argument("--sepal-length", type=float, help="Sepal length in cm (e.g. 5.1)")
    parser.add_argument("--sepal-width", type=float, help="Sepal width in cm (e.g. 3.5)")
    parser.add_argument("--petal-length", type=float, help="Petal length in cm (e.g. 1.4)")
    parser.add_argument("--petal-width", type=float, help="Petal width in cm (e.g. 0.2)")

    args = parser.parse_args()
    model_artifact = load_model()

    # If any feature was passed via CLI flags
    cli_flags = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
    if any(f is not None for f in cli_flags):
        if any(f is None for f in cli_flags):
            print("Error: All 4 measurements (--sepal-length, --sepal-width, --petal-length, --petal-width) must be supplied.")
            sys.exit(1)
        predict_specimen(
            model_artifact["pipeline"],
            model_artifact["feature_names"],
            model_artifact["target_names"],
            args.sepal_length,
            args.sepal_width,
            args.petal_length,
            args.petal_width
        )
    else:
        run_interactive(model_artifact)


if __name__ == "__main__":
    main()
