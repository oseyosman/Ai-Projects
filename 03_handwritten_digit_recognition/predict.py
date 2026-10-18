"""
Interactive Handwritten Digit Predictor.
Classifies individual digit images from the test set and renders ASCII previews.
"""

import os
import argparse
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_digit_classifier.joblib")


def render_digit_ascii(flat_pixels, width=8, scale=2):
    """Render a flat 64-length array as ASCII art."""
    chars = " .:-=+*#@"
    image = flat_pixels.reshape(width, width)
    lines = []
    for row in image:
        line = ""
        for pixel in row:
            idx = min(int((pixel / 17.0) * len(chars)), len(chars) - 1)
            line += chars[idx] * scale
        lines.append(line)
    return lines


def render_pixel_grid(flat_pixels, width=8):
    """Render pixel values as a color-coded numeric grid."""
    image = flat_pixels.reshape(width, width)
    lines = []
    for row in image:
        line = "  "
        for val in row:
            if val == 0:
                line += "  . "
            elif val < 6:
                line += f" {int(val):>2} "
            elif val < 12:
                line += f"[{int(val):>2}]"
            else:
                line += f" {int(val):>2} "
        lines.append(line)
    return lines


def predict_single(artifact, pixels_flat):
    pipeline = artifact["pipeline"]
    pred = pipeline.predict([pixels_flat])[0]

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([pixels_flat])[0]
    elif hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([pixels_flat])[0]
        exp_s = np.exp(scores - scores.max())
        proba = exp_s / exp_s.sum()
    else:
        proba = np.zeros(10)
        proba[pred] = 1.0

    return int(pred), proba


def print_prediction(pred, proba, true_label=None):
    print("\n" + "=" * 52)
    correct = (pred == true_label) if true_label is not None else None
    status = ""
    if correct is True:
        status = "  ✅ CORRECT"
    elif correct is False:
        status = f"  ❌ WRONG (true label: {true_label})"

    print(f"  🔢 PREDICTED DIGIT:  {pred}{status}")
    if true_label is not None:
        print(f"  🏷️  True Label:       {true_label}")
    print(f"  🎯 Confidence:       {proba[pred] * 100:.1f}%")

    # Top-3 confidence bar chart
    top3 = np.argsort(proba)[::-1][:3]
    print("\n  Top 3 Candidate Probabilities:")
    for d in top3:
        bar_len = int(proba[d] * 20)
        print(f"  Digit {d}: {'█' * bar_len}{'░' * (20 - bar_len)} {proba[d] * 100:>5.1f}%")
    print("=" * 52)


def interactive_menu(artifact):
    print("\n" + "=" * 60)
    print(f" ✍️  Handwritten Digit Predictor")
    print(f" Loaded Model: {artifact['model_name']}")
    print(f" Accuracy:     {artifact['metrics']['accuracy']:.2%}")
    print("=" * 60)

    # Load full dataset and create test split so we can pick examples
    digits = load_digits()
    X_all, y_all = digits.data, digits.target
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    n_test = len(X_test)

    print(f"\n{n_test} test images available (indices 0 – {n_test - 1})")
    print("\nOptions:")
    print(" [1] Show 10 sample predictions (one per digit class)")
    print(" [2] Show random test image")
    print(" [3] Enter specific test index")
    print(" [4] Run full evaluation on all test images")
    print(" [q] Quit")

    choice = input("\nYour choice [1-4, q]: ").strip()

    if choice == "1":
        print("\n─── One Sample Per Digit Class ───────────────────────────")
        for target_digit in range(10):
            idxs = np.where(y_test == target_digit)[0]
            if len(idxs) == 0:
                continue
            i = idxs[0]
            pixels = X_test[i]
            pred, proba = predict_single(artifact, pixels)

            ascii_art = render_digit_ascii(pixels)
            print(f"\n  ── True Digit: {target_digit} ──────────────────────")
            for row in ascii_art:
                print(f"  {row}")
            status = "✅" if pred == target_digit else "❌"
            print(f"  Predicted: {pred}  Confidence: {proba[pred] * 100:.1f}%  {status}")

    elif choice == "2":
        rng = np.random.default_rng()
        i = rng.integers(0, n_test)
        pixels = X_test[i]
        true_label = y_test[i]

        print(f"\nRandom test image (index {i}):")
        ascii_art = render_digit_ascii(pixels)
        for row in ascii_art:
            print(f"  {row}")

        print(f"\nPixel Value Grid (0=blank, 16=dark ink):")
        for row in render_pixel_grid(pixels):
            print(row)

        pred, proba = predict_single(artifact, pixels)
        print_prediction(pred, proba, true_label)

    elif choice == "3":
        try:
            idx = int(input(f"Enter test index [0–{n_test - 1}]: "))
            if 0 <= idx < n_test:
                pixels = X_test[idx]
                true_label = y_test[idx]

                print(f"\nTest image at index {idx}:")
                for row in render_digit_ascii(pixels):
                    print(f"  {row}")

                pred, proba = predict_single(artifact, pixels)
                print_prediction(pred, proba, true_label)
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid input.")

    elif choice == "4":
        print(f"\nRunning full evaluation on {n_test} test images...")
        correct = 0
        wrong_cases = []
        per_digit_correct = {d: 0 for d in range(10)}
        per_digit_total = {d: 0 for d in range(10)}

        for i in range(n_test):
            pixels = X_test[i]
            true_label = int(y_test[i])
            pred, proba = predict_single(artifact, pixels)

            per_digit_total[true_label] += 1
            if pred == true_label:
                correct += 1
                per_digit_correct[true_label] += 1
            else:
                wrong_cases.append((i, true_label, pred, proba[pred]))

        acc = correct / n_test
        print(f"\nOverall Accuracy: {correct}/{n_test} = {acc:.2%}")
        print("\nPer-Digit Accuracy:")
        for d in range(10):
            tot = per_digit_total[d]
            cor = per_digit_correct[d]
            pct = cor / tot if tot > 0 else 0
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(f"  Digit {d}: {bar}  {cor}/{tot} ({pct:.1%})")

        if wrong_cases:
            print(f"\nFirst 5 Misclassifications (true→predicted):")
            for i, true_l, pred_l, conf in wrong_cases[:5]:
                print(f"  Sample {i:>4}: true={true_l}  predicted={pred_l}  conf={conf:.1%}")

    elif choice.lower() == "q":
        print("Goodbye!")
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digits")
    parser.add_argument("--index", type=int, default=None, help="Test set index (0 to ~359)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please run: python3 train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    if args.index is not None:
        digits = load_digits()
        X_all, y_all = digits.data, digits.target
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        idx = args.index
        if 0 <= idx < len(X_test):
            pixels = X_test[idx]
            true_label = y_test[idx]
            print(f"\nTest image at index {idx}:")
            for row in render_digit_ascii(pixels):
                print(f"  {row}")
            pred, proba = predict_single(artifact, pixels)
            print_prediction(pred, proba, true_label)
        else:
            print(f"Index {idx} out of range (0–{len(X_test) - 1}).")
    else:
        interactive_menu(artifact)


if __name__ == "__main__":
    main()
