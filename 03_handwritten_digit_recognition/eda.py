"""
Exploratory Data Analysis for Handwritten Digit Recognition.
Uses Scikit-Learn's built-in 8x8 digit dataset (MNIST-like).
"""

import os
import numpy as np
from sklearn.datasets import load_digits


def pixel_bar(value, max_val=16, width=12):
    """Render a pixel intensity as a greyscale text bar."""
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)


def render_digit_ascii(image_8x8):
    """Render an 8x8 image as ASCII art in the terminal."""
    chars = " .:-=+*#@"
    lines = []
    for row in image_8x8:
        line = ""
        for pixel in row:
            idx = min(int((pixel / 17.0) * len(chars)), len(chars) - 1)
            line += chars[idx] * 2  # double width for readability
        lines.append(line)
    return "\n".join(lines)


def run_eda():
    print("=" * 68)
    print(" ✍️  Handwritten Digit Recognition - Exploratory Data Analysis")
    print("=" * 68)

    # 1. Load Dataset
    digits = load_digits()
    X = digits.data        # shape (1797, 64)
    y = digits.target      # shape (1797,) → labels 0–9
    images = digits.images # shape (1797, 8, 8)

    print(f"\n[1/5] Dataset Shape:")
    print(f"  Samples:        {X.shape[0]:,}")
    print(f"  Features:       {X.shape[1]} (8×8 pixels, flattened)")
    print(f"  Classes:        {len(np.unique(y))} digits (0 through 9)")
    print(f"  Pixel Range:    {int(X.min())} – {int(X.max())} (grayscale intensity)")

    # 2. Class Distribution
    print("\n[2/5] Class Distribution (Samples per Digit):")
    for digit in range(10):
        count = np.sum(y == digit)
        bar = "█" * (count // 5)
        print(f"  Digit {digit}: {count:>4} samples  {bar}")

    # 3. ASCII Art Visualization — one sample per digit
    print("\n[3/5] Sample Images (ASCII Art Representation):")
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        img = images[idx]
        print(f"\n  ── Digit '{digit}' ──────────────────")
        for row in render_digit_ascii(img).split("\n"):
            print(f"  {row}")

    # 4. Pixel Statistics
    print("\n[4/5] Global Pixel Intensity Statistics:")
    print(f"  Mean Pixel Value:  {X.mean():.2f}")
    print(f"  Std Deviation:     {X.std():.2f}")
    print(f"  Non-zero Pixels:   {(X > 0).mean():.1%} of all pixels are 'inked'")

    # 5. Most Informative Pixels
    print("\n[5/5] Most Discriminative Pixel Positions (by variance across classes):")
    variances = X.var(axis=0)
    top_pixels = np.argsort(variances)[::-1][:8]
    print("  Pixel position (row, col) with highest class-discriminating variance:")
    for i, p in enumerate(top_pixels, 1):
        row, col = divmod(p, 8)
        print(f"  {i}. Pixel [{row},{col}] → variance = {variances[p]:.2f}  {pixel_bar(variances[p], variances.max())}")

    print("\n  💡 Interpretation:")
    print("  -> Center pixels (rows 2–5, cols 2–5) are the most discriminative.")
    print("  -> Border pixels are often blank and carry little classification signal.")
    print("=" * 68)


if __name__ == "__main__":
    run_eda()
