"""
Interactive Customer Segmentation & Persona Profiler.
Classifies new shoppers into behavioral clusters and outputs actionable marketing strategies.
"""

import os
import sys
import argparse
import joblib
import numpy as np


PRESET_ARCHETYPES = [
    {
        "name": "Luxury Tech Executive",
        "income": 115,
        "spending": 86,
        "age": 34,
        "archetype": "High Income, High Spending"
    },
    {
        "name": "Frugal High-Earner (Doctor/Dentist)",
        "income": 105,
        "spending": 18,
        "age": 48,
        "archetype": "High Income, Low Spending"
    },
    {
        "name": "College Trend Follower",
        "income": 25,
        "spending": 82,
        "age": 21,
        "archetype": "Low Income, High Spending"
    },
    {
        "name": "Sensible Budget Shopper",
        "income": 22,
        "spending": 14,
        "age": 55,
        "archetype": "Low Income, Low Spending"
    },
    {
        "name": "Middle-Class Suburban Shopper",
        "income": 58,
        "spending": 50,
        "age": 38,
        "archetype": "Mid Income, Mid Spending"
    }
]


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "customer_segmentation.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'.")
        print("Please train the model first by running:\n  python3 06_customer_segmentation/train.py")
        sys.exit(1)
    return joblib.load(model_path)


def render_ascii_quadrant(income: float, spending: float, persona_name: str):
    """Renders visual 2D position within the retail landscape."""
    inc_pos = min(max(int(income / 14), 0), 9)
    spd_pos = min(max(int(spending / 10), 0), 9)

    print("\n  2D Retail Landscape Position (X: Income, Y: Spending):")
    print("  Spending Score ^")
    for r in range(9, -1, -1):
        row_str = "    "
        for c in range(10):
            if r == spd_pos and c == inc_pos:
                row_str += " 📍"
            else:
                row_str += "  ."
        if r == 8:
            row_str += "   [Impulsive Spenders]         [VIP Luxury Shoppers]"
        elif r == 5:
            row_str += "                [Standard Balanced]"
        elif r == 1:
            row_str += "   [Budget Savers]              [Careful High-Earners]"
        print(row_str)
    print("                 +---------------------------------------------------->")
    print("                    $15k                                       $140k  Annual Income\n")


def profile_customer(income: float, spending: float, age: float = None, model_artifact: dict = None):
    km_model = model_artifact["model"]
    personas = model_artifact["personas"]

    # Predict cluster
    point = np.array([[income, spending]])
    cluster_id = int(km_model.predict(point)[0])
    p_info = personas[cluster_id]

    # Calculate Euclidean distance to centroid
    centroid = np.array(p_info["centroid"])
    dist = float(np.linalg.norm(point - centroid))
    fit_confidence = max(0.0, 100.0 - (dist * 1.5))

    print("\n" + "=" * 70)
    print(f" {p_info['badge']} CUSTOMER SEGMENT: {p_info['persona'].upper()} (Cluster #{cluster_id + 1})")
    print("=" * 70)
    print(f"  • Annual Income        : ${income:,.0f}k")
    print(f"  • Spending Score       : {spending:.0f} / 100")
    if age:
        print(f"  • Age                  : {age:.0f} years old")
    print(f"  • Centroid Proximity   : {fit_confidence:.1f}% fit affinity (dist: {dist:.2f})")
    print(f"  • Cluster Cohort Size  : {p_info['count']} customers ({p_info['share_pct']:.1f}% of mall shoppers)")
    print(f"  • Demographic Avg      : Income ${p_info['mean_income']:.1f}k | Spending {p_info['mean_spending']:.1f}/100 | Age {p_info['mean_age']:.1f}y")

    render_ascii_quadrant(income, spending, p_info["persona"])

    print("  📋 BEHAVIORAL PROFILE:")
    print(f"     {p_info['description']}")
    print("\n  🎯 RECOMMENDED MARKETING PLAYBOOK:")
    print(f"     {p_info['strategy']}")
    print("=" * 70 + "\n")


def run_interactive(model_artifact):
    while True:
        print("\n" + "=" * 65)
        print(" 🛍️ RETAIL CUSTOMER SEGMENTATION - INTERACTIVE PROFILER 🛍️")
        print("=" * 65)
        print("  1) Profile New Customer (Enter Income & Spending Score)")
        print("  2) Test Pre-defined Consumer Archetypes")
        print("  3) View All 5 Segment Playbooks & Centroids")
        print("  4) Exit")
        print("-" * 65)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            try:
                income = float(input("\n  Enter Annual Income in thousands (e.g. 75 for $75,000): ").strip())
                spending = float(input("  Enter Spending Score (1 to 100): ").strip())
                age_str = input("  Enter Age (optional, press Enter to skip): ").strip()
                age = float(age_str) if age_str else None
                profile_customer(income, spending, age, model_artifact)
            except ValueError:
                print("Invalid input. Please enter valid numerical values.")

        elif choice == "2":
            print("\nSelect a pre-defined customer archetype:")
            for idx, arch in enumerate(PRESET_ARCHETYPES, 1):
                print(f"  {idx}) {arch['name']} (Income: ${arch['income']}k | Spend: {arch['spending']}/100 | Age: {arch['age']})")

            sel = input(f"Select (1-{len(PRESET_ARCHETYPES)}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(PRESET_ARCHETYPES):
                chosen = PRESET_ARCHETYPES[int(sel) - 1]
                profile_customer(chosen["income"], chosen["spending"], chosen["age"], model_artifact)
            else:
                print("Invalid selection.")

        elif choice == "3":
            print("\n" + "=" * 70)
            print(" 👥 THE 5 CORE RETAIL SEGMENT CENTROIDS & COHORTS")
            print("=" * 70)
            for c_id, p in model_artifact["personas"].items():
                print(f"  {p['badge']} {p['persona']:<20} | Centroid: (${p['mean_income']:>5.1f}k, {p['mean_spending']:>4.1f}/100) | {p['share_pct']:>4.1f}% share")
                print(f"     Action: {p['strategy']}\n")

        elif choice == "4":
            print("Exiting Customer Segmentation Engine. Happy retail targeting!\n")
            break
        else:
            print("Invalid option. Please choose 1 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Classify customer into behavioral shopping segments.")
    parser.add_argument("--income", type=float, help="Annual Income in k$ (e.g. 85)")
    parser.add_argument("--spending-score", type=float, help="Spending Score 1-100 (e.g. 78)")
    parser.add_argument("--age", type=float, help="Age in years (optional)")

    args = parser.parse_args()
    model_artifact = load_model()

    if args.income is not None and args.spending_score is not None:
        profile_customer(args.income, args.spending_score, args.age, model_artifact)
    elif args.income is not None or args.spending_score is not None:
        print("Error: Both --income and --spending-score are required for CLI profiling.")
    else:
        run_interactive(model_artifact)


if __name__ == "__main__":
    main()
