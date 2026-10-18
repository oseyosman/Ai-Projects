"""
Exploratory Data Analysis (EDA) for Titanic Survival Prediction.
Analyzes passenger demographics, class disparities, family dynamics,
missing value structures, and survival rates across RMS Titanic passengers.
"""

import os
import numpy as np
import pandas as pd


def print_banner(text: str, char: str = "=", width: int = 72):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def render_ascii_bar(ratio: float, width: int = 30) -> str:
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def extract_title(name: str) -> str:
    """Extract honorific title from passenger name."""
    if pd.isna(name):
        return "Unknown"
    # Format: Last, Title. First Middle
    parts = name.split(",")
    if len(parts) > 1:
        title_part = parts[1].split(".")[0].strip()
        return title_part
    return "Unknown"


def categorize_title(title: str) -> str:
    """Standardize honorific titles into common cohorts."""
    if title in ["Mr"]:
        return "Mr"
    elif title in ["Miss", "Mlle", "Ms"]:
        return "Miss"
    elif title in ["Mrs", "Mme"]:
        return "Mrs"
    elif title in ["Master"]:
        return "Master"  # Young boys
    elif title in ["Dr", "Rev", "Col", "Major", "Capt"]:
        return "Officer"
    elif title in ["Don", "Sir", "Lady", "the Countess", "Jonkheer", "Dona"]:
        return "Noble"
    else:
        return "Rare"


def main():
    print_banner("🚢 PROJECT 09: TITANIC SURVIVAL EXPLORATORY DATA ANALYSIS 🚢")

    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    total_passengers = len(df)

    print(f"Dataset Loaded: {total_passengers} passenger records with {df.shape[1]} raw attributes.")
    print("Target Variable: 'Survived' (0 = Perished, 1 = Survived)")

    # 1. Missing Value Audit
    print_banner("1. DATA INTEGRITY & MISSING VALUE AUDIT")
    null_counts = df.isnull().sum()
    print(f"{'Feature':<16} | {'Missing Count':<14} | {'Missing %':<12} | {'Data Type':<10}")
    print("-" * 60)
    for col in df.columns:
        n_missing = null_counts[col]
        pct = (n_missing / total_passengers) * 100
        dtype = str(df[col].dtype)
        flag = " ⚠️ Needs Imputation" if n_missing > 0 else " ✅ Complete"
        print(f"{col:<16} | {n_missing:<14} | {pct:>9.2f}%  | {dtype:<10} {flag}")

    # 2. Overall Target Distribution
    print_banner("2. OVERALL SURVIVAL DISTRIBUTION")
    survived_cnt = df["Survived"].sum()
    perished_cnt = total_passengers - survived_cnt
    surv_rate = (survived_cnt / total_passengers) * 100
    per_rate = (perished_cnt / total_passengers) * 100

    print(f"  Perished (0): {perished_cnt:>4} passengers ({per_rate:5.1f}%) |{render_ascii_bar(per_rate / 100, 24)}|")
    print(f"  Survived (1): {survived_cnt:>4} passengers ({surv_rate:5.1f}%) |{render_ascii_bar(surv_rate / 100, 24)}|")
    print(f"  Baseline Survival Rate: {surv_rate:.2f}% (Significant Class Imbalance: ~62% casualty rate)")

    # 3. Gender Disparity ("Women and Children First")
    print_banner("3. SURVIVAL BY GENDER (THE MARITIME PROTOCOL)")
    gender_group = df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"])
    print(f"{'Gender':<12} | {'Total':<8} | {'Survived':<10} | {'Perished':<10} | {'Survival Rate':<16}")
    print("-" * 66)
    for sex, row in gender_group.iterrows():
        total = int(row["count"])
        surv = int(row["sum"])
        per = total - surv
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 18)
        print(f"{sex.title():<12} | {total:<8} | {surv:<10} | {per:<10} | {rate:>5.1f}% |{bar}|")

    # 4. Socioeconomic Class (Pclass) Disparity
    print_banner("4. SURVIVAL BY PASSENGER TICKET CLASS (PCLASS)")
    class_map = {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Steerage)"}
    class_group = df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"])
    print(f"{'Ticket Class':<22} | {'Total':<8} | {'Survived':<10} | {'Survival Rate':<18}")
    print("-" * 66)
    for pclass, row in class_group.iterrows():
        name = class_map[pclass]
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 18)
        print(f"{name:<22} | {int(row['count']):<8} | {int(row['sum']):<10} | {rate:>5.1f}% |{bar}|")

    # 5. Interaction: Class x Gender
    print_banner("5. INTERACTION EFFECT: TICKET CLASS × GENDER")
    pivot_cg = df.pivot_table(index="Pclass", columns="Sex", values="Survived", aggfunc=["count", "mean"])
    print(f"{'Ticket Class':<22} | {'Female Survival':<18} | {'Male Survival':<18}")
    print("-" * 64)
    for pclass in [1, 2, 3]:
        f_rate = pivot_cg[("mean", "female")][pclass] * 100
        m_rate = pivot_cg[("mean", "male")][pclass] * 100
        f_cnt = pivot_cg[("count", "female")][pclass]
        m_cnt = pivot_cg[("count", "male")][pclass]
        print(f"{class_map[pclass]:<22} | {f_rate:>5.1f}% ({f_cnt:>3} pax)     | {m_rate:>5.1f}% ({m_cnt:>3} pax)")

    # 6. Age Demographics
    print_banner("6. SURVIVAL BY AGE BRACKETS")
    df_age = df.dropna(subset=["Age"]).copy()
    bins = [0, 9, 18, 35, 55, 100]
    labels = ["Children (0-9)", "Teens (10-18)", "Young Adults (19-35)", "Middle Aged (36-55)", "Seniors (56+)"]
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=bins, labels=labels, right=True)
    age_group = df_age.groupby("AgeGroup", observed=False)["Survived"].agg(["count", "sum", "mean"])

    print(f"{'Age Bracket':<22} | {'Sample Count':<14} | {'Survival Rate':<18}")
    print("-" * 60)
    for group_name, row in age_group.iterrows():
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 18)
        print(f"{group_name:<22} | {int(row['count']):<14} | {rate:>5.1f}% |{bar}|")

    # 7. Family Size & Traveling Party
    print_banner("7. FAMILY SIZE DYNAMICS (SibSp + Parch + 1)")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    fam_group = df.groupby("FamilySize")["Survived"].agg(["count", "mean"])
    print(f"{'Family Size':<14} | {'Passengers':<12} | {'Survival Rate':<18} | {'Interpretation':<20}")
    print("-" * 72)
    for fam_sz, row in fam_group.iterrows():
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 14)
        interp = "Solo traveler" if fam_sz == 1 else ("Optimal size (2-4)" if fam_sz <= 4 else "Large family burden")
        print(f"{fam_sz:>2} person(s)   | {int(row['count']):<12} | {rate:>5.1f}% |{bar}| | {interp}")

    # 8. Feature Engineering: Honorific Titles
    print_banner("8. ENGINEERED FEATURE: HONORIFIC TITLE EXTRACTION")
    df["Title"] = df["Name"].apply(extract_title).apply(categorize_title)
    title_group = df.groupby("Title")["Survived"].agg(["count", "mean"])
    print(f"{'Title Cohort':<14} | {'Count':<8} | {'Survival Rate':<18} | {'Demographic Meaning':<28}")
    print("-" * 74)
    descriptions = {
        "Mr": "Adult men (lowest priority)",
        "Mrs": "Married women (high priority)",
        "Miss": "Unmarried women / girls",
        "Master": "Young male children (boy priority)",
        "Officer": "Ship & military officers (duty bound)",
        "Noble": "Aristocracy / dignitaries",
        "Rare": "Uncommon ceremonial titles"
    }
    for title, row in title_group.sort_values(by="count", ascending=False).iterrows():
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 14)
        desc = descriptions.get(title, "Special designation")
        print(f"{title:<14} | {int(row['count']):<8} | {rate:>5.1f}% |{bar}| | {desc}")

    # 9. Embarkation Port & Fare
    print_banner("9. EMBARKATION PORT & FARE ECONOMICS")
    port_map = {"S": "Southampton (UK)", "C": "Cherbourg (France)", "Q": "Queenstown (Ireland)"}
    emb_group = df.groupby("Embarked")["Survived"].agg(["count", "mean"])
    fare_group = df.groupby("Embarked")["Fare"].median()
    print(f"{'Port of Departure':<24} | {'Count':<8} | {'Median Fare':<12} | {'Survival Rate':<18}")
    print("-" * 70)
    for port, row in emb_group.iterrows():
        pname = port_map.get(port, str(port))
        med_fare = fare_group.get(port, 0.0)
        rate = row["mean"] * 100
        bar = render_ascii_bar(row["mean"], 14)
        print(f"{pname:<24} | {int(row['count']):<8} | £{med_fare:>8.2f}  | {rate:>5.1f}% |{bar}|")

    # 10. Key Takeaways
    print_banner("10. KEY EDA TAKEAWAYS & MACHINE LEARNING IMPLICATIONS")
    print("  • Sex is the single dominant predictive feature: Women survived at ~74.2% vs Men at ~18.9%.")
    print("  • Socioeconomic class creates a stark gradient: 1st class had 2.6x the survival rate of 3rd class.")
    print("  • 1st and 2nd class females had >92% survival, while 3rd class males had only 13.5%.")
    print("  • Small families (2-4 persons) survived at twice the rate of solo travelers or large families (>4).")
    print("  • Title extraction isolates young boys ('Master' = 57.5% survival) from adult men ('Mr' = 15.7%).")
    print("  • Age has ~20% missing values; imputing Age by Title & Pclass groups is far more faithful than")
    print("    global median imputation.")
    print("  • Tree-based ensembles (Random Forest, Gradient Boosting) are uniquely suited to model the")
    print("    sharp non-linear interaction between Sex, Ticket Class, and Family Size.\n")


if __name__ == "__main__":
    main()
