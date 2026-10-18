"""
Feature Engineering and Data Preparation Pipeline for Titanic Survival Prediction.
Handles honorific extraction, family dynamics, deck identification,
missing value imputation, and preprocessing transformations.
"""

import re
import numpy as np
import pandas as pd


# Canonical honorific categorization
TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Don": "Noble",
    "Sir": "Noble",
    "Lady": "Noble",
    "Countess": "Noble",
    "the Countess": "Noble",
    "Jonkheer": "Noble",
    "Dona": "Noble",
}

NUMERICAL_FEATURES = ["Age", "Fare_Log", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def extract_title_from_name(name: str) -> str:
    """Extract and standardize honorific title from passenger name."""
    if not isinstance(name, str):
        return "Mr"
    match = re.search(r" ([A-Za-z]+)\.", name)
    if match:
        raw_title = match.group(1)
        return TITLE_MAPPING.get(raw_title, "Rare")
    return "Mr"


def extract_deck_from_cabin(cabin: str) -> str:
    """Extract deck letter from cabin identifier or return 'U' for Unknown."""
    if not isinstance(cabin, str) or pd.isna(cabin) or not cabin.strip():
        return "U"
    deck = cabin.strip()[0].upper()
    return deck if deck in ["A", "B", "C", "D", "E", "F", "G"] else "U"


def engineer_features(df: pd.DataFrame, trained_medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Perform complete feature engineering and imputation.
    If trained_medians is None (during model training), computes and returns the fitted statistics.
    If trained_medians is provided (during inference), uses those statistics to prevent leakage.
    """
    data = df.copy()

    # 1. Title Extraction
    if "Title" not in data.columns:
        if "Name" in data.columns:
            data["Title"] = data["Name"].apply(extract_title_from_name)
        else:
            # Default fallback based on sex and age
            def fallback_title(row):
                sex = str(row.get("Sex", "male")).lower()
                age = float(row.get("Age", 30)) if not pd.isna(row.get("Age")) else 30
                if sex == "female":
                    return "Mrs" if age > 25 else "Miss"
                return "Master" if age < 14 else "Mr"
            data["Title"] = data.apply(fallback_title, axis=1)

    # 2. Family Dynamics
    data["SibSp"] = pd.to_numeric(data["SibSp"], errors="coerce").fillna(0).astype(int)
    data["Parch"] = pd.to_numeric(data["Parch"], errors="coerce").fillna(0).astype(int)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # 3. Deck Extraction
    if "Cabin" in data.columns:
        data["Deck"] = data["Cabin"].apply(extract_deck_from_cabin)
    elif "Deck" not in data.columns:
        data["Deck"] = "U"

    # 4. Compute or apply Medians for Fare and Age
    if trained_medians is None:
        pclass_fare_medians = data.groupby("Pclass")["Fare"].median().to_dict()
        global_fare_median = float(data["Fare"].median())
        title_age_medians = data.groupby("Title")["Age"].median().to_dict()
        global_age_median = float(data["Age"].median())

        trained_medians = {
            "pclass_fare": pclass_fare_medians,
            "global_fare": global_fare_median,
            "title_age": title_age_medians,
            "global_age": global_age_median,
        }

    # Apply Fare Imputation & Log transformation
    data["Fare"] = pd.to_numeric(data["Fare"], errors="coerce")
    fare_fallback = data["Pclass"].map(trained_medians.get("pclass_fare", {})).fillna(trained_medians.get("global_fare", 14.45))
    data["Fare"] = data["Fare"].fillna(fare_fallback)
    data["Fare_Log"] = np.log1p(data["Fare"].clip(lower=0))

    # Apply Age Imputation
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    age_fallback = data["Title"].map(trained_medians.get("title_age", {})).fillna(trained_medians.get("global_age", 28.0))
    data["Age"] = data["Age"].fillna(age_fallback)

    # 5. Clean Categoricals
    data["Sex"] = data["Sex"].astype(str).str.lower().str.strip()
    data["Embarked"] = data["Embarked"].astype(str).str.upper().str.strip()
    data["Embarked"] = data["Embarked"].replace({"NAN": "S", "": "S"}).fillna("S")
    data.loc[~data["Embarked"].isin(["S", "C", "Q"]), "Embarked"] = "S"

    data["Pclass"] = pd.to_numeric(data["Pclass"], errors="coerce").fillna(3).astype(int)

    return data[FEATURE_COLUMNS], trained_medians


def build_passenger_record(
    pclass: int,
    sex: str,
    age: float = None,
    sibsp: int = 0,
    parch: int = 0,
    fare: float = None,
    embarked: str = "S",
    title: str = None,
    name: str = None,
    cabin: str = None
) -> pd.DataFrame:
    """Helper to assemble a single passenger record into a DataFrame for inference."""
    clean_sex = str(sex).lower().strip()
    
    # Infer title if not explicitly provided
    if not title:
        if name:
            title = extract_title_from_name(name)
        else:
            if clean_sex == "female":
                title = "Mrs" if (age is not None and age >= 25) else "Miss"
            else:
                title = "Master" if (age is not None and age < 14) else "Mr"

    raw_dict = {
        "Pclass": [int(pclass)],
        "Sex": [clean_sex],
        "Age": [float(age) if age is not None else np.nan],
        "SibSp": [int(sibsp)],
        "Parch": [int(parch)],
        "Fare": [float(fare) if fare is not None else np.nan],
        "Embarked": [str(embarked).upper().strip()],
        "Name": [name if name else ""],
        "Title": [title],
        "Cabin": [cabin if cabin else ""]
    }

    return pd.DataFrame(raw_dict)
