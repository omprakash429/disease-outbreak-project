"""
Shared helper code: loading + cleaning the NORS dataset.

Both train_model.py and app.py use this file so that the cleaning steps
are exactly the same during training and prediction.
"""

import pandas as pd

# ---- What are we predicting? -------------------------------------------
TARGET = "Primary Mode"          # how the outbreak mainly spread

# We keep only the 4 clear transmission categories.
#  - "Indeterminate/unknown" is dropped: it means "nobody knows", not a real answer.
#  - "Environmental contamination..." is dropped: only ~130 rows, too few to learn from.
KEEP_CLASSES = ["Person-to-person", "Food", "Water", "Animal contact"]

# ---- Which columns are allowed as inputs? ------------------------------
# These are things known when an outbreak report is first filed.
#
# NOT used (on purpose, to avoid DATA LEAKAGE):
#   Food Vehicle, Food Contaminated Ingredient, IFSAC Category,
#   Water Exposure, Water Type, Animal Type
#       -> these are only filled in AFTER the mode is known
#          (e.g. Water Exposure exists ONLY when Primary Mode == Water).
#   Hospitalizations, Deaths (+ their "Info On" columns)
#       -> outcomes that are known later, and ~12% missing.
#   Serotype or Genotype, Etiology Status -> mostly missing / not useful for a beginner model.
#   Year -> skipped so the model does not just learn "which era" the report is from.
CATEGORY_FEATURES = ["Month", "State", "Etiology", "Setting"]
NUMBER_FEATURES = ["Illnesses"]
FEATURES = CATEGORY_FEATURES + NUMBER_FEATURES

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

TOP_ETIOLOGIES = 30   # keep the 30 most common germs, group the rest as "Other"
TOP_SETTINGS = 25     # keep the 25 most common places, group the rest as "Other"


def load_raw(csv_path):
    """Read the CSV exactly as it is."""
    return pd.read_csv(csv_path, low_memory=False)


def clean_features(df):
    """Clean the input columns (no target filtering here)."""
    out = pd.DataFrame(index=df.index)

    # Illnesses is stored as text like "3,500" -> convert to a number
    out["Illnesses"] = pd.to_numeric(
        df["Illnesses"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    out["Month"] = df["Month"].astype(int).map(lambda m: MONTH_NAMES[m - 1])
    out["State"] = df["State"].astype(str).str.strip()

    # Some rows list several germs separated by ";" -> keep the first one
    etiology = df["Etiology"].fillna("Not reported").astype(str)
    out["Etiology"] = etiology.str.split(";").str[0].str.strip()

    out["Setting"] = df["Setting"].fillna("Not reported").astype(str).str.strip()
    return out


def prepare_training_data(df):
    """
    Returns (X, y, options)
      X       = cleaned input table
      y       = target labels
      options = the allowed values for each dropdown in the web app
    """
    df = df.drop_duplicates()                       # remove exact duplicate rows
    df = df[df[TARGET].isin(KEEP_CLASSES)]          # keep the 4 classes

    X = clean_features(df)
    y = df[TARGET].copy()

    # Remove rows with impossible / missing illness counts
    good = X["Illnesses"].notna() & (X["Illnesses"] >= 1)
    X, y = X[good].copy(), y[good]

    # Group rare germs and settings into "Other" (keeps the model simple)
    top_e = X["Etiology"].value_counts().head(TOP_ETIOLOGIES).index
    top_s = X["Setting"].value_counts().head(TOP_SETTINGS).index
    X["Etiology"] = X["Etiology"].where(X["Etiology"].isin(top_e), "Other")
    X["Setting"] = X["Setting"].where(X["Setting"].isin(top_s), "Other")

    options = {
        "Month": MONTH_NAMES,
        "State": sorted(X["State"].unique()),
        "Etiology": sorted(X["Etiology"].unique()),
        "Setting": sorted(X["Setting"].unique()),
    }
    return X.reset_index(drop=True), y.reset_index(drop=True), options
