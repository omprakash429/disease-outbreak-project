
"""Helper functions for validating inputs and predicting outbreak transmission."""

import pandas as pd

from src.data_utils import FEATURES

MIN_ILLNESSES = 2


def validate_inputs(options, month, state, etiology, setting, illnesses):
    """Validate the values entered in the prediction form."""
    errors = []
    warnings = []

    try:
        count = float(illnesses)

        if not count.is_integer():
            errors.append("Number of people ill must be a whole number.")
        elif count < MIN_ILLNESSES:
            errors.append(
                f"Number of people ill must be at least {MIN_ILLNESSES}."
            )
        elif count > 1_000_000:
            errors.append("Number of people ill cannot exceed 1,000,000.")

    except (TypeError, ValueError):
        errors.append("Enter a valid number of people ill.")

    values = {
        "Month": month,
        "State": state,
        "Etiology": etiology,
        "Setting": setting,
    }

    for column, value in values.items():
        if value not in options.get(column, []):
            errors.append(f"Please select a valid value for {column}.")

    return errors, warnings


def predict_one(bundle, month, state, etiology, setting, illnesses):
    """Return the predicted transmission mode and its score table."""

    row = pd.DataFrame([{
        "Month": month,
        "State": state,
        "Etiology": etiology,
        "Setting": setting,
        "Illnesses": float(illnesses),
    }])[FEATURES]

    pipe = bundle["pipeline"]
    probabilities = pipe.predict_proba(row)[0]

    table = (
        pd.DataFrame({
            "Transmission mode": pipe.classes_,
            "Probability": probabilities,
        })
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    best_label = table.loc[0, "Transmission mode"]

    return best_label, table
