"""Fit the frozen model and create the two validated submission CSVs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DECEMBER_COLUMNS, IDENTIFIER, PREDICTION_COLUMNS, load_december,
    load_development, load_prediction_template, load_validation,
)
from .features import build_features


def prepare_inference(
    frame: pd.DataFrame, version: str = "baseline", weight_treatment: str = "keep"
) -> pd.DataFrame:
    """Apply the same deterministic feature construction used in validation."""
    return build_features(frame, version=version, weight_treatment=weight_treatment)


def check_december_compatibility(
    path: Path = Path("data/december_chart_inputs.csv"),
    version: str = "baseline",
    weight_treatment: str = "keep",
) -> pd.DataFrame:
    """Return model-ready chart features without producing predictions."""
    return prepare_inference(load_december(path), version, weight_treatment)


def _checked_rates(values, label: str) -> np.ndarray:
    rates = np.asarray(values, dtype=float)
    if rates.ndim != 1 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError(f"{label}: predictions must be finite and positive")
    return rates


def validate_validation_output(
    output: pd.DataFrame, template: pd.DataFrame, validation: pd.DataFrame,
) -> np.ndarray:
    if tuple(output.columns) != PREDICTION_COLUMNS or len(output) != len(template):
        raise ValueError("Validation output has the wrong columns or row count")
    if output[IDENTIFIER].isna().any() or output[IDENTIFIER].duplicated().any():
        raise ValueError("Validation output has missing or duplicate load_id values")
    if output[IDENTIFIER].tolist() != template[IDENTIFIER].tolist():
        raise ValueError("Validation output does not preserve template load_id order")
    if set(output[IDENTIFIER]) != set(validation[IDENTIFIER]):
        raise ValueError("Validation output IDs do not match validation.csv")
    return _checked_rates(output["predicted_rate"], "Validation output")


def validate_december_output(output: pd.DataFrame, original: pd.DataFrame) -> np.ndarray:
    if tuple(output.columns) != DECEMBER_COLUMNS or len(output) != len(original):
        raise ValueError("December output has the wrong columns or row count")
    for column in DECEMBER_COLUMNS[:-1]:
        left = output[column]
        right = original[column]
        if column == "date":
            left = pd.to_datetime(left, format="%Y-%m-%d", errors="coerce")
            right = pd.to_datetime(right, format="%Y-%m-%d", errors="coerce")
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise ValueError(f"December output changed the {column} column or row order")
    if len(output) != 31 or output["date"].duplicated().any():
        raise ValueError("December output must have 31 distinct dates")
    return _checked_rates(output["predicted_rate"], "December output")


def _write_checked(frame: pd.DataFrame, path: Path, validate) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    validate(pd.read_csv(temporary))
    temporary.replace(path)


def main() -> None:
    from .train import (
        SELECTED_FEATURE_VERSION, SELECTED_WEIGHT_TREATMENT,
        make_selected_estimator,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train_test.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.csv"))
    parser.add_argument("--template", type=Path,
                        default=Path("data/validation_predictions_template.csv"))
    parser.add_argument("--december", type=Path,
                        default=Path("data/december_chart_inputs.csv"))
    parser.add_argument("--output", type=Path, default=Path("validation_predictions.csv"))
    args = parser.parse_args()

    development = load_development(args.train)
    validation = load_validation(args.validation)
    template = load_prediction_template(args.template)
    december = load_december(args.december)
    if len(development.target) != 48_000:
        raise ValueError("Expected 48,000 labeled training rows")
    if len(validation.raw) != len(template) or set(validation.raw[IDENTIFIER]) != set(template[IDENTIFIER]):
        raise ValueError("Validation and template load_id sets do not match")

    model = make_selected_estimator()
    model.fit(
        prepare_inference(development.features, SELECTED_FEATURE_VERSION,
                          SELECTED_WEIGHT_TREATMENT),
        development.target,
    )
    validation_rates = _checked_rates(model.predict(prepare_inference(
        validation.features, SELECTED_FEATURE_VERSION, SELECTED_WEIGHT_TREATMENT,
    )), "Validation")
    december_rates = _checked_rates(model.predict(prepare_inference(
        december, SELECTED_FEATURE_VERSION, SELECTED_WEIGHT_TREATMENT,
    )), "December")

    by_id = pd.Series(validation_rates, index=validation.raw[IDENTIFIER])
    output = template.copy(deep=True)
    output["predicted_rate"] = output[IDENTIFIER].map(by_id)
    completed_december = december.copy(deep=True)
    completed_december["predicted_rate"] = december_rates
    validate_validation_output(output, template, validation.raw)
    validate_december_output(completed_december, december)

    _write_checked(
        output, args.output,
        lambda loaded: validate_validation_output(loaded, template, validation.raw),
    )
    _write_checked(
        completed_december, args.december,
        lambda loaded: validate_december_output(loaded, december),
    )
    print(f"Trained frozen model on {len(development.target):,} labeled rows")
    for label, rates in (("Validation", validation_rates), ("December", december_rates)):
        print(
            f"{label}: {len(rates):,} rows, min={rates.min():.2f}, "
            f"median={np.median(rates):.2f}, max={rates.max():.2f}, "
            "missing=0, nonfinite=0, nonpositive=0"
        )
    print(f"Wrote {args.output} and {args.december}")


if __name__ == "__main__":
    main()
