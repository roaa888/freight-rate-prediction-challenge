"""Load and validate the supplied assessment data without changing it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "posted_rate"
IDENTIFIER = "load_id"
DEVELOPMENT_COLUMNS = (
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal", TARGET,
)
VALIDATION_COLUMNS = DEVELOPMENT_COLUMNS[:-1]
PREDICTION_COLUMNS = (IDENTIFIER, "predicted_rate")
DECEMBER_COLUMNS = (
    "pickup", "delivery", "distance", "equipment", "weight", "date",
    "predicted_rate",
)


@dataclass(frozen=True)
class DevelopmentData:
    raw: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class ValidationData:
    raw: pd.DataFrame
    features: pd.DataFrame


def _read_checked(path: Path, expected: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing or unexpected or tuple(frame.columns) != expected:
        raise ValueError(
            f"{path}: unexpected schema or column order; "
            f"missing={missing}, unexpected={unexpected}, columns={list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"{path}: empty input")
    return frame


def _parse_dates(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    parsed = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{path}: invalid date values")
    result = frame.copy(deep=True)
    result["date"] = parsed
    return result


def load_development(path: Path = Path("data/train_test.csv")) -> DevelopmentData:
    """Keep an unchanged raw frame and return separate labeled features."""
    raw = _read_checked(path, DEVELOPMENT_COLUMNS)
    parsed = _parse_dates(raw, path)
    target = pd.to_numeric(parsed[TARGET], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError(f"{path}: {TARGET} must be finite and positive")
    if parsed[IDENTIFIER].isna().any() or parsed[IDENTIFIER].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate {IDENTIFIER}")
    features = parsed.drop(columns=[TARGET, IDENTIFIER]).copy(deep=True)
    return DevelopmentData(raw=raw, features=features, target=target.copy())


def load_validation(path: Path = Path("data/validation.csv")) -> ValidationData:
    raw = _read_checked(path, VALIDATION_COLUMNS)
    parsed = _parse_dates(raw, path)
    if parsed[IDENTIFIER].isna().any() or parsed[IDENTIFIER].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate {IDENTIFIER}")
    return ValidationData(
        raw=raw, features=parsed.drop(columns=[IDENTIFIER]).copy(deep=True),
    )


def load_prediction_template(
    path: Path = Path("data/validation_predictions_template.csv"),
) -> pd.DataFrame:
    template = _read_checked(path, PREDICTION_COLUMNS)
    if template[IDENTIFIER].isna().any() or template[IDENTIFIER].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate {IDENTIFIER}")
    return template


def load_december(path: Path = Path("data/december_chart_inputs.csv")) -> pd.DataFrame:
    """Read the original or completed chart inputs without changing row order."""
    raw = _read_checked(path, DECEMBER_COLUMNS)
    return _parse_dates(raw, path)
