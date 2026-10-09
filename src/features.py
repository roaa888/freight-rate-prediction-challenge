"""Features shared by training, validation, and December chart inference."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


SOURCE_COLUMNS = ("pickup", "delivery", "distance", "equipment", "weight", "date")
NUMERIC_COLUMNS = ("distance", "weight", "month", "week_of_year", "day_of_week")
CATEGORICAL_COLUMNS = ("pickup", "delivery", "equipment")
MODEL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_VERSIONS = ("baseline", "calendar_bands", "lane_counts", "combined")
WEIGHT_TREATMENTS = ("keep", "missing", "zero")


def feature_columns(version: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if version not in FEATURE_VERSIONS:
        raise ValueError(f"Unknown feature version: {version}")
    numeric = NUMERIC_COLUMNS
    categorical = CATEGORICAL_COLUMNS
    if version in ("calendar_bands", "combined"):
        numeric += ("day_of_month", "is_weekend")
        categorical += ("distance_band", "weight_band")
    if version in ("lane_counts", "combined"):
        numeric += ("lane_count", "reverse_lane_count")
    return numeric, categorical


class LaneCountTransformer(BaseEstimator, TransformerMixin):
    """Add route frequencies learned from the current training fold only."""

    def fit(self, frame: pd.DataFrame, target=None):
        self.counts_ = frame.groupby(["pickup", "delivery"]).size().to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["lane_count"] = [
            self.counts_.get((pickup, delivery), 0)
            for pickup, delivery in zip(frame["pickup"], frame["delivery"])
        ]
        result["reverse_lane_count"] = [
            self.counts_.get((delivery, pickup), 0)
            for pickup, delivery in zip(frame["pickup"], frame["delivery"])
        ]
        return result


def build_features(
    frame: pd.DataFrame,
    version: str = "baseline",
    weight_treatment: str = "keep",
) -> pd.DataFrame:
    if weight_treatment not in WEIGHT_TREATMENTS:
        raise ValueError(f"Unknown weight treatment: {weight_treatment}")
    numeric_columns, categorical_columns = feature_columns(version)
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")

    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Model inputs contain invalid dates")

    result = pd.DataFrame(index=frame.index)
    for column in ("distance", "weight"):
        result[column] = pd.to_numeric(frame[column], errors="raise")
    if weight_treatment == "missing":
        result["weight"] = result["weight"].mask(result["weight"] < 0)
    elif weight_treatment == "zero":
        result["weight"] = result["weight"].clip(lower=0)
    result["month"] = dates.dt.month
    # ISO week rolls into the next year on 2025-12-29; keep December continuous.
    result["week_of_year"] = dates.dt.strftime("%W").astype(int)
    result["day_of_week"] = dates.dt.dayofweek
    for column in CATEGORICAL_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Model inputs contain missing {column}")
        result[column] = frame[column].astype(str)
    if version in ("calendar_bands", "combined"):
        result["day_of_month"] = dates.dt.day
        result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        result["distance_band"] = pd.cut(
            result["distance"], [-np.inf, 300, 600, 1200, 2000, np.inf],
            labels=["short", "regional", "medium", "long", "very_long"],
        ).astype(str)
        result["weight_band"] = pd.cut(
            result["weight"], [-np.inf, 0, 10000, 20000, 30000, 40000, np.inf],
            labels=["nonpositive", "light", "moderate", "medium", "heavy", "very_heavy"],
        ).astype(str)
    # Lane counts are inserted by a fitted transformer inside the model pipeline.
    return result.loc[:, tuple(c for c in numeric_columns + categorical_columns
                               if c not in ("lane_count", "reverse_lane_count"))]
