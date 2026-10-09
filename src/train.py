"""Run deterministic, chronological freight rate baseline experiments."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import load_development
from .features import (
    MODEL_COLUMNS, LaneCountTransformer, build_features, feature_columns,
)
from .predict import check_december_compatibility


SEED = 42
SELECTED_FEATURE_VERSION = "lane_counts"
SELECTED_WEIGHT_TREATMENT = "keep"
SELECTED_HGB_PARAMS = {"max_leaf_nodes": 15}


@dataclass(frozen=True)
class Fold:
    name: str
    train_rows: np.ndarray
    validation_rows: np.ndarray


def make_folds(dates: pd.Series) -> list[Fold]:
    """Use the last four observed months as two successive forecast windows."""
    months = pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M")
    observed = set(dates.dt.to_period("M"))
    if len(months) < 8 or set(months) != observed:
        raise ValueError("Rolling validation needs at least eight consecutive observed months")
    row_months = dates.dt.to_period("M")
    folds = [
        Fold("Fold 1", (row_months <= months[-5]).to_numpy(),
             row_months.isin(months[-4:-2]).to_numpy()),
        Fold("Fold 2", (row_months <= months[-3]).to_numpy(),
             row_months.isin(months[-2:]).to_numpy()),
        Fold("Latest month", (row_months < months[-1]).to_numpy(),
             (row_months == months[-1]).to_numpy()),
    ]
    for fold in folds:
        if not fold.train_rows.any() or not fold.validation_rows.any():
            raise ValueError(f"{fold.name}: empty training or validation period")
        if (fold.train_rows & fold.validation_rows).any():
            raise ValueError(f"{fold.name}: overlapping train and validation rows")
        if dates.loc[fold.train_rows].max() >= dates.loc[fold.validation_rows].min():
            raise ValueError(f"{fold.name}: training reaches the validation period")
    return folds


def make_estimator(
    model_name: str,
    log_target: bool,
    feature_version: str = "baseline",
    hgb_params: dict | None = None,
):
    numeric_columns, categorical_columns = feature_columns(feature_version)
    if model_name == "Ridge":
        preprocessing = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), list(numeric_columns)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"),
                 list(categorical_columns)),
            ]
        )
        regressor = Ridge(alpha=10.0, solver="lsqr")
    elif model_name == "HistGB":
        preprocessing = ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), list(numeric_columns)),
                ("categorical", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan,
                ), list(categorical_columns)),
            ]
        )
        settings = dict(
            learning_rate=0.1, max_iter=120, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
        )
        if hgb_params:
            settings.update(hgb_params)
        regressor = HistGradientBoostingRegressor(
            **settings,
            categorical_features=[False] * len(numeric_columns)
            + [True] * len(categorical_columns),
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps = []
    if feature_version in ("lane_counts", "combined"):
        steps.append(("lane_counts", LaneCountTransformer()))
    steps.extend([("preprocess", preprocessing), ("model", regressor)])
    pipeline = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1,
        )
    return pipeline


def make_selected_estimator():
    """Return the leading chronological-validation model without fitting it."""
    return make_estimator(
        "HistGB", True, SELECTED_FEATURE_VERSION, SELECTED_HGB_PARAMS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train_test.csv"))
    parser.add_argument("--december", type=Path, default=Path("data/december_chart_inputs.csv"))
    args = parser.parse_args()

    development = load_development(args.data)
    features = build_features(development.features)
    december_features = check_december_compatibility(args.december)
    if tuple(features.columns) != MODEL_COLUMNS or tuple(december_features.columns) != MODEL_COLUMNS:
        raise ValueError("December model features do not match training features")

    folds = make_folds(development.features["date"])
    for fold in folds:
        train_dates = development.features.loc[fold.train_rows, "date"]
        validation_dates = development.features.loc[fold.validation_rows, "date"]
        print(
            f"{fold.name}: train={fold.train_rows.sum():,} "
            f"({train_dates.min().date()} to {train_dates.max().date()}), "
            f"validate={fold.validation_rows.sum():,} "
            f"({validation_dates.min().date()} to {validation_dates.max().date()})"
        )
    print(f"December compatibility: {len(december_features)} rows, {len(MODEL_COLUMNS)} model features")
    print("Metric for selection: MAE (dollars); RMSE highlights large errors.")

    results = []
    for model_name in ("Ridge", "HistGB"):
        for log_target in (False, True):
            for fold in folds:
                estimator = make_estimator(model_name, log_target)
                estimator.fit(features.loc[fold.train_rows], development.target.loc[fold.train_rows])
                predictions = np.maximum(
                    np.asarray(estimator.predict(features.loc[fold.validation_rows])), 1e-6,
                )
                actual = development.target.loc[fold.validation_rows]
                results.append({
                    "model": model_name,
                    "target": "log1p" if log_target else "original",
                    "fold": fold.name,
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                    "r2": r2_score(actual, predictions),
                })
                # Exercise the fitted preprocessing path on the separate chart schema.
                if fold.name == "Latest month":
                    chart_check = estimator.predict(december_features)
                    if not np.isfinite(chart_check).all():
                        raise ValueError("Non-finite December compatibility predictions")

    metrics = pd.DataFrame(results)
    print("\nPer-fold results")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    primary = metrics[metrics["fold"].isin(("Fold 1", "Fold 2"))]
    comparison = primary.pivot(index=["model", "target"], columns="fold", values="mae")
    comparison["Mean MAE"] = primary.groupby(["model", "target"])["mae"].mean()
    comparison["Mean RMSE"] = primary.groupby(["model", "target"])["rmse"].mean()
    print("\nPrimary rolling comparison")
    print(comparison.sort_values("Mean MAE").to_string(float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
