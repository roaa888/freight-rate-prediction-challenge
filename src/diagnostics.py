"""Analyze chronological errors and compare small, December-compatible changes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import load_development
from .features import build_features
from .predict import check_december_compatibility
from .train import make_estimator, make_folds


PRIMARY_FOLDS = ("Fold 1", "Fold 2")
TUNING = {
    "slower_240": {"learning_rate": 0.05, "max_iter": 240},
    "faster_160": {"learning_rate": 0.15, "max_iter": 160},
    "small_leaves": {"max_leaf_nodes": 15},
    "large_leaves": {"max_leaf_nodes": 63, "min_samples_leaf": 10},
    "strong_regularization": {"min_samples_leaf": 40, "l2_regularization": 10.0},
    "light_regularization": {"min_samples_leaf": 10, "l2_regularization": 0.0},
}


def evaluate(data, folds, version="baseline", weight_treatment="keep", params=None,
             collect_errors=False):
    features = build_features(data.features, version, weight_treatment)
    chart = check_december_compatibility(
        version=version, weight_treatment=weight_treatment,
    )
    metrics = []
    error_frames = []
    for fold in folds:
        estimator = make_estimator("HistGB", True, version, params)
        estimator.fit(features.loc[fold.train_rows], data.target.loc[fold.train_rows])
        predictions = np.maximum(
            estimator.predict(features.loc[fold.validation_rows]), 1e-6,
        )
        actual = data.target.loc[fold.validation_rows]
        metrics.append({
            "fold": fold.name,
            "mae": mean_absolute_error(actual, predictions),
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
        })
        if collect_errors and fold.name in PRIMARY_FOLDS:
            frame = data.raw.loc[fold.validation_rows, [
                "load_id", "date", "pickup", "delivery", "distance",
                "equipment", "weight", "posted_rate",
            ]].copy()
            frame["fold"] = fold.name
            frame["prediction"] = predictions
            frame["error"] = predictions - actual.to_numpy()
            frame["absolute_error"] = np.abs(frame["error"])
            frame["absolute_pct_error"] = 100 * frame["absolute_error"] / actual.to_numpy()
            train = data.raw.loc[fold.train_rows]
            origin_seen = set(train["pickup"])
            destination_seen = set(train["delivery"])
            lane_seen = set(zip(train["pickup"], train["delivery"]))
            frame["origin_seen"] = frame["pickup"].isin(origin_seen)
            frame["destination_seen"] = frame["delivery"].isin(destination_seen)
            frame["lane_seen"] = [
                pair in lane_seen for pair in zip(frame["pickup"], frame["delivery"])
            ]
            error_frames.append(frame)
        chart_predictions = estimator.predict(chart)
        if not np.isfinite(chart_predictions).all():
            raise ValueError("December feature compatibility failed")
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else None
    return pd.DataFrame(metrics), errors


def primary_score(metrics: pd.DataFrame) -> tuple[float, float]:
    primary = metrics[metrics["fold"].isin(PRIMARY_FOLDS)]
    return primary["mae"].mean(), primary["rmse"].mean()


def group_errors(frame: pd.DataFrame, column: str, minimum: int = 1) -> pd.DataFrame:
    result = frame.groupby(column, observed=True, dropna=False).agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        bias=("error", "mean"),
        median_ape=("absolute_pct_error", "median"),
    )
    return result[result["rows"] >= minimum].sort_values("mae", ascending=False)


def print_error_analysis(data, errors: pd.DataFrame) -> None:
    print("\nNEGATIVE WEIGHT AUDIT")
    negative = data.raw.loc[data.raw["weight"] < 0, "weight"]
    print({
        "count": len(negative),
        "percent": round(100 * len(negative) / len(data.raw), 3),
        "min": float(negative.min()),
        "median": float(negative.median()),
        "max_negative": float(negative.max()),
        "absolute_p95": float(negative.abs().quantile(0.95)),
    })
    errors["weight_status"] = np.select(
        [errors["weight"].isna(), errors["weight"] < 0],
        ["missing", "negative"], default="nonnegative",
    )
    print(group_errors(errors, "weight_status").round(2).to_string())

    print("\nRATE BANDS (defined from actual rate for diagnostics only)")
    errors["rate_band"] = pd.cut(
        errors["posted_rate"], [0, 500, 5000, 10000, np.inf], right=False,
        labels=["under_500", "500_to_5000", "5000_to_10000", "10000_plus"],
    )
    print(group_errors(errors, "rate_band").round(2).to_string())
    extreme = errors["posted_rate"] >= 10000
    print("extreme_share_of_total_absolute_error=",
          round(100 * errors.loc[extreme, "absolute_error"].sum()
                / errors["absolute_error"].sum(), 2))
    print("\nDISTANCE BANDS")
    errors["distance_band"] = pd.cut(
        errors["distance"], [0, 300, 1000, 2000, np.inf], right=False,
        labels=["under_300", "300_to_1000", "1000_to_2000", "2000_plus"],
    )
    print(group_errors(errors, "distance_band").round(2).to_string())
    print("\nEQUIPMENT")
    print(group_errors(errors, "equipment").round(2).to_string())
    print("\nMONTH")
    errors["month"] = pd.to_datetime(errors["date"]).dt.to_period("M").astype(str)
    print(group_errors(errors, "month").sort_index().round(2).to_string())

    print("\nORIGIN AND DESTINATION CITIES (at least 100 rows)")
    for column in ("pickup", "delivery"):
        print(column)
        print(group_errors(errors, column, minimum=100).head(8).round(2).to_string())
    print("\nCITIES ON NON-EXTREME LOADS (at least 100 rows, rate below $10,000)")
    ordinary = errors[errors["posted_rate"] < 10000]
    for column in ("pickup", "delivery"):
        print(column)
        print(group_errors(ordinary, column, minimum=100).head(5).round(2).to_string())
    print("\nLANES (at least 8 rows)")
    errors["lane"] = errors["pickup"] + " -> " + errors["delivery"]
    print(group_errors(errors, "lane", minimum=8).head(10).round(2).to_string())

    print("\nSEEN / UNSEEN IN EACH TRAINING FOLD")
    for column in ("origin_seen", "destination_seen", "lane_seen"):
        print(column)
        print(group_errors(errors, column).round(2).to_string())
    print("\nTOP ABSOLUTE ERRORS")
    print(errors.nlargest(10, "absolute_error")[[
        "load_id", "date", "pickup", "delivery", "distance", "equipment",
        "posted_rate", "prediction", "absolute_error",
    ]].round(2).to_string(index=False))


def print_final_validation_coverage(data) -> None:
    final = pd.read_csv(Path("data/validation.csv"))
    origin = set(data.raw["pickup"])
    destination = set(data.raw["delivery"])
    lanes = set(zip(data.raw["pickup"], data.raw["delivery"]))
    print("\nUNLABELED FINAL VALIDATION COVERAGE")
    print({
        "rows": len(final),
        "unseen_origin_rows": int((~final["pickup"].isin(origin)).sum()),
        "unseen_destination_rows": int((~final["delivery"].isin(destination)).sum()),
        "unseen_city_either_rows": int((~final["pickup"].isin(origin)
                                        | ~final["delivery"].isin(destination)).sum()),
        "unseen_lane_rows": sum(
            pair not in lanes for pair in zip(final["pickup"], final["delivery"])
        ),
    })


def print_unseen_city_stress_test(data, folds) -> None:
    """Simulate novel cities on the Sep-Oct window; do not use for selection."""
    fold = next(item for item in folds if item.name == "Fold 2")
    cities = sorted(set(data.raw["pickup"]) | set(data.raw["delivery"]))
    held_out = set(np.random.default_rng(42).choice(cities, size=8, replace=False))
    training = data.raw.loc[fold.train_rows]
    keep = ~training["pickup"].isin(held_out) & ~training["delivery"].isin(held_out)
    train_rows = training.loc[keep].index
    validation_rows = data.raw.loc[fold.validation_rows].index
    features = build_features(data.features)
    estimator = make_estimator("HistGB", True)
    estimator.fit(features.loc[train_rows], data.target.loc[train_rows])
    predicted = np.maximum(estimator.predict(features.loc[validation_rows]), 1e-6)
    frame = data.raw.loc[validation_rows, ["pickup", "delivery"]].copy()
    frame["absolute_error"] = np.abs(predicted - data.target.loc[validation_rows].to_numpy())
    frame["error"] = predicted - data.target.loc[validation_rows].to_numpy()
    frame["absolute_pct_error"] = (
        100 * frame["absolute_error"] / data.target.loc[validation_rows].to_numpy()
    )
    known_origins = set(data.raw.loc[train_rows, "pickup"])
    known_destinations = set(data.raw.loc[train_rows, "delivery"])
    known_lanes = set(zip(data.raw.loc[train_rows, "pickup"],
                          data.raw.loc[train_rows, "delivery"]))
    frame["origin_seen"] = frame["pickup"].isin(known_origins)
    frame["destination_seen"] = frame["delivery"].isin(known_destinations)
    frame["lane_seen"] = [
        pair in known_lanes for pair in zip(frame["pickup"], frame["delivery"])
    ]
    print("\nSIMULATED UNSEEN-CITY STRESS TEST (not used for model selection)")
    print("held_out_cities=", sorted(map(str, held_out)), "training_rows=", len(train_rows))
    for column in ("origin_seen", "destination_seen", "lane_seen"):
        print(column)
        print(group_errors(frame, column).round(2).to_string())


def print_comparison(label: str, candidates: dict) -> tuple[str, dict]:
    print(f"\n{label}")
    rows = []
    for name, (metrics, _) in candidates.items():
        mean_mae, mean_rmse = primary_score(metrics)
        fold_1 = metrics.loc[metrics["fold"] == "Fold 1", "mae"].iloc[0]
        fold_2 = metrics.loc[metrics["fold"] == "Fold 2", "mae"].iloc[0]
        october = metrics.loc[metrics["fold"] == "Latest month", "mae"].iloc[0]
        rows.append({
            "version": name, "fold_1_mae": fold_1, "fold_2_mae": fold_2,
            "mean_mae": mean_mae, "mean_rmse": mean_rmse, "october_mae": october,
        })
    table = pd.DataFrame(rows).sort_values("mean_mae")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    best = table.iloc[0]["version"]
    return best, table.set_index("version").loc[best].to_dict()


def main() -> None:
    data = load_development()
    folds = make_folds(data.features["date"])
    baseline_metrics, errors = evaluate(data, folds, collect_errors=True)
    print_error_analysis(data, errors)
    print_final_validation_coverage(data)
    print_unseen_city_stress_test(data, folds)

    weight_candidates = {"keep": (baseline_metrics, errors)}
    for treatment in ("missing", "zero"):
        weight_candidates[treatment] = evaluate(data, folds, weight_treatment=treatment)
    best_weight, _ = print_comparison("NEGATIVE WEIGHT TREATMENTS", weight_candidates)

    feature_candidates = {}
    for version in ("baseline", "calendar_bands", "lane_counts", "combined"):
        if version == "baseline":
            feature_candidates[version] = weight_candidates[best_weight]
        else:
            feature_candidates[version] = evaluate(
                data, folds, version=version, weight_treatment=best_weight,
            )
    best_features, _ = print_comparison("FEATURE VERSIONS", feature_candidates)

    tuning_candidates = {"default": feature_candidates[best_features]}
    for name, params in TUNING.items():
        tuning_candidates[name] = evaluate(
            data, folds, version=best_features,
            weight_treatment=best_weight, params=params,
        )
    best_tuning, _ = print_comparison("HISTGRADIENTBOOSTING SETTINGS", tuning_candidates)
    print("\nSELECTED CONFIGURATION")
    print({
        "weight_treatment": best_weight,
        "feature_version": best_features,
        "tuning_version": best_tuning,
        "tuning_params": TUNING.get(best_tuning, {}),
    })


if __name__ == "__main__":
    main()
