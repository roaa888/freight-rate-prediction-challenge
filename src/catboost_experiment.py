"""Small CatBoost comparison on the existing chronological validation folds."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import load_development
from .features import CATEGORICAL_COLUMNS, LaneCountTransformer, build_features
from .predict import check_december_compatibility
from .train import (
    SEED, SELECTED_FEATURE_VERSION, SELECTED_WEIGHT_TREATMENT,
    make_folds, make_selected_estimator,
)


CONFIGURATIONS = {
    "shallow": {"depth": 4, "learning_rate": 0.05, "iterations": 500,
                "l2_leaf_reg": 3.0},
    "standard": {"depth": 6, "learning_rate": 0.05, "iterations": 500,
                 "l2_leaf_reg": 3.0},
    "regularized": {"depth": 6, "learning_rate": 0.08, "iterations": 350,
                    "l2_leaf_reg": 8.0},
}
PRIMARY_FOLDS = ("Fold 1", "Fold 2")


def fit_catboost(features: pd.DataFrame, target: pd.Series,
                 settings: dict, log_target: bool):
    lane_counts = LaneCountTransformer().fit(features)
    transformed = lane_counts.transform(features)
    model = CatBoostRegressor(
        **settings,
        cat_features=list(CATEGORICAL_COLUMNS),
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    fitted_target = np.log1p(target) if log_target else target
    model.fit(transformed, fitted_target)
    return lane_counts, model


def predict_catboost(fitted, features: pd.DataFrame, log_target: bool) -> np.ndarray:
    lane_counts, model = fitted
    predicted = np.asarray(model.predict(lane_counts.transform(features)))
    if log_target:
        predicted = np.expm1(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError("CatBoost returned non-finite predictions")
    return np.maximum(predicted, 1e-6)


def score(actual: pd.Series, predicted: np.ndarray) -> tuple[float, float]:
    return (
        mean_absolute_error(actual, predicted),
        float(np.sqrt(mean_squared_error(actual, predicted))),
    )


def run_catboost(data, folds, features, chart, settings, log_target,
                 collect_predictions=False):
    rows = []
    predictions = []
    for fold in folds:
        fitted = fit_catboost(
            features.loc[fold.train_rows], data.target.loc[fold.train_rows],
            settings, log_target,
        )
        validation = data.raw.loc[fold.validation_rows]
        predicted = predict_catboost(fitted, features.loc[fold.validation_rows], log_target)
        mae, rmse = score(data.target.loc[fold.validation_rows], predicted)
        rows.append({"fold": fold.name, "mae": mae, "rmse": rmse})
        if collect_predictions and fold.name in PRIMARY_FOLDS:
            predictions.append(category_errors(
                validation, data.raw.loc[fold.train_rows], predicted, "CatBoost",
            ))
        if fold.name == "Latest month":
            chart_predictions = predict_catboost(fitted, chart, log_target)
            if len(chart_predictions) != 31 or not (chart_predictions > 0).all():
                raise ValueError("CatBoost cannot process all December chart rows")
    return pd.DataFrame(rows), pd.concat(predictions) if predictions else None


def run_histgb(data, folds, features, chart, collect_predictions=False):
    rows = []
    predictions = []
    for fold in folds:
        model = make_selected_estimator()
        model.fit(features.loc[fold.train_rows], data.target.loc[fold.train_rows])
        validation = data.raw.loc[fold.validation_rows]
        predicted = np.maximum(model.predict(features.loc[fold.validation_rows]), 1e-6)
        mae, rmse = score(data.target.loc[fold.validation_rows], predicted)
        rows.append({"fold": fold.name, "mae": mae, "rmse": rmse})
        if collect_predictions and fold.name in PRIMARY_FOLDS:
            predictions.append(category_errors(
                validation, data.raw.loc[fold.train_rows], predicted, "HistGB",
            ))
        if fold.name == "Latest month":
            chart_predictions = model.predict(chart)
            if len(chart_predictions) != 31 or not np.isfinite(chart_predictions).all():
                raise ValueError("HistGB cannot process all December chart rows")
    return pd.DataFrame(rows), pd.concat(predictions) if predictions else None


def category_errors(validation: pd.DataFrame, training: pd.DataFrame,
                    predictions: np.ndarray, model_name: str) -> pd.DataFrame:
    lanes = set(zip(training["pickup"], training["delivery"]))
    result = pd.DataFrame(index=validation.index)
    result["model"] = model_name
    result["absolute_error"] = np.abs(
        predictions - validation["posted_rate"].to_numpy()
    )
    result["origin_seen"] = validation["pickup"].isin(set(training["pickup"]))
    result["destination_seen"] = validation["delivery"].isin(set(training["delivery"]))
    result["lane_seen"] = [
        pair in lanes for pair in zip(validation["pickup"], validation["delivery"])
    ]
    return result


def summarize_categories(errors: pd.DataFrame) -> None:
    for category in ("origin_seen", "destination_seen", "lane_seen"):
        summary = errors.groupby(["model", category]).agg(
            rows=("absolute_error", "size"), mae=("absolute_error", "mean"),
        )
        print(f"\n{category}")
        print(summary.to_string(float_format=lambda value: f"{value:.2f}"))


def unseen_city_stress_test(data, folds, features, settings, log_target):
    """Compare models on the same withheld-city Sep-Oct split."""
    fold = next(item for item in folds if item.name == "Fold 2")
    cities = sorted(set(data.raw["pickup"]) | set(data.raw["delivery"]))
    withheld = set(np.random.default_rng(SEED).choice(cities, size=8, replace=False))
    candidate_training = data.raw.loc[fold.train_rows]
    keep = (~candidate_training["pickup"].isin(withheld)
            & ~candidate_training["delivery"].isin(withheld))
    training = candidate_training.loc[keep]
    validation = data.raw.loc[fold.validation_rows]
    x_train = features.loc[training.index]
    x_validation = features.loc[validation.index]
    y_train = data.target.loc[training.index]

    cat = fit_catboost(x_train, y_train, settings, log_target)
    cat_predictions = predict_catboost(cat, x_validation, log_target)
    hist = make_selected_estimator()
    hist.fit(x_train, y_train)
    hist_predictions = np.maximum(hist.predict(x_validation), 1e-6)
    errors = pd.concat([
        category_errors(validation, training, cat_predictions, "CatBoost"),
        category_errors(validation, training, hist_predictions, "HistGB"),
    ])
    print("\nUNSEEN-CITY STRESS TEST (not used for model selection)")
    print("withheld_cities=", sorted(map(str, withheld)),
          "training_rows=", len(training))
    summarize_categories(errors)


def summary_row(label: str, metrics: pd.DataFrame) -> dict:
    primary = metrics[metrics["fold"].isin(PRIMARY_FOLDS)]
    return {
        "model": label,
        "fold_1_mae": metrics.loc[metrics["fold"] == "Fold 1", "mae"].iloc[0],
        "fold_2_mae": metrics.loc[metrics["fold"] == "Fold 2", "mae"].iloc[0],
        "mean_mae": primary["mae"].mean(),
        "mean_rmse": primary["rmse"].mean(),
        "october_mae": metrics.loc[metrics["fold"] == "Latest month", "mae"].iloc[0],
    }


def main() -> None:
    data = load_development()
    folds = make_folds(data.features["date"])
    features = build_features(
        data.features, SELECTED_FEATURE_VERSION, SELECTED_WEIGHT_TREATMENT,
    )
    chart = check_december_compatibility(
        version=SELECTED_FEATURE_VERSION,
        weight_treatment=SELECTED_WEIGHT_TREATMENT,
    )
    results = []
    for name, settings in CONFIGURATIONS.items():
        for log_target in (False, True):
            metrics, _ = run_catboost(
                data, folds, features, chart, settings, log_target,
            )
            label = f"CatBoost {name} {'log1p' if log_target else 'original'}"
            results.append(summary_row(label, metrics))
            print(label, "complete", flush=True)
    comparison = pd.DataFrame(results).sort_values("mean_mae")
    print("\nCATBOOST COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    best_label = comparison.iloc[0]["model"]
    best_name = next(name for name in CONFIGURATIONS if f"CatBoost {name} " in best_label)
    best_log_target = best_label.endswith("log1p")
    best_settings = CONFIGURATIONS[best_name]
    hist_metrics, hist_errors = run_histgb(
        data, folds, features, chart, collect_predictions=True,
    )
    cat_metrics, cat_errors = run_catboost(
        data, folds, features, chart, best_settings, best_log_target,
        collect_predictions=True,
    )
    print("\nBEST CATBOOST VS SELECTED HISTGB")
    print(pd.DataFrame([
        summary_row(best_label, cat_metrics),
        summary_row("HistGB selected log1p", hist_metrics),
    ]).to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nCHRONOLOGICAL FOLDS: SEEN VS UNSEEN")
    summarize_categories(pd.concat([cat_errors, hist_errors]))
    unseen_city_stress_test(data, folds, features, best_settings, best_log_target)
    print("\nDECEMBER COMPATIBILITY: 31 rows processed by best CatBoost; no file saved")


if __name__ == "__main__":
    main()
