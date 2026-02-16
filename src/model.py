"""
Model Module
Trains, tunes, and evaluates ML models for F1 race position prediction.
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

warnings.filterwarnings("ignore")


# ─── Evaluation Metrics ──────────────────────────────────────────────────────

def compute_spearman_per_race(y_true: pd.Series, y_pred: np.ndarray,
                               groups: pd.Series) -> float:
    """
    Compute average Spearman rank correlation per race.
    """
    correlations = []
    for group in groups.unique():
        mask = groups == group
        true = y_true[mask].values
        pred = y_pred[mask]
        if len(true) > 2 and len(np.unique(true)) > 1:
            rho, _ = spearmanr(true, pred)
            if not np.isnan(rho):
                correlations.append(rho)
    return np.mean(correlations) if correlations else 0.0


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    We treat lower race position (= better) as higher relevance.
    """
    max_pos = max(y_true.max(), 20)
    # Convert position to relevance: pos 1 → highest, pos 20 → lowest
    relevance_true = max_pos + 1 - y_true
    relevance_pred = max_pos + 1 - y_pred

    # Sort by predicted order (best predicted first)
    pred_order = np.argsort(y_pred)
    sorted_relevance = relevance_true[pred_order]

    # DCG
    dcg = np.sum(sorted_relevance[:k] / np.log2(np.arange(2, k + 2)))

    # Ideal DCG (sort by actual best)
    ideal_order = np.argsort(-relevance_true)
    ideal_sorted = relevance_true[ideal_order]
    idcg = np.sum(ideal_sorted[:k] / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_top_n_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                            groups: np.ndarray, n: int = 3) -> float:
    """
    For each race, check if the actual top-N finishers are in predicted top-N.
    Returns the average overlap fraction.
    """
    accuracies = []
    for group in np.unique(groups):
        mask = groups == group
        true = y_true[mask]
        pred = y_pred[mask]

        if len(true) < n:
            continue

        true_top_n = set(np.argsort(true)[:n])
        pred_top_n = set(np.argsort(pred)[:n])

        overlap = len(true_top_n & pred_top_n) / n
        accuracies.append(overlap)

    return np.mean(accuracies) if accuracies else 0.0


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                    groups: np.ndarray = None, model_name: str = "Model") -> dict:
    """
    Compute all evaluation metrics for a model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, p_val = spearmanr(y_true, y_pred)

    results = {
        "model": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "spearman_rho": round(rho, 3),
        "spearman_p": round(p_val, 6),
    }

    if groups is not None:
        results["per_race_spearman"] = round(
            compute_spearman_per_race(
                pd.Series(y_true), y_pred, pd.Series(groups)
            ), 3
        )
        results["top3_accuracy"] = round(
            compute_top_n_accuracy(y_true, y_pred, groups, n=3), 3
        )

    # NDCG (on full dataset)
    results["ndcg_10"] = round(compute_ndcg(y_true, y_pred, k=10), 3)

    return results


# ─── Model Training ──────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, feature_cols: list = None):
    """
    Prepare the training data by selecting features and target.
    Handles missing values with median imputation.
    """
    if feature_cols is None:
        feature_cols = config.ALL_FEATURES

    available_features = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing features (will skip): {missing}")

    X = df[available_features].copy()
    y = df[config.TARGET].copy()

    # Median imputation for missing values
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    return X, y, available_features


def train_baseline(X_train, y_train, X_test, y_test, groups_test=None):
    """
    Baseline model: Linear regression using only grid_position.
    """
    if "grid_position" not in X_train.columns:
        print("  ⚠ No grid_position in features, cannot train baseline")
        return None, None

    model = LinearRegression()
    X_tr = X_train[["grid_position"]].copy()
    X_te = X_test[["grid_position"]].copy()

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="Baseline (Grid Position)"
    )
    return model, results


def train_random_forest(X_train, y_train, X_test, y_test,
                         groups_train=None, groups_test=None,
                         tune: bool = True):
    """
    Random Forest Regressor with optional hyperparameter tuning.
    """
    if tune and groups_train is not None:
        print("  Tuning Random Forest...")
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            rf,
            param_distributions=config.RF_PARAM_GRID,
            n_iter=30,
            cv=cv,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train, groups=groups_train)
        model = search.best_estimator_
        print(f"  Best RF params: {search.best_params_}")
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="Random Forest"
    )
    return model, results


def train_xgboost(X_train, y_train, X_test, y_test,
                   groups_train=None, groups_test=None,
                   tune: bool = True):
    """
    XGBoost Regressor with optional hyperparameter tuning.
    """
    if tune and groups_train is not None:
        print("  Tuning XGBoost...")
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
        xgb = XGBRegressor(
            random_state=42, tree_method="hist", verbosity=0
        )
        search = RandomizedSearchCV(
            xgb,
            param_distributions=config.XGB_PARAM_GRID,
            n_iter=30,
            cv=cv,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train, groups=groups_train)
        model = search.best_estimator_
        print(f"  Best XGB params: {search.best_params_}")
    else:
        model = XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, tree_method="hist", verbosity=0
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="XGBoost"
    )
    return model, results


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a trained model.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(1)
    return fi


def train_and_evaluate(feature_matrix: pd.DataFrame = None,
                        tune: bool = True) -> dict:
    """
    Full training and evaluation pipeline.
    Train on 2023-2024, test on 2025.
    """
    if feature_matrix is None:
        from src.feature_engineering import load_feature_matrix
        feature_matrix = load_feature_matrix()

    # Split: train on TRAIN_YEARS, test on TEST_YEARS
    train_mask = feature_matrix["Year"].isin(config.TRAIN_YEARS)
    test_mask = feature_matrix["Year"].isin(config.TEST_YEARS)

    train_df = feature_matrix[train_mask].copy()
    test_df = feature_matrix[test_mask].copy()

    print(f"\nTraining set: {len(train_df)} samples ({config.TRAIN_YEARS})")
    print(f"Test set: {len(test_df)} samples ({config.TEST_YEARS})")

    X_train, y_train, feature_cols = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df, feature_cols)

    # Groups for cross-validation and per-race evaluation
    groups_train = (
        train_df["Year"].astype(str) + "_" + train_df["RoundNumber"].astype(str)
    ).values
    groups_test = (
        test_df["Year"].astype(str) + "_" + test_df["RoundNumber"].astype(str)
    ).values

    all_results = {}

    # 1. Baseline
    print("\n── Training Baseline ──")
    baseline_model, baseline_results = train_baseline(
        X_train, y_train, X_test, y_test, groups_test
    )
    if baseline_results:
        all_results["baseline"] = baseline_results
        print(f"  MAE: {baseline_results['mae']}, "
              f"Spearman: {baseline_results['spearman_rho']}")

    # 2. Random Forest
    print("\n── Training Random Forest ──")
    rf_model, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test,
        groups_train, groups_test, tune=tune
    )
    all_results["random_forest"] = rf_results
    print(f"  MAE: {rf_results['mae']}, Spearman: {rf_results['spearman_rho']}")

    # 3. XGBoost
    print("\n── Training XGBoost ──")
    xgb_model, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test,
        groups_train, groups_test, tune=tune
    )
    all_results["xgboost"] = xgb_results
    print(f"  MAE: {xgb_results['mae']}, Spearman: {xgb_results['spearman_rho']}")

    # Feature importance for best model
    best_model_key = min(all_results, key=lambda k: all_results[k]["mae"])
    best_model = {"baseline": baseline_model, "random_forest": rf_model, "xgboost": xgb_model}[best_model_key]
    fi = get_feature_importance(best_model, feature_cols)

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set - 2025)")
    print("=" * 60)
    comparison = pd.DataFrame(all_results).T
    print(comparison.to_string())

    return {
        "results": all_results,
        "models": {
            "baseline": baseline_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        },
        "feature_importance": fi,
        "feature_cols": feature_cols,
        "predictions": {
            "test_df": test_df,
            "y_pred_rf": rf_model.predict(X_test),
            "y_pred_xgb": xgb_model.predict(X_test),
        },
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print("\nFeature Importance:")
    print(results["feature_importance"])
