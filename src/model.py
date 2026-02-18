"""
Model Module
Trains, tunes, and evaluates ML models for F1 race position prediction.

Models: Baseline (Linear), Random Forest, XGBoost, LightGBM, Stacking Ensemble.
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, rankdata
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# ─── Time-Series Cross-Validation ────────────────────────────────────────────

class TimeSeriesGroupSplit:
    """
    Forward-chaining cross-validation that respects chronological order.
    Uses an expanding window: validation folds only come from the second half
    of the timeline, so each fold always has >= 50% of data for training.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided")

        unique_groups = np.unique(groups)
        # Sort groups chronologically (they're "Year_Round" strings)
        sorted_groups = sorted(unique_groups, key=lambda g: (int(g.split("_")[0]), int(g.split("_")[1])))

        n_groups = len(sorted_groups)

        # Build index lookup
        group_to_idx = {}
        for idx, g in enumerate(groups):
            group_to_idx.setdefault(g, []).append(idx)

        # Only use the second half of groups for validation to ensure min 50% training
        min_train_groups = n_groups // 2
        val_candidates = sorted_groups[min_train_groups:]
        n_val_groups = len(val_candidates)

        actual_splits = min(self.n_splits, n_val_groups)
        if actual_splits < 1:
            actual_splits = 1

        fold_size = max(1, n_val_groups // actual_splits)

        for i in range(actual_splits):
            val_start_idx = i * fold_size
            val_end_idx = min(val_start_idx + fold_size, n_val_groups)
            if i == actual_splits - 1:
                val_end_idx = n_val_groups

            val_groups = val_candidates[val_start_idx:val_end_idx]
            # Train on everything chronologically before the first val group
            first_val_pos = sorted_groups.index(val_groups[0])
            train_groups = sorted_groups[:first_val_pos]

            train_idx = [idx for g in train_groups for idx in group_to_idx.get(g, [])]
            val_idx = [idx for g in val_groups for idx in group_to_idx.get(g, [])]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield np.array(train_idx), np.array(val_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


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


def _build_sample_weights(X_train, groups_train):
    """Build exponential recency weights: later races get higher weight."""
    if groups_train is None:
        return None
    groups = np.asarray(groups_train)
    unique_groups = np.unique(groups)
    # Sort chronologically (year_round format)
    sorted_groups = sorted(unique_groups, key=lambda g: (int(g.split('_')[0]), int(g.split('_')[1])))
    n = len(sorted_groups)
    # Exponential decay: most recent group gets weight ~2x oldest
    group_weights = {g: 1.0 + (i / (n - 1)) for i, g in enumerate(sorted_groups)}
    weights = np.array([group_weights[g] for g in groups])
    return weights


def train_random_forest(X_train, y_train, X_test, y_test,
                         groups_train=None, groups_test=None,
                         tune: bool = True, sample_weight=None):
    """
    Random Forest Regressor with Optuna Bayesian tuning.
    """
    if tune and groups_train is not None:
        print("  Tuning Random Forest (Optuna)...")
        cv = TimeSeriesGroupSplit(n_splits=5)

        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400]),
                'max_depth': trial.suggest_int('max_depth', 4, 14),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            scores = cross_val_score(
                rf, X_train, y_train, cv=cv, groups=groups_train,
                scoring='neg_mean_absolute_error', n_jobs=-1,
                fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
            )
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(rf_objective, n_trials=60, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best RF params: {best_params}")
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        # Use hardcoded best params if available
        if hasattr(config, "RF_BEST_PARAMS"):
            print("  Using best RF params from config")
            model = RandomForestRegressor(
                **config.RF_BEST_PARAMS, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
        model.fit(X_train, y_train, sample_weight=sample_weight)

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
                   tune: bool = True, sample_weight=None):
    """
    XGBoost Regressor with Optuna Bayesian tuning.
    """
    if tune and groups_train is not None:
        print("  Tuning XGBoost (Optuna)...")
        cv = TimeSeriesGroupSplit(n_splits=5)

        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [200, 300, 500, 700]),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
            xgb = XGBRegressor(**params, random_state=42, tree_method='hist', verbosity=0)
            scores = cross_val_score(
                xgb, X_train, y_train, cv=cv, groups=groups_train,
                scoring='neg_mean_absolute_error', n_jobs=-1,
                fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
            )
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(xgb_objective, n_trials=80, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best XGB params: {best_params}")
        model = XGBRegressor(**best_params, random_state=42, tree_method='hist', verbosity=0)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        if hasattr(config, "XGB_BEST_PARAMS"):
            print("  Using best XGB params from config")
            model = XGBRegressor(
                **config.XGB_BEST_PARAMS, random_state=42, tree_method="hist", verbosity=0
            )
        else:
            model = XGBRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42, tree_method="hist", verbosity=0
            )
        model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="XGBoost"
    )
    return model, results


def train_lightgbm(X_train, y_train, X_test, y_test,
                    groups_train=None, groups_test=None,
                    tune: bool = True, sample_weight=None):
    """
    LightGBM Regressor with Optuna Bayesian tuning.
    """
    if tune and groups_train is not None:
        print("  Tuning LightGBM (Optuna)...")
        cv = TimeSeriesGroupSplit(n_splits=5)

        def lgbm_objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [200, 300, 500, 700]),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 127),
            }
            lgbm = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            scores = cross_val_score(
                lgbm, X_train, y_train, cv=cv, groups=groups_train,
                scoring='neg_mean_absolute_error', n_jobs=-1,
                fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
            )
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lgbm_objective, n_trials=80, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best LGBM params: {best_params}")
        model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        if hasattr(config, "LGBM_BEST_PARAMS"):
            print("  Using best LGBM params from config")
            model = LGBMRegressor(
                **config.LGBM_BEST_PARAMS, random_state=42, n_jobs=-1, verbose=-1
            )
        else:
            model = LGBMRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="LightGBM"
    )
    return model, results


def train_stacking_ensemble(
    models: dict,
    X_train, y_train, X_test, y_test,
    groups_train=None, groups_test=None,
    sample_weight=None
):
    """
    Stacking ensemble: uses out-of-fold predictions from base models
    as features for a Ridge regression meta-learner.
    """
    print("  Building stacking ensemble...")

    base_model_keys = [k for k in ["random_forest", "xgboost", "lightgbm"] if k in models]
    if len(base_model_keys) < 2:
        print("  ⚠ Need at least 2 base models for stacking")
        return None, None

    # Generate out-of-fold predictions for training data
    cv = TimeSeriesGroupSplit(n_splits=5)
    oof_preds = np.zeros((len(X_train), len(base_model_keys)))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        for i, key in enumerate(base_model_keys):
            model_cls = {
                "random_forest": lambda: RandomForestRegressor(
                    **{k: v for k, v in models[key].get_params().items()
                       if k in RandomForestRegressor().get_params()},
                ),
                "xgboost": lambda: XGBRegressor(
                    **{k: v for k, v in models[key].get_params().items()
                       if k in XGBRegressor().get_params()},
                ),
                "lightgbm": lambda: LGBMRegressor(
                    **{k: v for k, v in models[key].get_params().items()
                       if k in LGBMRegressor().get_params()},
                ),
            }[key]()

            model_cls.fit(X_fold_train, y_fold_train)
            oof_preds[val_idx, i] = model_cls.predict(X_fold_val)

    # Handle any rows with zero predictions (from incomplete folds)
    for i, key in enumerate(base_model_keys):
        zero_mask = oof_preds[:, i] == 0
        if zero_mask.any():
            oof_preds[zero_mask, i] = models[key].predict(
                X_train.iloc[zero_mask.nonzero()[0]]
            )

    # Train meta-learner on OOF predictions with CV-tuned alpha (Positive constraint)
    from sklearn.linear_model import Lasso
    best_alpha, best_score = 0.0001, float('inf')
    # Lasso requires very small alpha to behave like LS with constraints
    for alpha in [1e-5, 1e-4, 1e-3, 0.01, 0.1]:
        # positive=True enforces non-negative weights
        lasso = Lasso(alpha=alpha, positive=True, random_state=42)
        cv_inner = TimeSeriesGroupSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in cv_inner.split(oof_preds, y_train, groups_train):
            # Inner CV also needs to handle weights if provided
            sw_train = sample_weight[tr_idx] if sample_weight is not None else None
            sw_val = sample_weight[val_idx] if sample_weight is not None else None
            
            lasso.fit(oof_preds[tr_idx], y_train.iloc[tr_idx], sample_weight=sw_train)
            pred = lasso.predict(oof_preds[val_idx])
            
            # Weighted MAE for selection
            if sw_val is not None:
                score = np.average(np.abs(y_train.iloc[val_idx] - pred), weights=sw_val)
            else:
                score = mean_absolute_error(y_train.iloc[val_idx], pred)
            scores.append(score)
            
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha
            
    print(f"  Best Lasso (pos) alpha: {best_alpha}")
    meta_model = Lasso(alpha=best_alpha, positive=True, random_state=42)
    meta_model.fit(oof_preds, y_train, sample_weight=sample_weight)

    # Generate test predictions
    test_base_preds = np.column_stack([
        models[key].predict(X_test) for key in base_model_keys
    ])
    y_pred = meta_model.predict(test_base_preds)
    y_pred = np.clip(y_pred, 1, 20)

    results = evaluate_model(
        y_test.values, y_pred,
        groups=groups_test,
        model_name="Stacking Ensemble"
    )

    weights = dict(zip(base_model_keys, meta_model.coef_.round(3)))
    print(f"  Stacking weights: {weights}")

    return meta_model, results


def select_features(model, X_train, y_train, feature_names, groups_train=None):
    """
    Permutation importance-based feature selection.
    Drops features with near-zero or negative importance.
    """
    print("  Running permutation importance...")
    perm_result = permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Keep features with positive importance (> threshold)
    threshold = 0.0
    selected = importance_df[importance_df["importance_mean"] > threshold]["feature"].tolist()
    dropped = importance_df[importance_df["importance_mean"] <= threshold]["feature"].tolist()

    if dropped:
        print(f"  Dropping {len(dropped)} low-importance features: {dropped}")
    print(f"  Keeping {len(selected)} features")

    return selected, importance_df


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


def rank_predictions_per_race(y_pred, groups):
    """Convert raw predictions to rank order within each race."""
    y_ranked = y_pred.copy()
    for race_id in np.unique(groups):
        mask = groups == race_id
        y_ranked[mask] = rankdata(y_pred[mask], method='ordinal')
    return y_ranked


def train_and_evaluate(feature_matrix: pd.DataFrame = None,
                        tune: bool = True) -> dict:
    """
    Full training and evaluation pipeline.
    Train on 2023-2024, test on 2025.
    Includes: Optuna tuning, recency weighting, stacking, rank post-processing.
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

    print(f"Total features: {len(feature_cols)}")

    # Groups for cross-validation and per-race evaluation
    groups_train = (
        train_df["Year"].astype(str) + "_" + train_df["RoundNumber"].astype(str)
    ).values
    groups_test = (
        test_df["Year"].astype(str) + "_" + test_df["RoundNumber"].astype(str)
    ).values

    # Recency-weighted samples (2024 races weighted ~2x vs 2023)
    sample_weight = _build_sample_weights(X_train, groups_train)

    all_results = {}
    all_models = {}

    # 1. Baseline
    print("\n── Training Baseline ──")
    baseline_model, baseline_results = train_baseline(
        X_train, y_train, X_test, y_test, groups_test
    )
    if baseline_results:
        all_results["baseline"] = baseline_results
        all_models["baseline"] = baseline_model
        print(f"  MAE: {baseline_results['mae']}, "
              f"Spearman: {baseline_results['spearman_rho']}")

    # 2. Random Forest
    print("\n── Training Random Forest ──")
    rf_model, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test,
        groups_train, groups_test, tune=tune, sample_weight=sample_weight
    )
    all_results["random_forest"] = rf_results
    all_models["random_forest"] = rf_model
    print(f"  MAE: {rf_results['mae']}, Spearman: {rf_results['spearman_rho']}")

    # 3. XGBoost
    print("\n── Training XGBoost ──")
    xgb_model, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test,
        groups_train, groups_test, tune=tune, sample_weight=sample_weight
    )
    all_results["xgboost"] = xgb_results
    all_models["xgboost"] = xgb_model
    print(f"  MAE: {xgb_results['mae']}, Spearman: {xgb_results['spearman_rho']}")

    # 4. LightGBM
    print("\n── Training LightGBM ──")
    lgbm_model, lgbm_results = train_lightgbm(
        X_train, y_train, X_test, y_test,
        groups_train, groups_test, tune=tune, sample_weight=sample_weight
    )
    all_results["lightgbm"] = lgbm_results
    all_models["lightgbm"] = lgbm_model
    print(f"  MAE: {lgbm_results['mae']}, Spearman: {lgbm_results['spearman_rho']}")

    # 5. Stacking Ensemble
    print("\n── Training Stacking Ensemble ──")
    stack_model, stack_results = train_stacking_ensemble(
        all_models, X_train, y_train, X_test, y_test,
        groups_train, groups_test, sample_weight=sample_weight
    )
    if stack_results:
        all_results["stacking"] = stack_results
        all_models["stacking"] = stack_model
        print(f"  MAE: {stack_results['mae']}, Spearman: {stack_results['spearman_rho']}")

    # 6. Weighted Average Ensemble (inverse-MAE weights)
    print("\n── Training Weighted Average Ensemble ──")
    base_keys = ["random_forest", "xgboost", "lightgbm"]
    base_maes = {k: all_results[k]["mae"] for k in base_keys if k in all_results}
    if base_maes:
        # Weights proportional to 1/MAE
        inv_maes = {k: 1.0 / v for k, v in base_maes.items()}
        total_inv = sum(inv_maes.values())
        weights = {k: v / total_inv for k, v in inv_maes.items()}

        y_pred_wavg = sum(
            weights[k] * np.clip(all_models[k].predict(X_test), 1, 20)
            for k in weights
        )
        y_pred_wavg = np.clip(y_pred_wavg, 1, 20)

        wavg_results = evaluate_model(
            y_test.values, y_pred_wavg,
            groups=groups_test,
            model_name="Weighted Average"
        )
        all_results["weighted_avg"] = wavg_results
        print(f"  Weights: {{{', '.join(f'{k}: {v:.3f}' for k, v in weights.items())}}}")
        print(f"  MAE: {wavg_results['mae']}, Spearman: {wavg_results['spearman_rho']}")

    # Feature importance for best individual model
    non_ensemble_keys = {"baseline", "stacking", "weighted_avg"}
    best_model_key = min(
        {k: v for k, v in all_results.items() if k not in non_ensemble_keys},
        key=lambda k: all_results[k]["mae"]
    )
    best_model = all_models[best_model_key]
    fi = get_feature_importance(best_model, feature_cols) if best_model_key != "stacking" else pd.DataFrame()

    # Feature selection analysis (informational only)
    if best_model_key != "stacking":
        selected_features, perm_importance = select_features(
            best_model, X_train, y_train, feature_cols, groups_train
        )
    else:
        selected_features = feature_cols
        perm_importance = pd.DataFrame()

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set - 2025)")
    print("=" * 60)
    comparison = pd.DataFrame(all_results).T
    print(comparison.to_string())

    # Build test predictions for all models
    predictions = {"test_df": test_df}
    for key, model in all_models.items():
        if key == "baseline":
            predictions[f"y_pred_{key}"] = model.predict(X_test[["grid_position"]])
        elif key == "stacking":
            base_preds = np.column_stack([
                all_models[k].predict(X_test)
                for k in ["random_forest", "xgboost", "lightgbm"]
                if k in all_models
            ])
            predictions[f"y_pred_{key}"] = np.clip(model.predict(base_preds), 1, 20)
        else:
            predictions[f"y_pred_{key}"] = np.clip(model.predict(X_test), 1, 20)

    return {
        "results": all_results,
        "models": all_models,
        "feature_importance": fi,
        "perm_importance": perm_importance,
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "predictions": predictions,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print("\nFeature Importance:")
    print(results["feature_importance"])
