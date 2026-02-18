"""
Configuration for F1 Race Position Prediction Project.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── Season Configuration ────────────────────────────────────────────────────
TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]
ALL_YEARS = TRAIN_YEARS + TEST_YEARS

# ─── Session Types ────────────────────────────────────────────────────────────
PRACTICE_SESSIONS = ["Practice 1", "Practice 2", "Practice 3"]
QUALIFYING_SESSION = "Qualifying"
RACE_SESSION = "Race"

# ─── Feature Engineering Constants ────────────────────────────────────────────
# Fuel correction: ~0.03 seconds per kg per lap
FUEL_PENALTY_PER_KG = 0.03  # seconds
FUEL_BURN_RATE_PER_LAP = 1.7  # kg/lap

# Minimum stint length to be considered a "long run" in practice
LONG_RUN_MIN_LAPS = 5

# Rolling average window for recent form features
ROLLING_WINDOW = 3

# ─── Street Circuits (for the is_street_circuit feature) ──────────────────────
STREET_CIRCUITS = [
    "Monaco",
    "Singapore",
    "Jeddah",
    "Baku",
    "Las Vegas",
    "Melbourne",  # Semi-street
]

# ─── Model Hyperparameter Grids ──────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [5, 10, 20, 30],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "num_leaves": [15, 31, 63, 127],
}

# ─── Best Hyperparameters (Found via Optuna) ──────────────────────────────────
# Hardcoded to ensure reproducibility and skip tuning time.
RF_BEST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 14,
    'min_samples_split': 6,
    'min_samples_leaf': 5,
    'max_features': 'sqrt'
}

XGB_BEST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.01594573996384261,
    'subsample': 0.6154897603081889,
    'colsample_bytree': 0.8992357456867728,
    'min_child_weight': 9,
    'reg_alpha': 0.0015914915831927807,
    'reg_lambda': 6.476489920093153
}

LGBM_BEST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.009689157979499551,
    'subsample': 0.900327718178962,
    'colsample_bytree': 0.6836597766853224,
    'min_child_samples': 18,
    'reg_alpha': 0.0020260075386764466,
    'reg_lambda': 2.5735875032885382,
    'num_leaves': 96
}

# ─── Feature Columns ─────────────────────────────────────────────────────────
PRACTICE_FEATURES = [
    "fp_best_lap",
    "fp2_long_run_avg",
    "fp_consistency",
    "fp_avg_speed_i1",
    "fp_avg_speed_i2",
    "fp_avg_speed_st",
]

QUALIFYING_FEATURES = [
    "grid_position",
    "quali_best_time",
    "quali_gap_to_pole",
]

CONTEXT_FEATURES = [
    "circuit_encoded",
    "is_street_circuit",
    "track_temp_avg",
    "rainfall",
    "humidity_avg",
    "wind_speed_avg",
]

ROLLING_FEATURES = [
    "driver_recent_avg_pos",
    "team_recent_avg_pos",
]

IDENTITY_FEATURES = [
    "team_encoded",
    "driver_encoded",
]

INTERACTION_FEATURES = [
    "grid_vs_recent_form",
    "fp_race_pace_delta",
    "quali_position_change",
    "driver_consistency_score",
    "team_quali_avg",
    "position_delta_potential",
    "sector_speed_rank_i1",
    "sector_speed_rank_i2",
]

TARGET_ENC_FEATURES = [
    "driver_target_enc",
    "team_target_enc",
    "circuit_target_enc",
]

TIRE_FEATURES = [
    "fp_n_compounds_used",
    "fp_soft_pct",
    "fp_stint_count",
    "fp_avg_tyre_life",
]

CIRCUIT_HISTORY_FEATURES = [
    "driver_circuit_avg_pos",
    "team_circuit_avg_pos",
    "driver_circuit_best_pos",
]

DNF_FEATURES = [
    "driver_dnf_rate",
    "team_dnf_rate",
]

PHYSICS_FEATURES = [
    "fp_fuel_corrected_pace",
    "fp_tire_deg_gradient",
    "fp_top_speed_max",
    "constructor_points",
    "teammate_quali_gap",
]

ALL_FEATURES = (
    PRACTICE_FEATURES
    + QUALIFYING_FEATURES
    + CONTEXT_FEATURES
    + ROLLING_FEATURES
    + IDENTITY_FEATURES
    + INTERACTION_FEATURES
    + TIRE_FEATURES
    + CIRCUIT_HISTORY_FEATURES
    + DNF_FEATURES
)

RANDOM_SEED = 42

TARGET = "race_position"
