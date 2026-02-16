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
]

ROLLING_FEATURES = [
    "driver_recent_avg_pos",
    "team_recent_avg_pos",
]

IDENTITY_FEATURES = [
    "team_encoded",
    "driver_encoded",
]

ALL_FEATURES = (
    PRACTICE_FEATURES
    + QUALIFYING_FEATURES
    + CONTEXT_FEATURES
    + ROLLING_FEATURES
    + IDENTITY_FEATURES
)

TARGET = "race_position"
