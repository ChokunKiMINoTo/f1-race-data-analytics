"""
Feature Engineering Module
Transforms raw lap/qualifying/race data into a feature matrix for ML modeling.
"""
import sys
import os
import warnings

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_collection import load_saved_data

warnings.filterwarnings("ignore")


def compute_practice_features(practice_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-driver, per-race practice features from raw lap data.
    
    Features:
      - fp_best_lap: best lap time across all FP sessions
      - fp2_long_run_avg: avg lap time for long stints (>= 5 laps) in FP2
      - fp_consistency: std dev of lap times (lower = more consistent)
      - fp_avg_speed_i1/i2/st: avg speed trap readings
    """
    if practice_laps.empty:
        return pd.DataFrame()

    # Filter out pit in/out laps and invalid laps
    laps = practice_laps.copy()
    if "LapTime_sec" in laps.columns:
        laps = laps[laps["LapTime_sec"].notna()].copy()
        laps = laps[laps["LapTime_sec"] > 0].copy()

    if laps.empty:
        return pd.DataFrame()

    group_cols = ["Year", "RoundNumber", "Driver"]

    # ── Best lap time across all practice sessions ──
    best_laps = (
        laps.groupby(group_cols)["LapTime_sec"]
        .min()
        .reset_index()
        .rename(columns={"LapTime_sec": "fp_best_lap"})
    )

    # ── FP2 long run pace (stints >= LONG_RUN_MIN_LAPS laps) ──
    fp2 = laps[laps["SessionType"] == "Practice 2"].copy()
    fp2_long_run = pd.DataFrame()
    if not fp2.empty and "Stint" in fp2.columns:
        # Count laps per stint per driver
        stint_lengths = (
            fp2.groupby(group_cols + ["Stint"])["LapTime_sec"]
            .agg(["count", "mean"])
            .reset_index()
        )
        stint_lengths.columns = group_cols + ["Stint", "stint_laps", "stint_avg"]

        # Keep only long stints
        long_stints = stint_lengths[
            stint_lengths["stint_laps"] >= config.LONG_RUN_MIN_LAPS
        ]

        if not long_stints.empty:
            fp2_long_run = (
                long_stints.groupby(group_cols)["stint_avg"]
                .mean()
                .reset_index()
                .rename(columns={"stint_avg": "fp2_long_run_avg"})
            )

    # ── Consistency (std of lap times) ──
    consistency = (
        laps.groupby(group_cols)["LapTime_sec"]
        .std()
        .reset_index()
        .rename(columns={"LapTime_sec": "fp_consistency"})
    )

    # ── Speed trap averages ──
    speed_features = pd.DataFrame()
    speed_cols_map = {
        "SpeedI1": "fp_avg_speed_i1",
        "SpeedI2": "fp_avg_speed_i2",
        "SpeedST": "fp_avg_speed_st",
    }
    available_speed = {k: v for k, v in speed_cols_map.items() if k in laps.columns}
    if available_speed:
        speed_features = (
            laps.groupby(group_cols)[list(available_speed.keys())]
            .mean()
            .reset_index()
            .rename(columns=available_speed)
        )

    # ── Merge all features ──
    features = best_laps
    for df in [fp2_long_run, consistency, speed_features]:
        if not df.empty:
            features = features.merge(df, on=group_cols, how="left")

    return features


def compute_qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-driver qualifying features.

    Features:
      - grid_position: qualifying/starting position
      - quali_best_time: best qualifying lap time (seconds)
      - quali_gap_to_pole: gap to pole position (seconds)
    """
    if qualifying.empty:
        return pd.DataFrame()

    features = qualifying.copy()
    group_cols = ["Year", "RoundNumber", "Driver"]

    cols_to_keep = group_cols + [
        "quali_position",
        "quali_best_time",
        "quali_gap_to_pole",
        "Team",
    ]
    available = [c for c in cols_to_keep if c in features.columns]
    features = features[available].copy()

    # Rename quali_position to grid_position
    if "quali_position" in features.columns:
        features.rename(columns={"quali_position": "grid_position"}, inplace=True)

    return features


def compute_rolling_features(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling average finish positions for drivers and teams.
    Uses the last N races as a window.

    Features:
      - driver_recent_avg_pos: rolling avg finish (last 3 races)
      - team_recent_avg_pos: rolling avg team finish (last 3 races)
    """
    if race_results.empty:
        return pd.DataFrame()

    results = race_results.copy()
    results = results.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

    # Driver rolling average
    driver_rolling = (
        results.groupby("Driver")["race_position"]
        .apply(
            lambda x: x.shift(1).rolling(
                window=config.ROLLING_WINDOW, min_periods=1
            ).mean()
        )
        .reset_index(level=0, drop=True)
    )
    results["driver_recent_avg_pos"] = driver_rolling

    # Team rolling average
    team_avg = results.groupby(["Year", "RoundNumber", "Team"])[
        "race_position"
    ].mean().reset_index()
    team_avg = team_avg.sort_values(["Year", "RoundNumber"])
    team_rolling = (
        team_avg.groupby("Team")["race_position"]
        .apply(
            lambda x: x.shift(1).rolling(
                window=config.ROLLING_WINDOW, min_periods=1
            ).mean()
        )
        .reset_index(level=0, drop=True)
    )
    team_avg["team_recent_avg_pos"] = team_rolling
    team_avg = team_avg[["Year", "RoundNumber", "Team", "team_recent_avg_pos"]]

    results = results.merge(team_avg, on=["Year", "RoundNumber", "Team"], how="left")

    return results[
        ["Year", "RoundNumber", "Driver", "Team", "driver_recent_avg_pos", "team_recent_avg_pos"]
    ]


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode driver, team, and circuit names.
    """
    df = df.copy()

    if "Team" in df.columns:
        team_codes = {name: i for i, name in enumerate(sorted(df["Team"].dropna().unique()))}
        df["team_encoded"] = df["Team"].map(team_codes)

    if "Driver" in df.columns:
        driver_codes = {name: i for i, name in enumerate(sorted(df["Driver"].dropna().unique()))}
        df["driver_encoded"] = df["Driver"].map(driver_codes)

    if "circuit_key" in df.columns:
        circuit_codes = {name: i for i, name in enumerate(sorted(df["circuit_key"].dropna().unique()))}
        df["circuit_encoded"] = df["circuit_key"].map(circuit_codes)

    return df


def build_feature_matrix(years: list = None) -> pd.DataFrame:
    """
    Build the full feature matrix by merging practice, qualifying, race, 
    and weather features for the specified years.
    
    Returns a DataFrame with one row per driver per race, 
    with all features + target (race_position).
    """
    if years is None:
        years = config.ALL_YEARS

    all_dataframes = []

    for year in years:
        print(f"\nBuilding features for {year}...")
        data = load_saved_data(year)

        practice_laps = data["practice_laps"]
        qualifying = data["qualifying"]
        race_results = data["race_results"]
        weather = data["weather"]

        # Compute partial features
        fp_features = compute_practice_features(practice_laps)
        quali_features = compute_qualifying_features(qualifying)

        if race_results.empty:
            print(f"  ⚠ No race results for {year}, skipping")
            continue

        # Base: race results (has the target variable)
        base = race_results[
            [c for c in [
                "Year", "RoundNumber", "EventName", "Driver", "Team",
                "DriverNumber", "race_position", "grid_position",
                "circuit_key", "is_street_circuit",
            ] if c in race_results.columns]
        ].copy()

        # Merge practice features
        if not fp_features.empty:
            base = base.merge(
                fp_features, on=["Year", "RoundNumber", "Driver"], how="left"
            )
            print(f"  ✓ Practice features merged ({len(fp_features)} rows)")

        # Merge qualifying features
        if not quali_features.empty:
            merge_keys = ["Year", "RoundNumber", "Driver"]
            # Only bring in columns not already in base (except merge keys)
            quali_cols = [c for c in quali_features.columns
                          if c not in base.columns or c in merge_keys]
            # If grid_position exists in both, prefer the one from race_results
            if "grid_position" in quali_cols and "grid_position" in base.columns:
                quali_cols = [c for c in quali_cols if c != "grid_position"]
            base = base.merge(
                quali_features[quali_cols],
                on=merge_keys,
                how="left",
            )
            print(f"  ✓ Qualifying features merged ({len(quali_features)} rows)")

        # Merge weather
        if not weather.empty:
            weather_cols = ["Year", "RoundNumber", "track_temp_avg", "rainfall"]
            avail_w = [c for c in weather_cols if c in weather.columns]
            base = base.merge(weather[avail_w], on=["Year", "RoundNumber"], how="left")
            print(f"  ✓ Weather features merged")

        all_dataframes.append(base)
        print(f"  → {len(base)} driver-race entries for {year}")

    if not all_dataframes:
        return pd.DataFrame()

    # Combine all years
    full_df = pd.concat(all_dataframes, ignore_index=True)

    # ── Compute rolling features (needs all years combined & sorted) ──
    full_df = full_df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    rolling = compute_rolling_features(
        full_df[["Year", "RoundNumber", "Driver", "Team", "race_position"]].copy()
    )
    if not rolling.empty:
        full_df = full_df.merge(
            rolling, on=["Year", "RoundNumber", "Driver", "Team"], how="left"
        )
        print(f"  ✓ Rolling features computed")

    # ── Encode categorical features ──
    full_df = encode_categorical_features(full_df)

    # ── Clean up ──
    # Drop rows without target
    full_df = full_df[full_df["race_position"].notna()].reset_index(drop=True)

    print(f"\n✅ Feature matrix complete: {full_df.shape[0]} rows × {full_df.shape[1]} columns")
    print(f"   Features: {[c for c in full_df.columns if c in config.ALL_FEATURES]}")

    return full_df


def save_feature_matrix(df: pd.DataFrame, filename: str = "feature_matrix.parquet"):
    """Save the feature matrix to disk."""
    filepath = os.path.join(config.PROCESSED_DIR, filename)
    df.to_parquet(filepath, index=False)
    print(f"Saved feature matrix to {filepath}")


def load_feature_matrix(filename: str = "feature_matrix.parquet") -> pd.DataFrame:
    """Load the feature matrix from disk."""
    filepath = os.path.join(config.PROCESSED_DIR, filename)
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    df = build_feature_matrix()
    save_feature_matrix(df)
    print(df.head())
    print(df.describe())
