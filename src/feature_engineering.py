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


def compute_tire_features(practice_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tire strategy features from practice lap data.

    Features:
      - fp_n_compounds_used: distinct compounds tested in practice
      - fp_soft_pct: fraction of laps on SOFT compound
      - fp_stint_count: number of stints (run plan complexity)
      - fp_avg_tyre_life: average tyre life across stints
    """
    if practice_laps.empty:
        return pd.DataFrame()

    laps = practice_laps.copy()
    group_cols = ["Year", "RoundNumber", "Driver"]

    features_list = []

    # Number of distinct compounds used
    if "Compound" in laps.columns:
        compound_counts = (
            laps[laps["Compound"].notna() & ~laps["Compound"].isin(["None", "nan", "UNKNOWN", "TEST_UNKNOWN"])]
            .groupby(group_cols)["Compound"]
            .nunique()
            .reset_index()
            .rename(columns={"Compound": "fp_n_compounds_used"})
        )
        features_list.append(compound_counts)

        # Fraction on SOFT
        def soft_pct(group):
            valid = group[group.notna() & ~group.isin(["None", "nan", "UNKNOWN", "TEST_UNKNOWN"])]
            if len(valid) == 0:
                return 0.0
            return (valid == "SOFT").sum() / len(valid)

        soft_fracs = (
            laps.groupby(group_cols)["Compound"]
            .apply(soft_pct)
            .reset_index()
            .rename(columns={"Compound": "fp_soft_pct"})
        )
        features_list.append(soft_fracs)

    # Number of stints
    if "Stint" in laps.columns:
        stint_counts = (
            laps.groupby(group_cols)["Stint"]
            .nunique()
            .reset_index()
            .rename(columns={"Stint": "fp_stint_count"})
        )
        features_list.append(stint_counts)

    # Average tyre life
    if "TyreLife" in laps.columns:
        avg_tyre_life = (
            laps.groupby(group_cols)["TyreLife"]
            .mean()
            .reset_index()
            .rename(columns={"TyreLife": "fp_avg_tyre_life"})
        )
        features_list.append(avg_tyre_life)

    if not features_list:
        return pd.DataFrame()

    result = features_list[0]
    for df in features_list[1:]:
        result = result.merge(df, on=group_cols, how="outer")

    return result


def compute_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-driver and per-team historical performance at each circuit.
    Uses only prior races (shifted) to avoid leakage.

    Features:
      - driver_circuit_avg_pos: driver's rolling avg finish at this circuit
      - team_circuit_avg_pos: team's rolling avg finish at this circuit
      - driver_circuit_best_pos: driver's best-ever finish at this circuit
    """
    df = df.copy()
    df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

    if "circuit_key" not in df.columns or "race_position" not in df.columns:
        return df

    # Driver circuit history (shifted to exclude current race)
    driver_circuit_avg = (
        df.groupby(["Driver", "circuit_key"])["race_position"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["driver_circuit_avg_pos"] = driver_circuit_avg

    # Driver best at circuit (shifted)
    driver_circuit_best = (
        df.groupby(["Driver", "circuit_key"])["race_position"]
        .apply(lambda x: x.shift(1).expanding().min())
        .reset_index(level=[0, 1], drop=True)
    )
    df["driver_circuit_best_pos"] = driver_circuit_best

    # Team circuit history (avg of both drivers, shifted)
    team_race_avg = df.groupby(["Year", "RoundNumber", "Team", "circuit_key"])[
        "race_position"
    ].mean().reset_index()
    team_race_avg = team_race_avg.sort_values(["Year", "RoundNumber"])
    team_circuit_avg = (
        team_race_avg.groupby(["Team", "circuit_key"])["race_position"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    team_race_avg["team_circuit_avg_pos"] = team_circuit_avg
    df = df.merge(
        team_race_avg[["Year", "RoundNumber", "Team", "circuit_key", "team_circuit_avg_pos"]],
        on=["Year", "RoundNumber", "Team", "circuit_key"],
        how="left",
    )

    return df


def compute_dnf_features(df: pd.DataFrame, race_results_all: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling DNF rate per driver and team.
    Only Retired/Did not start/Disqualified count as DNFs (not lapped cars).

    Features:
      - driver_dnf_rate: rolling DNF rate over last 10 races
      - team_dnf_rate: rolling team DNF rate over last 10 races
    """
    df = df.copy()

    if "Status" not in race_results_all.columns:
        return df

    rr = race_results_all.copy()
    rr = rr.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

    # DNF flag: only Retired, Did not start, Disqualified
    dnf_statuses = ["Retired", "Did not start", "Disqualified"]
    rr["is_dnf"] = rr["Status"].isin(dnf_statuses).astype(int)

    # Driver rolling DNF rate (last 10 races, shifted)
    driver_dnf = (
        rr.groupby("Driver")["is_dnf"]
        .apply(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rr["driver_dnf_rate"] = driver_dnf

    # Team rolling DNF rate
    team_dnf_avg = rr.groupby(["Year", "RoundNumber", "Team"])["is_dnf"].mean().reset_index()
    team_dnf_avg = team_dnf_avg.sort_values(["Year", "RoundNumber"])
    team_dnf = (
        team_dnf_avg.groupby("Team")["is_dnf"]
        .apply(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    team_dnf_avg["team_dnf_rate"] = team_dnf

    # Merge driver DNF rate
    dnf_merge_cols = ["Year", "RoundNumber", "Driver"]
    avail = [c for c in dnf_merge_cols + ["driver_dnf_rate"] if c in rr.columns]
    df = df.merge(rr[avail].drop_duplicates(subset=dnf_merge_cols), on=dnf_merge_cols, how="left")

    # Merge team DNF rate
    team_merge_cols = ["Year", "RoundNumber", "Team"]
    df = df.merge(
        team_dnf_avg[team_merge_cols + ["team_dnf_rate"]].drop_duplicates(subset=team_merge_cols),
        on=team_merge_cols,
        how="left",
    )

    return df


def compute_physics_features(practice_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core physics / true-pace features from FP2 long-run data.

    Features:
      - fp_fuel_corrected_pace: median long-run time adjusted for ~0.03s/lap fuel burn
      - fp_tire_deg_gradient: slope of lap times during long runs (higher = worse deg)
      - fp_long_run_consistency: std dev of long-run laps only
      - fp_top_speed_max: maximum speed trap reading in practice
    """
    if practice_laps.empty:
        return pd.DataFrame()

    laps = practice_laps.copy()
    group_cols = ["Year", "RoundNumber", "Driver"]

    # Filter valid laps
    if "LapTime_sec" in laps.columns:
        laps = laps[laps["LapTime_sec"].notna() & (laps["LapTime_sec"] > 0)].copy()

    if laps.empty:
        return pd.DataFrame()

    # === FP2 long-run features ===
    fp2 = laps[laps["SessionType"] == "Practice 2"].copy()
    long_run_features = []

    if not fp2.empty and "Stint" in fp2.columns and "TyreLife" in fp2.columns:
        # Identify long-run laps: stint >= 5 laps, TyreLife >= 3 (skip outlap/warmup)
        stint_sizes = fp2.groupby(group_cols + ["Stint"]).size().reset_index(name="stint_len")
        long_stints = stint_sizes[stint_sizes["stint_len"] >= 5]

        if not long_stints.empty:
            fp2_lr = fp2.merge(long_stints[group_cols + ["Stint"]], on=group_cols + ["Stint"])
            fp2_lr = fp2_lr[fp2_lr["TyreLife"] >= 3]  # skip outlap/warmup

            if not fp2_lr.empty:
                # Fuel-corrected pace: subtract estimated fuel effect
                # Fuel burns ~1.8kg/lap, each kg costs ~0.03s → ~0.054s/lap correction
                # Correct by subtracting (max_tyre_life_in_stint - tyre_life) * 0.05
                def fuel_correct_group(g):
                    max_life = g["TyreLife"].max()
                    correction = (max_life - g["TyreLife"]) * 0.05
                    return (g["LapTime_sec"] - correction).median()

                fuel_corrected = (
                    fp2_lr.groupby(group_cols)
                    .apply(fuel_correct_group)
                    .reset_index()
                    .rename(columns={0: "fp_fuel_corrected_pace"})
                )
                long_run_features.append(fuel_corrected)

                # Tire degradation gradient: slope of lap time vs tyre life
                from scipy.stats import linregress

                def deg_gradient(g):
                    if len(g) < 3:
                        return np.nan
                    try:
                        slope, _, _, _, _ = linregress(g["TyreLife"], g["LapTime_sec"])
                        return slope
                    except Exception:
                        return np.nan

                tire_deg = (
                    fp2_lr.groupby(group_cols)
                    .apply(deg_gradient)
                    .reset_index()
                    .rename(columns={0: "fp_tire_deg_gradient"})
                )
                long_run_features.append(tire_deg)

                # Long-run consistency (std dev of long run laps only)
                lr_consistency = (
                    fp2_lr.groupby(group_cols)["LapTime_sec"]
                    .std()
                    .reset_index()
                    .rename(columns={"LapTime_sec": "fp_long_run_consistency"})
                )
                long_run_features.append(lr_consistency)

    # === Top speed max (across ALL practice sessions) ===
    if "SpeedST" in laps.columns:
        top_speed = (
            laps.groupby(group_cols)["SpeedST"]
            .max()
            .reset_index()
            .rename(columns={"SpeedST": "fp_top_speed_max"})
        )
        long_run_features.append(top_speed)

    if not long_run_features:
        return pd.DataFrame()

    result = long_run_features[0]
    for df in long_run_features[1:]:
        result = result.merge(df, on=group_cols, how="outer")

    return result


def compute_constructor_teammate_features(
    df: pd.DataFrame, race_results_all: pd.DataFrame, qualifying_all: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute constructor strength and teammate qualifying gap features.

    Features:
      - constructor_points: cumulative team championship points before this race
      - teammate_quali_gap: qualifying time delta to teammate (positive = slower)
    """
    df = df.copy()

    # === Constructor strength: cumulative points before each race ===
    if "Points" in race_results_all.columns:
        rr = race_results_all.copy()
        rr = rr.sort_values(["Year", "RoundNumber"])

        # Cumulative team points per race (sum of both drivers)
        race_team_pts = rr.groupby(["Year", "RoundNumber", "Team"])["Points"].sum().reset_index()
        race_team_pts = race_team_pts.sort_values(["Year", "RoundNumber"])

        # Cumulative sum shifted by 1 (points BEFORE this race)
        race_team_pts["constructor_points"] = (
            race_team_pts.groupby("Team")["Points"]
            .apply(lambda x: x.shift(1).cumsum().fillna(0))
            .reset_index(level=0, drop=True)
        )

        df = df.merge(
            race_team_pts[["Year", "RoundNumber", "Team", "constructor_points"]],
            on=["Year", "RoundNumber", "Team"],
            how="left",
        )

    # === Teammate qualifying gap ===
    if "quali_best_time" in qualifying_all.columns and "Team" in qualifying_all.columns:
        q = qualifying_all[["Year", "RoundNumber", "Driver", "Team", "quali_best_time"]].copy()
        q = q.dropna(subset=["quali_best_time"])

        # Compute teammate's best time for each driver
        team_times = q.groupby(["Year", "RoundNumber", "Team"])["quali_best_time"].mean().reset_index()
        team_times.rename(columns={"quali_best_time": "team_avg_quali_time"}, inplace=True)

        q = q.merge(team_times, on=["Year", "RoundNumber", "Team"], how="left")
        # Gap = driver time - team avg (positive = slower than teammate avg)
        q["teammate_quali_gap"] = q["quali_best_time"] - q["team_avg_quali_time"]

        df = df.merge(
            q[["Year", "RoundNumber", "Driver", "teammate_quali_gap"]],
            on=["Year", "RoundNumber", "Driver"],
            how="left",
        )

    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction and derived features that capture
    cross-feature relationships for better prediction.
    """
    df = df.copy()

    # 1. Grid vs recent form — is the driver qualifying above or below their pace?
    if "grid_position" in df.columns and "driver_recent_avg_pos" in df.columns:
        df["grid_vs_recent_form"] = df["grid_position"] - df["driver_recent_avg_pos"]

    # 2. FP race-pace delta — proxy for race-vs-quali trim
    if "fp2_long_run_avg" in df.columns and "quali_best_time" in df.columns:
        df["fp_race_pace_delta"] = df["fp2_long_run_avg"] - df["quali_best_time"]

    # 3. Quali position change — driver grid vs teammate average grid
    if "grid_position" in df.columns and "Team" in df.columns:
        team_quali_avg = df.groupby(["Year", "RoundNumber", "Team"])["grid_position"].transform("mean")
        df["quali_position_change"] = df["grid_position"] - team_quali_avg

    # 4. Driver consistency score — rolling std of recent race positions
    if "race_position" in df.columns and "Driver" in df.columns:
        df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
        driver_std = (
            df.groupby("Driver")["race_position"]
            .apply(lambda x: x.shift(1).rolling(window=5, min_periods=2).std())
            .reset_index(level=0, drop=True)
        )
        df["driver_consistency_score"] = driver_std

    # 5. Team quali average — team strength proxy for that race
    if "grid_position" in df.columns and "Team" in df.columns:
        df["team_quali_avg"] = df.groupby(
            ["Year", "RoundNumber", "Team"]
        )["grid_position"].transform("mean")

    # 6. Position delta potential — room to gain or lose
    if "grid_position" in df.columns and "team_recent_avg_pos" in df.columns:
        df["position_delta_potential"] = df["grid_position"] - df["team_recent_avg_pos"]

    # 7. Sector speed ranks — normalized competitiveness per race
    for speed_col, rank_col in [
        ("fp_avg_speed_i1", "sector_speed_rank_i1"),
        ("fp_avg_speed_i2", "sector_speed_rank_i2"),
    ]:
        if speed_col in df.columns:
            # Higher speed = better = lower rank number
            df[rank_col] = df.groupby(["Year", "RoundNumber"])[speed_col].rank(
                ascending=False, method="min"
            )

    return df


def compute_target_encoding(
    df: pd.DataFrame,
    train_mask: pd.Series,
    target_col: str = "race_position",
) -> pd.DataFrame:
    """
    Target-encode driver, team, and circuit using leave-one-out on training data
    and global mean on test data to prevent leakage.
    """
    df = df.copy()
    train_df = df[train_mask]
    global_mean = train_df[target_col].mean()

    for col, new_col in [
        ("Driver", "driver_target_enc"),
        ("Team", "team_target_enc"),
        ("circuit_key", "circuit_target_enc"),
    ]:
        if col not in df.columns:
            continue

        # Compute category means on training data
        cat_means = train_df.groupby(col)[target_col].mean()

        # Leave-one-out for training: (sum - current) / (count - 1)
        cat_sums = train_df.groupby(col)[target_col].transform("sum")
        cat_counts = train_df.groupby(col)[target_col].transform("count")
        loo_values = (cat_sums - train_df[target_col]) / (cat_counts - 1)
        loo_values = loo_values.fillna(global_mean)

        # Apply
        df.loc[train_mask, new_col] = loo_values.values

        # Test set: use category mean from training; unseen categories get global mean
        test_mask = ~train_mask
        df.loc[test_mask, new_col] = (
            df.loc[test_mask, col].map(cat_means).fillna(global_mean)
        )

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
        tire_features = compute_tire_features(practice_laps)
        physics_features = compute_physics_features(practice_laps)

        if race_results.empty:
            print(f"  ⚠ No race results for {year}, skipping")
            continue

        # Base: race results (has the target variable)
        base = race_results[
            [c for c in [
                "Year", "RoundNumber", "EventName", "Driver", "Team",
                "DriverNumber", "race_position", "grid_position",
                "circuit_key", "is_street_circuit", "Status",
            ] if c in race_results.columns]
        ].copy()

        # Merge practice features
        if not fp_features.empty:
            base = base.merge(
                fp_features, on=["Year", "RoundNumber", "Driver"], how="left"
            )
            print(f"  ✓ Practice features merged ({len(fp_features)} rows)")

        # Merge tire strategy features
        if not tire_features.empty:
            base = base.merge(
                tire_features, on=["Year", "RoundNumber", "Driver"], how="left"
            )
            print(f"  ✓ Tire strategy features merged ({len(tire_features)} rows)")

        # Merge physics features (fuel-corrected pace, deg gradient, etc.)
        if not physics_features.empty:
            base = base.merge(
                physics_features, on=["Year", "RoundNumber", "Driver"], how="left"
            )
            print(f"  ✓ Physics features merged ({len(physics_features)} rows)")

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

        # Merge weather (expanded to include humidity and wind)
        if not weather.empty:
            weather_cols = ["Year", "RoundNumber", "track_temp_avg", "rainfall",
                           "humidity_avg", "wind_speed_avg"]
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

    # ── Compute interaction features ──
    full_df = compute_interaction_features(full_df)
    print(f"  ✓ Interaction features computed")

    # ── Compute circuit history features ──
    full_df = compute_circuit_history_features(full_df)
    print(f"  ✓ Circuit history features computed")

    # ── Compute DNF risk features ──
    # Build combined race results and qualifying for DNF + constructor + teammate
    all_race_results = []
    all_qualifying = []
    for year in years:
        data = load_saved_data(year)
        if not data["race_results"].empty:
            all_race_results.append(data["race_results"])
        if not data["qualifying"].empty:
            all_qualifying.append(data["qualifying"])

    if all_race_results:
        combined_rr = pd.concat(all_race_results, ignore_index=True)
        full_df = compute_dnf_features(full_df, combined_rr)
        print(f"  ✓ DNF risk features computed")

        # ── Compute constructor strength + teammate gap ──
        combined_q = pd.concat(all_qualifying, ignore_index=True) if all_qualifying else pd.DataFrame()
        full_df = compute_constructor_teammate_features(full_df, combined_rr, combined_q)
        print(f"  ✓ Constructor & teammate features computed")

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
