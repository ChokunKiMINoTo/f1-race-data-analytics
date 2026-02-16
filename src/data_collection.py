"""
Data Collection Module
Fetches F1 practice, qualifying, and race data using FastF1.
Caches data locally to avoid repeated API calls.
"""
import sys
import os
import warnings
import logging

import fastf1
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Suppress FastF1 verbose warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.Cache.enable_cache(config.CACHE_DIR)

logger = logging.getLogger(__name__)


def get_race_schedule(year: int) -> pd.DataFrame:
    """
    Get the race schedule for a given year.
    Returns a DataFrame with round numbers, event names, and locations.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Filter to only conventional events (rounds >= 1)
    schedule = schedule[schedule["RoundNumber"] >= 1].reset_index(drop=True)
    return schedule


def load_session_safely(year: int, round_num: int, session_type: str):
    """
    Load a FastF1 session with error handling.
    Returns None if the session cannot be loaded.
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(
            laps=True,
            telemetry=False,  # Skip heavy telemetry to keep it fast
            weather=True,
            messages=False,
        )
        return session
    except Exception as e:
        logger.warning(
            f"Could not load {session_type} for {year} Round {round_num}: {e}"
        )
        return None


def extract_lap_data(session) -> pd.DataFrame:
    """
    Extract per-driver lap data from a loaded session.
    Returns lap times, sector times, speed traps, tire info per driver.
    """
    if session is None or session.laps.empty:
        return pd.DataFrame()

    laps = session.laps.copy()

    # Select relevant columns
    cols_to_keep = [
        "DriverNumber",
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
        "Compound",
        "TyreLife",
        "Stint",
        "IsPersonalBest",
        "FreshTyre",
        "TrackStatus",
    ]

    available_cols = [c for c in cols_to_keep if c in laps.columns]
    laps = laps[available_cols].copy()

    # Convert timedelta columns to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[f"{col}_sec"] = laps[col].dt.total_seconds()

    # Add session metadata
    laps["Year"] = session.event.year if hasattr(session.event, "year") else session.event["EventDate"].year
    laps["RoundNumber"] = session.event["RoundNumber"]
    laps["EventName"] = session.event["EventName"]
    laps["SessionType"] = session.name

    return laps


def extract_weather_data(session) -> dict:
    """
    Extract average weather conditions from a session.
    Returns a dict with track temp, air temp, humidity, rainfall, wind.
    """
    if session is None:
        return {}

    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return {}

        return {
            "track_temp_avg": weather["TrackTemp"].mean(),
            "air_temp_avg": weather["AirTemp"].mean(),
            "humidity_avg": weather["Humidity"].mean(),
            "rainfall": int(weather["Rainfall"].any()),
            "wind_speed_avg": weather["WindSpeed"].mean(),
        }
    except Exception as e:
        logger.warning(f"Could not extract weather: {e}")
        return {}


def extract_race_results(session) -> pd.DataFrame:
    """
    Extract final race classification.
    Returns DataFrame with driver, team, position, status, points.
    """
    if session is None:
        return pd.DataFrame()

    try:
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()

        cols = [
            "DriverNumber",
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "Position",
            "ClassifiedPosition",
            "Status",
            "Points",
        ]
        available = [c for c in cols if c in results.columns]
        results = results[available].copy()

        # Rename for clarity
        rename_map = {}
        if "Position" in results.columns:
            rename_map["Position"] = "race_position"
        if "GridPosition" in results.columns:
            rename_map["GridPosition"] = "grid_position"
        if "Abbreviation" in results.columns:
            rename_map["Abbreviation"] = "Driver"
        if "TeamName" in results.columns:
            rename_map["TeamName"] = "Team"

        results.rename(columns=rename_map, inplace=True)

        # Convert position to numeric
        if "race_position" in results.columns:
            results["race_position"] = pd.to_numeric(
                results["race_position"], errors="coerce"
            )
        if "grid_position" in results.columns:
            results["grid_position"] = pd.to_numeric(
                results["grid_position"], errors="coerce"
            )

        return results
    except Exception as e:
        logger.warning(f"Could not extract race results: {e}")
        return pd.DataFrame()


def extract_qualifying_results(session) -> pd.DataFrame:
    """
    Extract qualifying results (Q1, Q2, Q3 times and grid positions).
    """
    if session is None:
        return pd.DataFrame()

    try:
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()

        cols = [
            "DriverNumber",
            "Abbreviation",
            "TeamName",
            "Position",
            "Q1",
            "Q2",
            "Q3",
        ]
        available = [c for c in cols if c in results.columns]
        results = results[available].copy()

        rename_map = {}
        if "Abbreviation" in results.columns:
            rename_map["Abbreviation"] = "Driver"
        if "TeamName" in results.columns:
            rename_map["TeamName"] = "Team"
        if "Position" in results.columns:
            rename_map["Position"] = "quali_position"
        results.rename(columns=rename_map, inplace=True)

        # Convert Q times to seconds
        for q_col in ["Q1", "Q2", "Q3"]:
            if q_col in results.columns:
                results[f"{q_col}_sec"] = results[q_col].dt.total_seconds()

        # Compute best qualifying time
        q_sec_cols = [c for c in ["Q1_sec", "Q2_sec", "Q3_sec"] if c in results.columns]
        if q_sec_cols:
            results["quali_best_time"] = results[q_sec_cols].min(axis=1)
            pole_time = results["quali_best_time"].min()
            results["quali_gap_to_pole"] = results["quali_best_time"] - pole_time

        if "quali_position" in results.columns:
            results["quali_position"] = pd.to_numeric(
                results["quali_position"], errors="coerce"
            )

        return results
    except Exception as e:
        logger.warning(f"Could not extract qualifying results: {e}")
        return pd.DataFrame()


def collect_season_data(year: int) -> dict:
    """
    Collect all data for an entire season.
    Returns a dict with keys: 'practice_laps', 'qualifying', 'race_results', 'weather'.
    Each is a list of DataFrames per round.
    """
    schedule = get_race_schedule(year)
    print(f"\n{'='*60}")
    print(f"Collecting data for {year} season ({len(schedule)} events)")
    print(f"{'='*60}")

    all_practice_laps = []
    all_qualifying = []
    all_race_results = []
    all_weather = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        print(f"\n  Round {round_num}: {event_name}...")

        # — Practice sessions —
        for fp in config.PRACTICE_SESSIONS:
            session = load_session_safely(year, round_num, fp)
            laps = extract_lap_data(session)
            if not laps.empty:
                all_practice_laps.append(laps)
                print(f"    ✓ {fp}: {len(laps)} laps")
            else:
                print(f"    ✗ {fp}: no data")

        # — Qualifying —
        quali_session = load_session_safely(year, round_num, config.QUALIFYING_SESSION)
        quali_results = extract_qualifying_results(quali_session)
        if not quali_results.empty:
            quali_results["Year"] = year
            quali_results["RoundNumber"] = round_num
            quali_results["EventName"] = event_name
            all_qualifying.append(quali_results)
            print(f"    ✓ Qualifying: {len(quali_results)} drivers")
        else:
            print(f"    ✗ Qualifying: no data")

        # — Race Results —
        race_session = load_session_safely(year, round_num, config.RACE_SESSION)
        race_results = extract_race_results(race_session)
        if not race_results.empty:
            race_results["Year"] = year
            race_results["RoundNumber"] = round_num
            race_results["EventName"] = event_name

            # Get circuit info
            if race_session is not None:
                race_results["circuit_key"] = event.get("Location", "Unknown")
                race_results["is_street_circuit"] = int(
                    any(
                        sc.lower() in str(event.get("Location", "")).lower()
                        for sc in config.STREET_CIRCUITS
                    )
                    or any(
                        sc.lower() in str(event.get("EventName", "")).lower()
                        for sc in config.STREET_CIRCUITS
                    )
                )

            all_race_results.append(race_results)
            print(f"    ✓ Race: {len(race_results)} classified")
        else:
            print(f"    ✗ Race: no data")

        # — Weather (from race session) —
        weather = extract_weather_data(race_session)
        if weather:
            weather["Year"] = year
            weather["RoundNumber"] = round_num
            weather["EventName"] = event_name
            all_weather.append(weather)
            print(f"    ✓ Weather: track={weather.get('track_temp_avg', 'N/A'):.1f}°C")
        else:
            print(f"    ✗ Weather: no data")

    # Combine
    practice_df = pd.concat(all_practice_laps, ignore_index=True) if all_practice_laps else pd.DataFrame()
    qualifying_df = pd.concat(all_qualifying, ignore_index=True) if all_qualifying else pd.DataFrame()
    results_df = pd.concat(all_race_results, ignore_index=True) if all_race_results else pd.DataFrame()
    weather_df = pd.DataFrame(all_weather) if all_weather else pd.DataFrame()

    print(f"\n  Season {year} complete!")
    print(f"    Practice laps: {len(practice_df)}")
    print(f"    Qualifying entries: {len(qualifying_df)}")
    print(f"    Race results: {len(results_df)}")
    print(f"    Weather records: {len(weather_df)}")

    return {
        "practice_laps": practice_df,
        "qualifying": qualifying_df,
        "race_results": results_df,
        "weather": weather_df,
    }


def collect_and_save_all_data():
    """
    Collect data for all configured seasons and save to disk.
    """
    for year in config.ALL_YEARS:
        data = collect_season_data(year)

        # Save each DataFrame to parquet
        for key, df in data.items():
            if not df.empty:
                filepath = os.path.join(config.PROCESSED_DIR, f"{key}_{year}.parquet")
                # Convert timedelta columns to seconds before saving
                for col in df.columns:
                    if pd.api.types.is_timedelta64_dtype(df[col]):
                        df[f"{col}_seconds"] = df[col].dt.total_seconds()
                        df.drop(columns=[col], inplace=True)
                df.to_parquet(filepath, index=False)
                print(f"  Saved: {filepath}")

    print("\n✅ All data collected and saved!")


def load_saved_data(year: int) -> dict:
    """
    Load previously saved data from parquet files.
    """
    data = {}
    for key in ["practice_laps", "qualifying", "race_results", "weather"]:
        filepath = os.path.join(config.PROCESSED_DIR, f"{key}_{year}.parquet")
        if os.path.exists(filepath):
            data[key] = pd.read_parquet(filepath)
        else:
            data[key] = pd.DataFrame()
            logger.warning(f"File not found: {filepath}")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_and_save_all_data()
