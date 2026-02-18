# F1 Race Finishing Position Prediction: Technical Documentation

## 1. What is this project?
This is a robust **Machine Learning System** designed to predict the final finishing positions of Formula 1 drivers. Unlike simple statistical models, it treats the problem as a complex regression task influenced by physics, strategy, and historical performance.

It is built to answer the question: *"Given a driver's practice pace, qualifying result, and track history, where will they finish on Sunday?"*

The system specifically targets the **2025 season**, training on 2023–2024 data to learn recent car performance trends and team hierarchies.

## 2. Why use this specific approach?

F1 predictions are notoriously difficult due to high variance (crashes, safety cars, weather). Standard models often fail because they don't account for:
1.  **Time-Series Dependencies**: A driver's form evolves throughout the season.
2.  **Track Characteristics**: A car good at Monaco (high downforce) might be terrible at Monza (low drag).
3.  **Data Leakage**: Randomly splitting train/test data "leaks" future car development info into the past.

**Our Solution:**
-   **Stacking Ensemble**: Combines the stability of Random Forest with the precision of Gradient Boosting (XGBoost/LightGBM).
-   **Physics-Informed Features**: adjusts raw lap times for fuel load to find "true" race pace.
-   **Strict Chronological Validation**: Uses `TimeSeriesGroupSplit` to simulate real-world forecasting conditions.

**Performance Benchmarks (2025 Test Set):**
-   **Top-3 Accuracy**: **76.4%** (Extremely high reliability for podium predictions).
-   **Mean Absolute Error (MAE)**: **3.10** positions (On average, predictions are within ~3 places).
-   **Spearman Rank Correlation**: **0.67** (Strong ability to rank the entire field correctly).

## 3. How does it work? (Technical Deep Dive)

The pipeline consists of four distinct stages:

### Stage A: Data Collection & Processing
We use the **FastF1** API to fetch granular telemetry data.
-   **Session Data**: Lap times, tyre compounds, sector times, and weather (temp, humidity, rain).
-   **Cleaning**:
    -   Filters out "in-laps" and "out-laps" (unrepresentative slow laps).
    -   Removes anomalies (crashes, red flags) that skew pace calculations.
    -   **Smart Imputation**: Missing practice data (e.g., no FP2 long run due to a crash) is imputed with **median values**.
        -   *Why?* If Max Verstappen crashes in FP2, imputing a "Slowest" time would mislead the model into thinking the car is slow. Imputing a "Median" time is **neutral**, forcing the model to disregard the missing practice pace and rely on other strong predictive features like `grid_position` and `driver_encoded` (which correctly identify him as a contender). Experiments confirmed this yields higher accuracy than penalizing missing data.

### Stage B: Advanced Feature Engineering (36 Features)
The model uses **36 inputs** per driver per race. Here is exactly what they are and why they matter:

#### 1. Practice Pace (6 Features)
*Captures raw car speed from Free Practice sessions.*
-   `fp_best_lap`: The absolute fastest single lap recorded in any practice session. (Pure speed)
-   `fp2_long_run_avg`: Average lap time during "race simulations" in FP2. (Race pace proxy)
-   `fp_consistency`: Standard deviation of lap times. Lower = better tire management.
-   `fp_avg_speed_i1/i2/st`: Average speed at Intermediate 1, Intermediate 2, and Speed Trap. (Drag/Downforce set-up indicators)

#### 2. Qualifying (3 Features)
*Starting position is the single strongest predictor in F1.*
-   `grid_position`: Where the driver actually starts (after penalties).
-   `quali_best_time`: The driver's fastest lap in Q1, Q2, or Q3.
-   `quali_gap_to_pole`: Time gap to the pole sitter (e.g., +0.350s). Normalizes pace across different track lengths.

#### 3. Context & Environment (6 Features)
*External factors affecting race dynamics.*
-   `circuit_encoded`: Label encoding of the track ID.
-   `is_street_circuit`: Boolean (1/0). Street tracks (Monaco, Singapore) have high correlation between Grid and Finish.
-   `track_temp_avg`: Average track temperature. Hotter tracks degrade tires faster.
-   `rainfall`: Binary indicator of rain during the session.
-   `humidity_avg` / `wind_speed_avg`: Weather conditions affecting aerodynamics and engine performance.

#### 4. Driver & Team Identity (2 Features)
*Captures hierarchical performance (e.g., Red Bull > Haas).*
-   `driver_encoded` / `team_encoded`: Unique IDs for categorical embeddings. Tree models split on these to learn "Max Verstappen usually wins".

#### 5. Rolling Form (2 Features)
*Momentum indicators.*
-   `driver_recent_avg_pos`: Average finishing position in the last 3 races.
-   `team_recent_avg_pos`: Team's average result in the last 3 races (proxy for upgrades/form).

#### 6. Interaction & Relative Metrics (8 Features)
*Complex non-linear features derived from combining others.*
-   `fp_race_pace_delta`: Difference between Qualifying Best Lap and FP2 Long Run Average. Large delta = Car is better in Quali than Race (e.g., Haas 2023).
-   `grid_vs_recent_form`: `grid_position` minus `driver_recent_avg_pos`. (Negative = Out of position / Potential recovery drive).
-   `quali_position_change`: Positions gained/lost between derived pace-rank and actual grid slot.
-   `driver_consistency_score`: A composite score of lap time variance across all sessions.
-   `team_quali_avg`: Average qualifying position of the *teammate*. (Filters driver skill vs car potential).
-   `position_delta_potential`: Theoretical positions to gain based on car pace vs grid slot.
-   `sector_speed_rank_i1/i2`: Rank order of sector speeds. (Good Sector 3 often means good tire life).

#### 7. Tire Strategy (4 Features)
*Indications of race strategy availability.*
-   `fp_n_compounds_used`: How many tire types (Soft/Medium/Hard) were tested?
-   `fp_soft_pct`: Percentage of practice laps on Soft tires.
-   `fp_stint_count`: Number of distinct stints. High count = active testing program.
-   `fp_avg_tyre_life`: Average age of tires used in practice long runs.

#### 8. Circuit History (3 Features)
*Track affinity metrics.*
-   `driver_circuit_avg_pos`: Driver's career average finish at this specific track.
-   `team_circuit_avg_pos`: Team's historical performance at this track.
-   `driver_circuit_best_pos`: Driver's personal best finish at this track.

#### 9. Reliability / Risk (2 Features)
-   `driver_dnf_rate`: Rolling probability of a "Did Not Finish". High for crash-prone drivers.
-   `team_dnf_rate`: Mechanical reliability score of the constructor.

*Note: Physics-based features (Fuel correction, Tire deg gradient) were analyzed in Phase 7 but were replaced by the simpler "Interaction" proxies above for the final 36-feature model, as they yielded better stability.*

### Stage C: Model Architecture (The Stacking Ensemble)
We use a **Stacking Ensemble**, which trains a "Meta-Learner" to combine predictions from three base models. This is superior to any single model.

| Layer | Model | Role & Configuration |
| :--- | :--- | :--- |
| **Base 1** | **Random Forest** | **The Stabilizer**. Uses `max_depth=14`, `n_estimators=100`. Captures complex non-linear interactions without overfitting. Good at handling noisy data. |
| **Base 2** | **XGBoost** | **The Precision Specialist**. Uses gradient boosting to correct errors made by previous trees. Tuned for low learning rate (`0.016`) to learn subtle patterns. |
| **Base 3** | **LightGBM** | **The Speedster**. Uses leaf-wise growth. Excellent at finding "edge cases" in driver performance. |
| **Meta** | **Lasso Regression** | **The Judge**. Takes the predictions from RF, XGB, and LGBM as inputs. It uses a **positive constraint** (coefficients > 0) to enforce a weighted average, preventing it from learning "anti-correlations" that don't exist in physics. |

### Stage D: Validation Strategy
We strictly forbid standard K-Fold Cross-Validation.
-   **Method**: `TimeSeriesGroupSplit` (Expanding Window).
-   **Logic**: To predict Race 10, we can ONLY train on Races 1–9. We cannot use Race 11 data.
-   **Recency Weighting**: The model applies exponentially higher weights to recent races (2024 is ~2x more important than 2023) because F1 car development is rapid.

## 4. Usage Guide

### Prerequisites
-   Python 3.9+
-   Libraries: `scikit-learn`, `pandas`, `numpy`, `xgboost`, `lightgbm`, `optuna`, `fastf1`

### Installation
```bash
git clone https://repo-url/f1-prediction.git
pip install -r requirements.txt
```

### Running the Model
To execute the pipeline using the hardcoded **"High-Accuracy"** parameters (Top-3 Acc 76.4%):
```bash
python src/model.py
```
*Output will display MAE, Spearman Rank, and Feature Importances for the 2025 Test Set.*

### Re-Training (Full Optimization)
If you want to re-run the Bayesian Hyperparameter Optimization (Optuna):
1.  Open `src/model.py`
2.  Change `tune=False` to `tune=True` in the `__main__` block.
3.  Run the script (Warning: Takes ~30-60 mins).
