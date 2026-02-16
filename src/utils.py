"""
Utility Module
Plotting, formatting, and helper functions for the F1 prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15,
                             title: str = "Feature Importance"):
    """Plot horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = fi_df.head(top_n).sort_values("importance")

    bars = ax.barh(data["feature"], data["importance"], color=sns.color_palette("viridis", len(data)))
    ax.set_xlabel("Importance")
    ax.set_title(title)

    # Add percentage labels
    for bar, pct in zip(bars, data["importance_pct"].values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model"):
    """Scatter plot of predicted vs actual positions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="k", linewidths=0.5, s=40)
    ax.plot([1, 20], [1, 20], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Race Position")
    ax.set_ylabel("Predicted Race Position")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict):
    """
    Bar chart comparing model metrics side by side.
    """
    metrics_to_plot = ["mae", "spearman_rho", "top3_accuracy", "ndcg_10"]
    models = list(results.keys())

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        values = [results[m].get(metric, 0) for m in models]
        colors = sns.color_palette("Set2", len(models))
        bars = ax.bar(models, values, color=colors, edgecolor="k", linewidth=0.5)
        ax.set_title(metric.upper().replace("_", " "))
        ax.set_ylabel(metric)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison on 2025 Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_per_race_mae(test_df: pd.DataFrame, y_pred: np.ndarray,
                       model_name: str = "Model"):
    """
    Bar chart showing MAE per race for a model.
    """
    df = test_df.copy()
    df["y_pred"] = y_pred
    df["abs_error"] = np.abs(df["race_position"] - df["y_pred"])

    race_mae = (
        df.groupby(["RoundNumber", "EventName"])["abs_error"]
        .mean()
        .reset_index()
        .sort_values("RoundNumber")
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = sns.color_palette("coolwarm", len(race_mae))
    bars = ax.bar(race_mae["EventName"], race_mae["abs_error"],
                  color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Grand Prix")
    ax.set_ylabel("Mean Absolute Error (positions)")
    ax.set_title(f"{model_name}: MAE per Race (2025)")
    ax.axhline(y=race_mae["abs_error"].mean(), color="red", linestyle="--",
               label=f"Average MAE: {race_mae['abs_error'].mean():.2f}")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_qualifying_vs_race(df: pd.DataFrame):
    """
    Scatter plot showing correlation between qualifying and race positions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df["grid_position"], df["race_position"],
               alpha=0.3, s=30, edgecolors="k", linewidths=0.3)
    ax.plot([1, 20], [1, 20], "r--", linewidth=2, alpha=0.7,
            label="Grid = Race Position")
    ax.set_xlabel("Grid Position (Qualifying)")
    ax.set_ylabel("Race Finishing Position")
    ax.set_title("Qualifying Position vs Race Position")
    ax.legend()
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_aspect("equal")

    # Add correlation text
    rho, p = spearmanr_safe(df["grid_position"], df["race_position"])
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    return fig


def plot_team_performance(df: pd.DataFrame, year: int = None):
    """
    Box plot of race finishing positions per team.
    """
    data = df.copy()
    if year is not None:
        data = data[data["Year"] == year]

    # Order teams by median position
    team_order = (
        data.groupby("Team")["race_position"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=data, x="Team", y="race_position",
                order=team_order, ax=ax, palette="Set3")
    ax.set_xlabel("Team")
    ax.set_ylabel("Race Position")
    title = f"Race Finishing Positions by Team ({year})" if year else "Race Finishing Positions by Team (All Years)"
    ax.set_title(title)
    ax.invert_yaxis()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    """
    Heatmap of feature correlations.
    """
    available = [c for c in features + ["race_position"] if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig


def spearmanr_safe(x, y):
    """Compute Spearman correlation, handling NaN and constant arrays."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = np.array(x)[mask], np.array(y)[mask]
    if len(x_clean) < 3 or len(np.unique(x_clean)) < 2:
        return 0.0, 1.0
    return spearmanr(x_clean, y_clean)


def format_results_table(results: dict) -> pd.DataFrame:
    """
    Format model evaluation results into a clean DataFrame.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.reset_index()
