from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.classification import (
    FEATURES, MIN_TRAIN_SEASONS, calibration_table, walk_forward_logit,
)
from models.walk_forward import walk_forward_simple

DATA_PATH = Path("data/processed/team_season_playoffs.csv")
FROZEN_PATH = Path("predictions/2026_27_playoff_probabilities_v1.0.csv")
FIG_DIR = Path("figures")
GAMES = 82


def wins_figure(data):
    """Naive vs fitted forecasts against actual wins, on the same walk-forward test rows."""
    wf = walk_forward_simple(data)
    actual = wf["ACTUAL_W_PCT"].to_numpy() * GAMES
    panels = [
        ("Naive: last season's wins", wf["PREV_W_PCT"].to_numpy() * GAMES),
        ("v1.0: fitted line on last season's win %", wf["PRED_FITTED"].to_numpy() * GAMES),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, (title, pred) in zip(axes, panels):
        mae = np.mean(np.abs(actual - pred))
        ax.scatter(pred, actual, alpha=0.4, s=18)
        ax.plot([5, 75], [5, 75], linestyle="--", color="gray")
        ax.set_xlim(5, 75)
        ax.set_ylim(5, 75)
        ax.set_title(f"{title}\nMAE {mae:.2f} wins")
        ax.set_xlabel("Predicted wins")
    axes[0].set_ylabel("Actual wins (win % x 82)")
    fig.suptitle(f"Walk-forward test: {len(wf)} team-seasons, 2016-17 to 2025-26", y=1.06)
    fig.savefig(FIG_DIR / "walk_forward_vs_naive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_figure(data):
    """Reliability diagram for 'made the playoffs' on the walk-forward test rows."""
    first_test = sorted(data["START_YEAR"].unique())[MIN_TRAIN_SEASONS]
    test = data[data["START_YEAR"] >= first_test]
    p = walk_forward_logit(data, "ACTUAL_MADE_PLAYOFFS", FEATURES)
    table = calibration_table(test["ACTUAL_MADE_PLAYOFFS"].to_numpy(), p.to_numpy(), 5)
    se = np.sqrt(table["avg_pred"] * (1 - table["avg_pred"]) / table["n"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.errorbar(table["avg_pred"], table["actual"], yerr=se, fmt="o", capsize=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average predicted probability (bin)")
    ax.set_ylabel("Share that actually made the playoffs")
    ax.set_title(f"Made playoffs: calibration, {len(test)} team-seasons\n"
                 "bars: 1 standard error if perfectly calibrated")
    fig.savefig(FIG_DIR / "playoff_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def title_figure():
    """The frozen 2026-27 title probabilities for the ten most likely teams."""
    frozen = pd.read_csv(FROZEN_PATH).sort_values("P_TITLE_NORM", ascending=False).head(10)
    frozen = frozen.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(frozen["TEAM_NAME"], frozen["P_TITLE_NORM"] * 100)
    ax.set_xlabel("Title probability (%)")
    ax.set_title("Frozen 2026-27 title probabilities (v1.0, made before tip-off)")
    fig.savefig(FIG_DIR / "title_probabilities_2026_27.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    wins_figure(data)
    calibration_figure(data)
    title_figure()
    print("Saved figures/walk_forward_vs_naive.png, playoff_calibration.png, title_probabilities_2026_27.png")


if __name__ == "__main__":
    main()
