from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/processed/team_season_dataset.csv")
FIG_DIR = Path("figures")
GAMES = 82


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot


def evaluate(name, actual, predicted):
    return {
        "model": name,
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2": r_squared(actual, predicted),
    }


def main():
    data = pd.read_csv(DATA_PATH)

    # Put everything on an 82-game scale so seasons are comparable.
    actual = data["ACTUAL_W_PCT"].to_numpy() * GAMES
    naive = data["PREV_W_PCT"].to_numpy() * GAMES
    average = np.full_like(actual, 0.5 * GAMES)

    # Cross-check my formulas against scikit-learn.
    assert np.isclose(mae(actual, naive), mean_absolute_error(actual, naive))
    assert np.isclose(rmse(actual, naive), np.sqrt(mean_squared_error(actual, naive)))
    assert np.isclose(r_squared(actual, naive), r2_score(actual, naive))

    results = pd.DataFrame([
        evaluate("Everyone .500", actual, average),
        evaluate("Naive: last season's win %", actual, naive),
    ])
    print(f"Evaluated on {len(actual)} team-seasons")
    print(results.round(2).to_string(index=False))

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(naive, actual, alpha=0.5)
    ax.plot([5, 75], [5, 75], linestyle="--", color="gray")
    ax.set_xlim(5, 75)
    ax.set_ylim(5, 75)
    ax.set_xlabel("Predicted wins (last season's win % x 82)")
    ax.set_ylabel("Actual wins (win % x 82)")
    ax.set_title("Naive baseline: predicted vs. actual wins")
    fig.savefig(FIG_DIR / "baseline_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    print("Saved figures/baseline_predicted_vs_actual.png")


if __name__ == "__main__":
    main()
