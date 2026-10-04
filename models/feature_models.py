from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.baseline import evaluate

DATA_PATH = Path("data/processed/team_season_dataset.csv")
GAMES = 82
MIN_TRAIN_SEASONS = 5

# Candidate specifications, fixed in advance. Never combine OFF, DEF and NET.
FEATURE_SETS = {
    "Prev win %": ["PREV_W_PCT"],
    "Prev net rating": ["PREV_NET_RATING"],
    "Prev win % + net rating": ["PREV_W_PCT", "PREV_NET_RATING"],
    "Prev off + def rating": ["PREV_OFF_RATING", "PREV_DEF_RATING"],
    "Win % + net + pace/eFG/TOV/OREB": [
        "PREV_W_PCT", "PREV_NET_RATING", "PREV_PACE",
        "PREV_EFG_PCT", "PREV_TM_TOV_PCT", "PREV_OREB_PCT",
    ],
}


def walk_forward(data, features):
    """Expanding-window walk-forward predictions of ACTUAL_W_PCT.

    Input: the dataset and a list of PREV_ columns to use as inputs.
    Output: a Series of predictions indexed like the test rows.
    The assert is a structural leakage guard: only PREV_ columns may be inputs.
    """
    assert all(c.startswith("PREV_") for c in features), "Inputs must be PREV_ columns"
    seasons = sorted(data["START_YEAR"].unique())
    pieces = []
    for test_year in seasons[MIN_TRAIN_SEASONS:]:
        train = data[data["START_YEAR"] < test_year]
        test = data[data["START_YEAR"] == test_year]
        model = LinearRegression().fit(train[features], train["ACTUAL_W_PCT"])
        pieces.append(pd.Series(model.predict(test[features]), index=test.index))
    return pd.concat(pieces)


def per_season_mae(test, actual, predicted):
    """Mean absolute error computed separately for each season."""
    err = pd.Series(np.abs(actual - predicted), index=test.index)
    return err.groupby(test["SEASON"]).mean()


def main():
    data = pd.read_csv(DATA_PATH)

    cols = ["PREV_W_PCT", "PREV_NET_RATING", "PREV_OFF_RATING", "PREV_DEF_RATING"]
    print(data[cols].corr().round(2))
    gap = (data["PREV_OFF_RATING"] - data["PREV_DEF_RATING"] - data["PREV_NET_RATING"]).abs().max()
    print(f"Largest |OFF - DEF - NET| gap: {gap:.2f}\n")

    first_test_year = sorted(data["START_YEAR"].unique())[MIN_TRAIN_SEASONS]
    test = data[data["START_YEAR"] >= first_test_year]
    actual = test["ACTUAL_W_PCT"].to_numpy() * GAMES
    naive = test["PREV_W_PCT"].to_numpy() * GAMES
    naive_by_season = per_season_mae(test, actual, naive)

    rows = [
        evaluate("Everyone .500", actual, np.full_like(actual, 0.5 * GAMES)),
        evaluate("Naive: last season's win %", actual, naive),
    ]
    rows[0]["Beats naive"] = "-"
    rows[1]["Beats naive"] = "-"

    for name, feats in FEATURE_SETS.items():
        pred = walk_forward(data, feats)
        assert pred.index.equals(test.index), "Test rows misaligned"
        pred_wins = pred.to_numpy() * GAMES
        row = evaluate(name, actual, pred_wins)
        wins = (per_season_mae(test, actual, pred_wins) < naive_by_season).sum()
        row["Beats naive"] = f"{wins}/{naive_by_season.size} seasons"
        rows.append(row)

    print(f"{len(test)} test team-seasons")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
