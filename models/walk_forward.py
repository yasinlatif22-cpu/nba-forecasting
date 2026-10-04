from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline import evaluate

DATA_PATH = Path("data/processed/team_season_dataset.csv")
GAMES = 82
MIN_TRAIN_SEASONS = 5


def walk_forward_simple(data):
    """For each test season t: fit on seasons before t only, then predict t.

    Input: the team-season dataset from build_dataset.py.
    Output: one row per test team-season with the fitted prediction and the
    slope/intercept used. Assumes START_YEAR is an int and seasons are consecutive.
    """
    seasons = sorted(data["START_YEAR"].unique())
    pieces = []
    for test_year in seasons[MIN_TRAIN_SEASONS:]:
        train = data[data["START_YEAR"] < test_year]
        test = data[data["START_YEAR"] == test_year]
        assert train["START_YEAR"].max() < test_year, "Leakage: training on the future"

        slope, intercept = np.polyfit(train["PREV_W_PCT"], train["ACTUAL_W_PCT"], 1)

        out = test[["SEASON", "TEAM_NAME", "PREV_W_PCT", "ACTUAL_W_PCT"]].copy()
        out["PRED_FITTED"] = intercept + slope * test["PREV_W_PCT"]
        out["SLOPE"] = slope
        out["INTERCEPT"] = intercept
        out["TRAIN_ROWS"] = len(train)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def main():
    data = pd.read_csv(DATA_PATH)
    preds = walk_forward_simple(data)

    actual = preds["ACTUAL_W_PCT"].to_numpy() * GAMES
    results = pd.DataFrame([
        evaluate("Everyone .500", actual, np.full_like(actual, 0.5 * GAMES)),
        evaluate("Naive: last season's win %", actual, preds["PREV_W_PCT"].to_numpy() * GAMES),
        evaluate("Fitted: win % ~ prev win %", actual, preds["PRED_FITTED"].to_numpy() * GAMES),
    ])

    print(f"Test seasons {preds['SEASON'].iloc[0]} to {preds['SEASON'].iloc[-1]}: {len(preds)} team-seasons")
    print(results.round(2).to_string(index=False))
    print()
    print(preds.groupby("SEASON")[["SLOPE", "INTERCEPT", "TRAIN_ROWS"]].first().round(3))


if __name__ == "__main__":
    main()
