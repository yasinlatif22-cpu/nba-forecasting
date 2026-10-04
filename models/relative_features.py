import numpy as np
import pandas as pd

from analysis.features import add_relative_features
from models.baseline import evaluate
from models.feature_models import (
    DATA_PATH, GAMES, MIN_TRAIN_SEASONS, per_season_mae, walk_forward,
)

SPECS = {
    "Prev win %": ["PREV_W_PCT"],
    "Off + def (raw)": ["PREV_OFF_RATING", "PREV_DEF_RATING"],
    "Off + def (league-relative)": ["PREV_OFF_RATING_REL", "PREV_DEF_RATING_REL"],
    "Win% + net + 4 factors (raw)": [
        "PREV_W_PCT", "PREV_NET_RATING", "PREV_PACE",
        "PREV_EFG_PCT", "PREV_TM_TOV_PCT", "PREV_OREB_PCT",
    ],
    "Win% + net + 4 factors (relative)": [
        "PREV_W_PCT", "PREV_NET_RATING", "PREV_PACE_REL",
        "PREV_EFG_PCT_REL", "PREV_TM_TOV_PCT_REL", "PREV_OREB_PCT_REL",
    ],
}


def main():
    data = pd.read_csv(DATA_PATH)

    # 1. Does the league average drift over time?
    league = data.groupby("PREV_SEASON")[["PREV_OFF_RATING", "PREV_DEF_RATING", "PREV_PACE", "PREV_EFG_PCT"]].mean()
    print("League average by season:")
    print(league.round(3), "\n")

    # 2. Build league-relative features and sanity-check them.
    data = add_relative_features(data)
    max_mean = data.groupby("PREV_SEASON")["PREV_OFF_RATING_REL"].mean().abs().max()
    assert max_mean < 1e-9, "Relative features should average zero within each season"

    # 3. Score every spec on the same walk-forward test rows.
    first_test_year = sorted(data["START_YEAR"].unique())[MIN_TRAIN_SEASONS]
    test = data[data["START_YEAR"] >= first_test_year]
    actual = test["ACTUAL_W_PCT"].to_numpy() * GAMES
    naive = test["PREV_W_PCT"].to_numpy() * GAMES

    rows = [evaluate("Naive: last season's win %", actual, naive)]
    season_mae = {"Naive": per_season_mae(test, actual, naive)}
    for name, feats in SPECS.items():
        pred = walk_forward(data, feats)
        assert pred.index.equals(test.index), "Test rows misaligned"
        rows.append(evaluate(name, actual, pred.to_numpy() * GAMES))
        if name == "Prev win %":
            season_mae["Fitted prev win %"] = per_season_mae(test, actual, pred.to_numpy() * GAMES)

    print(f"{len(test)} test team-seasons")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nMAE by season (wins):")
    print(pd.DataFrame(season_mae).round(2))


if __name__ == "__main__":
    main()
