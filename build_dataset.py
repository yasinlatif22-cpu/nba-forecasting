from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/team_advanced_2010_2025.csv")
OUT_PATH = Path("data/processed/team_season_dataset.csv")

PREV_COLS = [
    "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PACE", "EFG_PCT", "TS_PCT", "TM_TOV_PCT", "OREB_PCT",
]


def build_dataset(raw):
    """Turn one-row-per-team-season stats into a prediction dataset.

    Each output row = one team in season t.
      PREV_* columns  -> that team's final stats from season t-1 (usable inputs)
      ACTUAL_* columns -> what happened in season t (the target; never an input)
    """
    df = raw.copy()
    df["START_YEAR"] = df["SEASON"].str[:4].astype(int)

    # Last season's stats, relabelled so they line up with THIS season's row.
    prev = df[["TEAM_ID", "START_YEAR", "SEASON"] + PREV_COLS].copy()
    prev["START_YEAR"] = prev["START_YEAR"] + 1
    prev = prev.rename(columns={"SEASON": "PREV_SEASON"})
    prev = prev.rename(columns={c: f"PREV_{c}" for c in PREV_COLS})

    # What actually happened this season: the answer key.
    target = df[["SEASON", "START_YEAR", "TEAM_ID", "TEAM_NAME", "GP", "W", "W_PCT"]]
    target = target.rename(
        columns={"GP": "ACTUAL_GP", "W": "ACTUAL_W", "W_PCT": "ACTUAL_W_PCT"}
    )

    # Inner join: 2010-11 has no prior season in our data, so it drops out.
    out = target.merge(prev, on=["TEAM_ID", "START_YEAR"], how="inner")
    return out.sort_values(["START_YEAR", "TEAM_ID"]).reset_index(drop=True)


def check_no_leakage(data):
    """Every row's inputs must come from exactly one season earlier."""
    prev_year = data["PREV_SEASON"].str[:4].astype(int)
    assert (prev_year == data["START_YEAR"] - 1).all(), "Leakage: PREV season is wrong"


def main():
    raw = pd.read_csv(RAW_PATH)
    data = build_dataset(raw)
    check_no_leakage(data)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(data)} rows to {OUT_PATH}")
    print("Missing values per column:")
    print(data.isna().sum()[data.isna().sum() > 0])
    print(data[data.TEAM_NAME == "Chicago Bulls"].head(2).T)


if __name__ == "__main__":
    main()
