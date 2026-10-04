from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/team_season_dataset.csv")
PLAYOFFS_PATH = Path("data/raw/team_playoffs_2010_2025.csv")
OUT_PATH = Path("data/processed/team_season_playoffs.csv")

LABELS = ["ACTUAL_MADE_PLAYOFFS", "ACTUAL_CONF_FINALS", "ACTUAL_FINALS", "ACTUAL_CHAMPION"]
EXPECTED_PER_SEASON = {"ACTUAL_MADE_PLAYOFFS": 16, "ACTUAL_CONF_FINALS": 4,
                       "ACTUAL_FINALS": 2, "ACTUAL_CHAMPION": 1}


def add_playoff_labels(data, playoffs):
    """Attach what happened in the playoffs to each team-season in the dataset.

    ACTUAL_MADE_PLAYOFFS: the team appears in the playoff table.
    ACTUAL_SERIES_WON: playoff wins // 4 (0 if the team missed the playoffs).
    ACTUAL_CONF_FINALS / _FINALS / _CHAMPION: won at least 2 / 3 / 4 series.
    These describe the season being predicted, so they are outcomes and never inputs.
    """
    po = playoffs[["SEASON", "TEAM_ID", "PO_W"]]
    out = data.merge(po, on=["SEASON", "TEAM_ID"], how="left")
    assert len(out) == len(data), "Merge changed the number of rows"
    out["ACTUAL_MADE_PLAYOFFS"] = out["PO_W"].notna().astype(int)
    out["ACTUAL_SERIES_WON"] = (out["PO_W"].fillna(0) // 4).astype(int)
    out["ACTUAL_CONF_FINALS"] = (out["ACTUAL_SERIES_WON"] >= 2).astype(int)
    out["ACTUAL_FINALS"] = (out["ACTUAL_SERIES_WON"] >= 3).astype(int)
    out["ACTUAL_CHAMPION"] = (out["ACTUAL_SERIES_WON"] >= 4).astype(int)
    return out.drop(columns=["PO_W"])


def check_labels(labeled):
    """Each season must have exactly 16 playoff teams, 4 conference finalists, 2 finalists, 1 champion."""
    per_season = labeled.groupby("SEASON")[LABELS].sum()
    for label, expected in EXPECTED_PER_SEASON.items():
        bad = per_season[per_season[label] != expected]
        assert bad.empty, f"{label}: wrong count in {list(bad.index)}"


def main():
    data = pd.read_csv(DATA_PATH)
    playoffs = pd.read_csv(PLAYOFFS_PATH)
    labeled = add_playoff_labels(data, playoffs)
    check_labels(labeled)
    labeled.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(labeled)} rows to {OUT_PATH}")
    print("Playoff-label counts per season all match 16 / 4 / 2 / 1")
    print(labeled[LABELS].sum().to_string())


if __name__ == "__main__":
    main()
