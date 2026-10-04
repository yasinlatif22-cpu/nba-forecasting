import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

from collect_data import FIRST_SEASON_START, LAST_SEASON_START, SLEEP_SECONDS, season_label

RAW_DIR = Path("data/raw")


def fetch_playoff_stats(season):
    """Playoff games played, wins and losses for every team that made the playoffs."""
    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Playoffs",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=60,
    )
    df = endpoint.get_data_frames()[0]
    out = df[["TEAM_ID", "TEAM_NAME", "GP", "W", "L"]].copy()
    out = out.rename(columns={"GP": "PO_GP", "W": "PO_W", "L": "PO_L"})
    out.insert(0, "SEASON", season)
    return out


def check_playoff_structure(df):
    """Every season: 16 teams made the playoffs, and 8 / 4 / 2 / 1 of them won at least
    1 / 2 / 3 / 4 series. Every series is best-of-7, so series won = playoff wins // 4.
    """
    for season, grp in df.groupby("SEASON"):
        series_won = grp["PO_W"] // 4
        counts = [len(grp)] + [int((series_won >= k).sum()) for k in (1, 2, 3, 4)]
        assert counts == [16, 8, 4, 2, 1], f"{season}: unexpected bracket counts {counts}"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []
    for start in range(FIRST_SEASON_START, LAST_SEASON_START + 1):
        season = season_label(start)
        print(f"Fetching playoffs {season} ...", end=" ", flush=True)
        try:
            frames.append(fetch_playoff_stats(season))
            print(f"{len(frames[-1])} teams")
        except Exception as e:
            failed.append(season)
            print(f"FAILED ({type(e).__name__})")
        time.sleep(SLEEP_SECONDS)
    if failed:
        raise SystemExit(f"Failed seasons {failed}. Re-run before checking.")

    df = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "team_playoffs_2010_2025.csv"
    df.to_csv(out_path, index=False)
    check_playoff_structure(df)
    print(f"\nSaved {len(df)} rows to {out_path}; bracket structure checks passed")


if __name__ == "__main__":
    main()
