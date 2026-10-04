import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

RAW_DIR = Path("data/raw")
FIRST_SEASON_START = 2010   # the 2010-11 season
LAST_SEASON_START = 2025    # the 2025-26 season, most recent completed
SLEEP_SECONDS = 2


def season_label(start_year):
    """2010 -> '2010-11', the format the NBA API expects."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_team_stats(season):
    """Download one season of team advanced stats as a DataFrame."""
    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        timeout=60,
    )
    df = endpoint.get_data_frames()[0]
    df.insert(0, "SEASON", season)
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []

    for start in range(FIRST_SEASON_START, LAST_SEASON_START + 1):
        season = season_label(start)
        print(f"Fetching {season} ...", end=" ", flush=True)
        try:
            df = fetch_team_stats(season)
            frames.append(df)
            print(f"{len(df)} teams")
        except Exception as e:
            failed.append(season)
            print(f"FAILED ({type(e).__name__})")
        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise SystemExit("No seasons downloaded. Check your connection.")

    all_seasons = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "team_advanced_2010_2025.csv"
    all_seasons.to_csv(out_path, index=False)

    print(f"\nSaved {len(all_seasons)} rows to {out_path}")
    if failed:
        print(f"Failed seasons (re-run to retry): {failed}")


if __name__ == "__main__":
    main()
