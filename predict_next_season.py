from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.walk_forward import walk_forward_simple

RAW_PATH = Path("data/raw/team_advanced_2010_2025.csv")
DATA_PATH = Path("data/processed/team_season_dataset.csv")
PRED_DIR = Path("predictions")
GAMES = 82
MODEL_VERSION = "v1.0"
TARGET_SEASON = "2026-27"
LAST_COMPLETED = "2025-26"


def empirical_interval(data):
    """10th and 90th percentile of past walk-forward errors, in wins.

    Input: the training dataset. Output: (low, high) offsets to add to a prediction.
    No normality assumption: it just asks how far off the model has actually been.
    """
    wf = walk_forward_simple(data)
    errors = (wf["ACTUAL_W_PCT"] - wf["PRED_FITTED"]) * GAMES
    return np.quantile(errors, 0.10), np.quantile(errors, 0.90)


def main():
    out_path = PRED_DIR / f"{TARGET_SEASON.replace('-', '_')}_predictions_{MODEL_VERSION}.csv"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists. Predictions are never overwritten.")

    raw = pd.read_csv(RAW_PATH)
    data = pd.read_csv(DATA_PATH)

    # Leakage guards: nothing from the target season may be in the training data.
    assert TARGET_SEASON not in set(data["SEASON"]), "Training data contains the target season"
    last = raw[raw["SEASON"] == LAST_COMPLETED]
    assert len(last) == 30 and last["TEAM_ID"].is_unique, "Expected 30 unique teams"
    # The league is zero-sum (every game has a winner and a loser), so win % must average .500.
    assert abs(last["W_PCT"].mean() - 0.5) < 0.01, "Last season's win % should average .500"

    # Fit on every completed (prior season -> next season) pair we have.
    slope, intercept = np.polyfit(data["PREV_W_PCT"], data["ACTUAL_W_PCT"], 1)
    low, high = empirical_interval(data)

    pred_wins = (intercept + slope * last["W_PCT"].to_numpy()) * GAMES

    pred = pd.DataFrame({
        "TEAM_ID": last["TEAM_ID"].to_numpy(),
        "TEAM_NAME": last["TEAM_NAME"].to_numpy(),
        "TARGET_SEASON": TARGET_SEASON,
        "PREV_W_PCT": last["W_PCT"].to_numpy(),
        "PRED_W": pred_wins,
        "PRED_W_LOW80": np.clip(pred_wins + low, 0, GAMES),
        "PRED_W_HIGH80": np.clip(pred_wins + high, 0, GAMES),
        "MODEL_VERSION": MODEL_VERSION,
        "SLOPE": slope,
        "INTERCEPT": intercept,
        "TRAIN_ROWS": len(data),
        "CREATED_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }).round(3)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)

    show = pred.sort_values("PRED_W", ascending=False)
    print(f"Saved {len(pred)} predictions to {out_path}")
    print(f"slope={slope:.3f} intercept={intercept:.3f} trained on {len(data)} rows")
    print(f"80% interval offsets: {low:+.1f} to {high:+.1f} wins")
    print(f"mean predicted wins: {pred_wins.mean():.1f}\n")
    print(show[["TEAM_NAME", "PREV_W_PCT", "PRED_W", "PRED_W_LOW80", "PRED_W_HIGH80"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
