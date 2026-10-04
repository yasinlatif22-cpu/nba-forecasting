from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit, logit

from models.classification import FEATURES, TARGETS

RAW_PATH = Path("data/raw/team_advanced_2010_2025.csv")
DATA_PATH = Path("data/processed/team_season_playoffs.csv")
PRED_DIR = Path("predictions")
MODEL_VERSION = "v1.0"
TARGET_SEASON = "2026-27"
LAST_COMPLETED = "2025-26"
SHORT = {"Made playoffs": "PLAYOFFS", "Conference finals": "CONF_FINALS",
         "Finals": "FINALS", "Champion": "TITLE"}


def shift_to_total(p, total):
    """Shift every team's log-odds by one common amount so the probabilities sum to `total`.

    The bracket fixes the totals (16 playoff teams, 4 conference finalists, 2 finalists,
    1 champion), and that is known before the season, so this adds no future information.
    """
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))
    delta = brentq(lambda d: expit(z + d).sum() - total, -10, 10)
    return expit(z + delta)


def main():
    out_path = PRED_DIR / f"{TARGET_SEASON.replace('-', '_')}_playoff_probabilities_{MODEL_VERSION}.csv"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists. Predictions are never overwritten.")

    raw = pd.read_csv(RAW_PATH)
    data = pd.read_csv(DATA_PATH)

    # Leakage guards, same as the wins forecast.
    assert TARGET_SEASON not in set(data["SEASON"]), "Training data contains the target season"
    last = raw[raw["SEASON"] == LAST_COMPLETED]
    assert len(last) == 30 and last["TEAM_ID"].is_unique, "Expected 30 unique teams"
    assert abs(last["W_PCT"].mean() - 0.5) < 0.01, "Last season's win % should average .500"

    X_new = sm.add_constant(
        pd.DataFrame({"PREV_W_PCT": last["W_PCT"].to_numpy()}), has_constant="add"
    )
    out = pd.DataFrame({
        "TEAM_ID": last["TEAM_ID"].to_numpy(),
        "TEAM_NAME": last["TEAM_NAME"].to_numpy(),
        "TARGET_SEASON": TARGET_SEASON,
        "PREV_W_PCT": last["W_PCT"].to_numpy(),
    })
    for label, (target, slots, _) in TARGETS.items():
        X_train = sm.add_constant(data[FEATURES], has_constant="add")
        fit = sm.Logit(data[target], X_train).fit(disp=0)
        p = np.asarray(fit.predict(X_new))
        norm = shift_to_total(p, slots)
        assert abs(norm.sum() - slots) < 1e-6, f"{label}: normalized probabilities do not sum to {slots}"
        out[f"P_{SHORT[label]}_RAW"] = p
        out[f"P_{SHORT[label]}_NORM"] = norm
        print(f"{label}: raw probabilities sum to {p.sum():.2f}, normalized to {norm.sum():.2f}")

    out["MODEL_VERSION"] = MODEL_VERSION
    out["TRAIN_ROWS"] = len(data)
    out["CREATED_UTC"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = out.round(4)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}\n")
    show = out.sort_values("P_TITLE_NORM", ascending=False)
    print(show[["TEAM_NAME", "PREV_W_PCT", "P_PLAYOFFS_NORM", "P_FINALS_NORM",
                "P_TITLE_RAW", "P_TITLE_NORM"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
