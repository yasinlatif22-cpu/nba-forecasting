from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

DATA_PATH = Path("data/processed/team_season_playoffs.csv")
MIN_TRAIN_SEASONS = 5
FEATURES = ["PREV_W_PCT"]

# label: (target column, calibration bins). The base rate is fixed by the bracket: K / 30.
TARGETS = {
    "Made playoffs": ("ACTUAL_MADE_PLAYOFFS", 16, 5),
    "Conference finals": ("ACTUAL_CONF_FINALS", 4, 3),
    "Finals": ("ACTUAL_FINALS", 2, 3),
    "Champion": ("ACTUAL_CHAMPION", 1, 3),
}


def brier(y, p):
    """Mean squared gap between the predicted probability and the 0/1 outcome."""
    return np.mean((p - y) ** 2)


def log_loss(y, p, eps=1e-12):
    """Average negative log of the probability given to what actually happened."""
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward_logit(data, target, features):
    """Expanding-window logistic regression: fit on earlier seasons, predict the next.

    Input: the labeled dataset, a target column, and PREV_ feature columns.
    Output: a Series of predicted probabilities for the test seasons.
    """
    assert all(c.startswith("PREV_") for c in features), "Inputs must be PREV_ columns"
    years = sorted(data["START_YEAR"].unique())
    pieces = []
    for test_year in years[MIN_TRAIN_SEASONS:]:
        train = data[data["START_YEAR"] < test_year]
        test = data[data["START_YEAR"] == test_year]
        X_train = sm.add_constant(train[features], has_constant="add")
        X_test = sm.add_constant(test[features], has_constant="add")
        fit = sm.Logit(train[target], X_train).fit(disp=0)
        pieces.append(fit.predict(X_test))
    return pd.concat(pieces)


def calibration_table(y, p, bins):
    """Bin the predictions and compare the average prediction with the actual frequency."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        n=("y", "size"), avg_pred=("p", "mean"), actual=("y", "mean")
    )


def main():
    data = pd.read_csv(DATA_PATH)
    first_test = sorted(data["START_YEAR"].unique())[MIN_TRAIN_SEASONS]
    test = data[data["START_YEAR"] >= first_test]

    rows, calibrations = [], {}
    for label, (target, slots, bins) in TARGETS.items():
        pred = walk_forward_logit(data, target, FEATURES)
        assert pred.index.equals(test.index), "Test rows misaligned"
        y, p = test[target].to_numpy(), pred.to_numpy()
        base = np.full_like(p, slots / 30)
        season_sums = pred.groupby(test["SEASON"]).sum()
        rows.append({
            "target": label, "events": int(y.sum()), "n": len(y),
            "brier": brier(y, p), "brier_base": brier(y, base),
            "skill": 1 - brier(y, p) / brier(y, base),
            "logloss": log_loss(y, p), "logloss_base": log_loss(y, base),
            "auc": roc_auc_score(y, p),
            "avg_sum": season_sums.mean(), "expected_sum": slots,
        })
        calibrations[label] = calibration_table(y, p, bins)

    print(f"Walk-forward, {len(test)} team-seasons. 'base' = same probability for every team.")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for label, table in calibrations.items():
        print(f"\nCalibration: {label}")
        print(table.round(3).to_string())

    print("\nFull-sample logistic fits (odds multiplier for +0.10 in last season's win %):")
    for label, (target, _, _) in TARGETS.items():
        fit = sm.Logit(data[target], sm.add_constant(data[FEATURES])).fit(disp=0)
        coef = fit.params["PREV_W_PCT"]
        print(f"  {label}: coefficient {coef:.2f}, odds x{np.exp(coef * 0.1):.2f}")


if __name__ == "__main__":
    main()
