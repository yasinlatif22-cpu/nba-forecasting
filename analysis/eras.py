from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from models.walk_forward import walk_forward_simple

DATA_PATH = Path("data/processed/team_season_dataset.csv")
GAMES = 82

# Era is set by the START YEAR of the season being predicted.
ERAS = [("2011-2015", 2011, 2015), ("2016-2020", 2016, 2020), ("2021-2025", 2021, 2025)]


def era_of(start_year):
    """Map a season's start year (e.g. 2017) to its era label."""
    for label, low, high in ERAS:
        if low <= start_year <= high:
            return label
    raise ValueError(f"No era defined for {start_year}")


def fit_clustered(frame, formula_cols):
    """OLS of win % on the given columns, standard errors clustered by team."""
    X = sm.add_constant(frame[formula_cols])
    return sm.OLS(frame["ACTUAL_W_PCT"], X).fit(
        cov_type="cluster", cov_kwds={"groups": frame["TEAM_ID"].to_numpy()}, use_t=True
    )


def main():
    data = pd.read_csv(DATA_PATH)
    data["ERA"] = data["START_YEAR"].map(era_of)
    assert data.groupby("ERA").size().eq(150).all(), "Each era should be 5 seasons x 30 teams"

    # 1. The same regression, fitted separately in each era.
    rows = []
    for era, grp in data.groupby("ERA"):
        fit = fit_clustered(grp, ["PREV_W_PCT"])
        low, high = fit.conf_int().loc["PREV_W_PCT"]
        rows.append({
            "era": era, "n": int(fit.nobs),
            "slope": fit.params["PREV_W_PCT"], "ci_low": low, "ci_high": high,
            "intercept": fit.params["const"],
            "resid_sd_wins": fit.resid.std() * GAMES, "R2": fit.rsquared,
        })
    print("Slope of next-season win % on previous win %, by era:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 2. Formal test: are the slopes equal across eras?
    full = smf.ols("ACTUAL_W_PCT ~ PREV_W_PCT * C(ERA)", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["TEAM_ID"].to_numpy()}, use_t=True
    )
    names = [n for n in full.params.index if n.startswith("PREV_W_PCT:")]
    R = np.zeros((len(names), len(full.params)))
    for row, name in enumerate(names):
        R[row, list(full.params.index).index(name)] = 1.0
    test = full.wald_test(R, use_f=True, scalar=True)
    print(f"\nTest that all slopes are equal: F = {float(test.statistic):.2f}, p = {float(test.pvalue):.3f}")

    # 3. Out-of-sample error by era (the first era is training-only, so it has no test rows).
    wf = walk_forward_simple(data)
    wf["ERA"] = wf["SEASON"].str[:4].astype(int).map(era_of)
    wf["naive_miss"] = (wf["ACTUAL_W_PCT"] - wf["PREV_W_PCT"]).abs() * GAMES
    wf["fitted_miss"] = (wf["ACTUAL_W_PCT"] - wf["PRED_FITTED"]).abs() * GAMES
    print("\nMean absolute miss in wins, walk-forward, by era:")
    print(wf.groupby("ERA")[["naive_miss", "fitted_miss"]].mean().round(2))


if __name__ == "__main__":
    main()
