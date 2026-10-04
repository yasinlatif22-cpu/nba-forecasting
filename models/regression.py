from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from analysis.features import add_relative_features

DATA_PATH = Path("data/processed/team_season_dataset.csv")
FIG_DIR = Path("figures")
GAMES = 82


def fit_ols(data, features):
    """OLS of this season's win % on the given PREV_ columns.

    Input: the dataset and a list of feature names.
    Output: (default_fit, clustered_fit). Coefficients are identical; only the
    standard errors differ. 'Clustered' lets each team's errors be correlated
    across seasons, which they are (the same team appears every year).
    """
    X = sm.add_constant(data[features])
    y = data["ACTUAL_W_PCT"]
    default = sm.OLS(y, X).fit()
    clustered = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": data["TEAM_ID"].to_numpy()}, use_t=True
    )
    return default, clustered


def show(name, default, clustered):
    ci = clustered.conf_int()
    table = pd.DataFrame({
        "coef": default.params,
        "se_default": default.bse,
        "se_clustered": clustered.bse,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
    })
    print(f"\n{name}")
    print(table.round(3).to_string())
    print(f"R2={default.rsquared:.3f}  adj R2={default.rsquared_adj:.3f}  n={int(default.nobs)}")


def main():
    data = add_relative_features(pd.read_csv(DATA_PATH))

    # Model A: the v1.0 model. Is the slope really below 1?
    a_def, a_cl = fit_ols(data, ["PREV_W_PCT"])
    show("A: win % ~ previous win %", a_def, a_cl)
    print(a_cl.t_test("PREV_W_PCT = 1"))

    # Model B: do the offense and defense coefficients cancel the league drift?
    b_def, b_cl = fit_ols(data, ["PREV_OFF_RATING", "PREV_DEF_RATING"])
    show("B: win % ~ previous off rating + previous def rating (raw)", b_def, b_cl)
    print(b_cl.t_test("PREV_OFF_RATING + PREV_DEF_RATING = 0"))

    # Model C: two near-duplicate inputs, to see multicollinearity.
    c_def, c_cl = fit_ols(data, ["PREV_W_PCT", "PREV_NET_RATING"])
    show("C: win % ~ previous win % + previous net rating", c_def, c_cl)
    X = sm.add_constant(data[["PREV_W_PCT", "PREV_NET_RATING"]])
    vif = {col: variance_inflation_factor(X.to_numpy(), i) for i, col in enumerate(X.columns)}
    print("VIF:", {k: round(float(v), 1) for k, v in vif.items() if k != "const"})

    # Residuals for Model A.
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(a_def.fittedvalues * GAMES, a_def.resid * GAMES, alpha=0.4)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Fitted wins")
    ax.set_ylabel("Residual (actual - fitted, wins)")
    ax.set_title("Model A residuals")
    fig.savefig(FIG_DIR / "model_a_residuals.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/model_a_residuals.png")


if __name__ == "__main__":
    main()
