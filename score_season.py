import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.playoffs import add_playoff_labels, check_labels
from models.baseline import evaluate
from models.classification import TARGETS, brier, log_loss
from predict_playoffs import SHORT

GAMES = 82
RAW_PATH = Path("data/raw/team_advanced_2010_2025.csv")
PLAYOFFS_PATH = Path("data/raw/team_playoffs_2010_2025.csv")


def load_actuals(raw, playoffs, season):
    """One row per team: actual wins on an 82-game scale plus the playoff labels.

    Refuses to continue unless the season is complete: 30 teams and a full
    16 / 4 / 2 / 1 playoff bracket. This stops a season in progress being scored.
    """
    reg = raw[raw["SEASON"] == season][["SEASON", "TEAM_ID", "TEAM_NAME", "W_PCT"]]
    if len(reg) != 30:
        raise SystemExit(f"No complete regular-season data for {season}; refusing to score.")
    labeled = add_playoff_labels(reg, playoffs)
    try:
        check_labels(labeled)
    except AssertionError as err:
        raise SystemExit(f"Playoffs for {season} look incomplete ({err}); refusing to score.")
    labeled["ACTUAL_WINS"] = labeled["W_PCT"] * GAMES
    return labeled


def score_wins(preds, actuals):
    """MAE / RMSE / R2 for the model and two baselines, plus 80% interval coverage."""
    m = preds.merge(actuals[["TEAM_ID", "ACTUAL_WINS"]], on="TEAM_ID", how="inner")
    assert len(m) == 30, "Predictions and actuals do not cover the same 30 teams"
    y = m["ACTUAL_WINS"].to_numpy()
    table = pd.DataFrame([
        evaluate("Model", y, m["PRED_W"].to_numpy()),
        evaluate("Naive: last season's win %", y, m["PREV_W_PCT"].to_numpy() * GAMES),
        evaluate("Everyone .500", y, np.full_like(y, GAMES / 2)),
    ])
    coverage = float(((y >= m["PRED_W_LOW80"]) & (y <= m["PRED_W_HIGH80"])).mean())
    return table, coverage, m


def score_probabilities(probs, actuals):
    """Brier score, skill and log loss for each playoff target, raw and normalized."""
    label_cols = [target for target, _, _ in TARGETS.values()]
    m = probs.merge(actuals[["TEAM_ID"] + label_cols], on="TEAM_ID", how="inner")
    assert len(m) == 30, "Probabilities and actuals do not cover the same 30 teams"
    rows = []
    for label, (target, slots, _) in TARGETS.items():
        y = m[target].to_numpy()
        base = np.full(len(y), slots / 30)
        for variant in ("RAW", "NORM"):
            p = m[f"P_{SHORT[label]}_{variant}"].to_numpy()
            rows.append({
                "target": label, "variant": variant,
                "brier": brier(y, p), "brier_base": brier(y, base),
                "skill": 1 - brier(y, p) / brier(y, base),
                "logloss": log_loss(y, p), "logloss_base": log_loss(y, base),
            })
    champ = m[m["ACTUAL_CHAMPION"] == 1].iloc[0]
    return pd.DataFrame(rows), champ["TEAM_NAME"], float(champ["P_TITLE_NORM"])


def score(season, version, pred_dir=Path("predictions"), raw_path=RAW_PATH,
          playoffs_path=PLAYOFFS_PATH, out_dir=Path("results")):
    """Score one frozen forecast against what actually happened, and save the results."""
    stem = season.replace("-", "_")
    wins_file = pred_dir / f"{stem}_predictions_{version}.csv"
    prob_file = pred_dir / f"{stem}_playoff_probabilities_{version}.csv"
    for path in (wins_file, prob_file):
        if not path.exists():
            raise SystemExit(f"Missing frozen forecast: {path}")

    actuals = load_actuals(pd.read_csv(raw_path), pd.read_csv(playoffs_path), season)
    wins_table, coverage, merged = score_wins(pd.read_csv(wins_file), actuals)
    prob_table, champ, champ_p = score_probabilities(pd.read_csv(prob_file), actuals)

    print(f"{season}, forecast {version}: wins forecast")
    print(wins_table.round(2).to_string(index=False))
    print(f"\n80% interval coverage: {coverage:.0%} of teams (about 80% expected)")
    merged["MISS"] = merged["ACTUAL_WINS"] - merged["PRED_W"]
    worst = merged.reindex(merged["MISS"].abs().sort_values(ascending=False).index).head(5)
    print("\nLargest misses (actual minus predicted wins):")
    print(worst[["TEAM_NAME", "PRED_W", "ACTUAL_WINS", "MISS"]].round(1).to_string(index=False))
    print("\nPlayoff and title probabilities (one season, so treat as a sanity check):")
    print(prob_table.round(3).to_string(index=False))
    print(f"\nChampion: {champ}, given a {champ_p:.1%} title probability (normalized).")

    out_dir.mkdir(parents=True, exist_ok=True)
    wins_table.to_csv(out_dir / f"{stem}_{version}_wins_scores.csv", index=False)
    prob_table.to_csv(out_dir / f"{stem}_{version}_probability_scores.csv", index=False)
    return wins_table, coverage, prob_table


def main():
    parser = argparse.ArgumentParser(description="Score a frozen forecast against actual results.")
    parser.add_argument("season", help="for example 2026-27")
    parser.add_argument("version", help="for example v1.0")
    args = parser.parse_args()
    score(args.season, args.version)


if __name__ == "__main__":
    main()
