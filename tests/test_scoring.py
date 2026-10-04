import numpy as np
import pandas as pd
import pytest

from predict_playoffs import SHORT
from score_season import score

SEASON, VERSION = "2099-00", "v1.0"
BRACKET = [16, 13, 10, 9, 6, 6, 5, 4, 3, 2, 2, 1, 1, 0, 3, 0]   # a valid 16 / 8 / 4 / 2 / 1 bracket


def write_files(tmp_path, complete=True):
    """A synthetic finished season, plus a frozen forecast that is off by exactly 3 wins."""
    ids = list(range(1, 31))
    wpct = [(20 + i) / 82 for i in range(30)]                       # actual wins 20 ... 49
    raw = pd.DataFrame({"SEASON": SEASON, "TEAM_ID": ids,
                        "TEAM_NAME": [f"T{i}" for i in ids], "W_PCT": wpct})
    po_w = BRACKET if complete else BRACKET[1:]                      # drop the champion if incomplete
    po = pd.DataFrame({"SEASON": SEASON, "TEAM_ID": ids[:len(po_w)], "PO_W": po_w})
    actual = np.array(wpct) * 82

    stem = SEASON.replace("-", "_")
    wins = pd.DataFrame({"TEAM_ID": ids, "TEAM_NAME": raw["TEAM_NAME"], "PREV_W_PCT": 0.5,
                         "PRED_W": actual + 3, "PRED_W_LOW80": actual - 2, "PRED_W_HIGH80": actual + 8})
    probs = pd.DataFrame({"TEAM_ID": ids, "TEAM_NAME": raw["TEAM_NAME"]})
    for name, slots in {"PLAYOFFS": 16, "CONF_FINALS": 4, "FINALS": 2, "TITLE": 1}.items():
        probs[f"P_{name}_RAW"] = slots / 30
        probs[f"P_{name}_NORM"] = slots / 30
    raw.to_csv(tmp_path / "raw.csv", index=False)
    po.to_csv(tmp_path / "po.csv", index=False)
    wins.to_csv(tmp_path / f"{stem}_predictions_{VERSION}.csv", index=False)
    probs.to_csv(tmp_path / f"{stem}_playoff_probabilities_{VERSION}.csv", index=False)


def run(tmp_path):
    return score(SEASON, VERSION, pred_dir=tmp_path, raw_path=tmp_path / "raw.csv",
                 playoffs_path=tmp_path / "po.csv", out_dir=tmp_path / "results")


def test_scoring_matches_hand_calculation(tmp_path):
    write_files(tmp_path)
    wins_table, coverage, prob_table = run(tmp_path)
    model = wins_table[wins_table["model"] == "Model"].iloc[0]
    assert model["MAE"] == pytest.approx(3.0)           # every forecast is 3 wins too high
    assert model["RMSE"] == pytest.approx(3.0)
    assert coverage == pytest.approx(1.0)               # every actual sits inside the interval
    # A constant base-rate forecast has exactly zero skill against itself.
    assert prob_table["skill"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert (tmp_path / "results" / "2099_00_v1.0_wins_scores.csv").exists()


def test_scoring_refuses_an_incomplete_season(tmp_path):
    write_files(tmp_path, complete=False)
    with pytest.raises(SystemExit):
        run(tmp_path)
