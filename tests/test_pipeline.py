import numpy as np
import pandas as pd
import pytest

from build_dataset import PREV_COLS, build_dataset, check_no_leakage
from collect_data import season_label
from models.baseline import mae, r_squared, rmse


def make_raw():
    """Two teams, three seasons. Every cell is a distinct number, so any mix-up shows."""
    rows = []
    for team_id in (1, 2):
        for i, season in enumerate(["2010-11", "2011-12", "2012-13"]):
            base = team_id * 100 + i * 10
            row = {"SEASON": season, "TEAM_ID": team_id, "TEAM_NAME": f"T{team_id}",
                   "GP": 82, "W": 40 + i}
            for j, col in enumerate(PREV_COLS):
                row[col] = base + j
            rows.append(row)
    return pd.DataFrame(rows)


def test_season_label_formats():
    assert season_label(2010) == "2010-11"
    assert season_label(2025) == "2025-26"
    assert season_label(1999) == "1999-00"      # century wrap


def test_metrics_match_hand_calculation():
    actual = np.array([40.0, 50.0, 60.0])
    pred = np.array([42.0, 48.0, 66.0])
    assert mae(actual, pred) == pytest.approx(10 / 3)
    assert rmse(actual, pred) == pytest.approx((44 / 3) ** 0.5)
    assert r_squared(actual, pred) == pytest.approx(0.78)
    assert r_squared(actual, np.full(3, actual.mean())) == pytest.approx(0.0)


def test_previous_season_stats_line_up_with_the_right_team_and_year():
    out = build_dataset(make_raw())
    assert "2010-11" not in set(out["SEASON"])      # no prior season, so it drops out
    assert len(out) == 4                            # 2 teams x 2 seasons
    row = out[(out.TEAM_ID == 2) & (out.SEASON == "2012-13")].iloc[0]
    assert row["PREV_SEASON"] == "2011-12"
    assert row["PREV_W_PCT"] == 210                 # team 2, 2011-12
    assert row["ACTUAL_W_PCT"] == 220               # team 2, 2012-13


def test_changing_the_season_being_predicted_does_not_change_its_inputs():
    raw = make_raw()
    before = build_dataset(raw)
    tampered = raw.copy()
    tampered.loc[tampered["SEASON"] == "2012-13", PREV_COLS] = 999.0
    after = build_dataset(tampered)

    prev_cols = [c for c in before.columns if c.startswith("PREV_")]
    b = before[before.SEASON == "2012-13"][prev_cols].reset_index(drop=True)
    a = after[after.SEASON == "2012-13"][prev_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)
    # Sanity check: the tampering did reach the target column, so the test can fail.
    assert (after[after.SEASON == "2012-13"]["ACTUAL_W_PCT"] == 999.0).all()


def test_leakage_check_catches_a_bad_dataset():
    out = build_dataset(make_raw())
    check_no_leakage(out)                           # passes on good data
    bad = out.copy()
    bad["PREV_SEASON"] = bad["SEASON"]              # inputs from the same season
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def test_existing_prediction_file_is_never_overwritten(tmp_path, monkeypatch):
    import predict_next_season as p

    monkeypatch.setattr(p, "PRED_DIR", tmp_path)
    out = tmp_path / f"{p.TARGET_SEASON.replace('-', '_')}_predictions_{p.MODEL_VERSION}.csv"
    out.write_text("frozen")
    with pytest.raises(SystemExit):
        p.main()
    assert out.read_text() == "frozen"
