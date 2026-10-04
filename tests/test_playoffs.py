import numpy as np
import pandas as pd
import pytest

from analysis.playoffs import LABELS, add_playoff_labels, check_labels
from models.classification import brier, log_loss
from predict_playoffs import shift_to_total


def test_brier_and_log_loss_match_hand_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.3, 0.6, 0.1])
    # Brier: (0.2^2 + 0.3^2 + 0.4^2 + 0.1^2) / 4 = 0.075
    assert brier(y, p) == pytest.approx(0.075)
    # Log loss: -(ln 0.8 + ln 0.7 + ln 0.6 + ln 0.9) / 4 = 0.2990
    assert log_loss(y, p) == pytest.approx(0.2990, abs=1e-3)


def test_playoff_labels_follow_series_won():
    data = pd.DataFrame({"SEASON": "2020-21", "TEAM_ID": [1, 2, 3, 4, 5]})
    playoffs = pd.DataFrame({
        "SEASON": "2020-21", "TEAM_ID": [1, 2, 5],
        "PO_W": [16, 13, 0],            # champion, Finals loser, swept in round one
    })
    out = add_playoff_labels(data, playoffs)
    assert len(out) == 5                                   # no rows added or dropped
    assert out["ACTUAL_SERIES_WON"].tolist() == [4, 3, 0, 0, 0]
    assert out["ACTUAL_CHAMPION"].tolist() == [1, 0, 0, 0, 0]
    assert out["ACTUAL_FINALS"].tolist() == [1, 1, 0, 0, 0]
    # A swept team still made the playoffs; a team that is absent from the table did not.
    assert out["ACTUAL_MADE_PLAYOFFS"].tolist() == [1, 1, 0, 0, 1]


def test_label_check_rejects_a_wrong_bracket():
    bad = pd.DataFrame({"SEASON": ["2020-21"] * 3, **{label: [0, 0, 0] for label in LABELS}})
    with pytest.raises(AssertionError):
        check_labels(bad)


def test_shift_to_total_hits_the_total_and_keeps_the_ranking():
    p = np.array([0.50, 0.30, 0.10, 0.05, 0.05])
    out = shift_to_total(p, 2.0)
    assert out.sum() == pytest.approx(2.0)
    assert ((out > 0) & (out < 1)).all()                  # a valid probability for every team
    assert (np.argsort(-out, kind="stable") == np.argsort(-p, kind="stable")).all()


def test_existing_playoff_file_is_never_overwritten(tmp_path, monkeypatch):
    import predict_playoffs as pp

    monkeypatch.setattr(pp, "PRED_DIR", tmp_path)
    out = tmp_path / f"{pp.TARGET_SEASON.replace('-', '_')}_playoff_probabilities_{pp.MODEL_VERSION}.csv"
    out.write_text("frozen")
    with pytest.raises(SystemExit):
        pp.main()
    assert out.read_text() == "frozen"
