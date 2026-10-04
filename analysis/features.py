import pandas as pd

REL_STATS = ["OFF_RATING", "DEF_RATING", "PACE", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT"]


def add_relative_features(data):
    """Add PREV_<stat>_REL: a team's previous-season stat minus that season's league average.

    Input: the team-season dataset (needs PREV_SEASON and the PREV_<stat> columns).
    Output: a copy with six new PREV_..._REL columns.
    The league average comes only from the PREV_SEASON, which is finished before the
    season being predicted starts, so this adds no look-ahead information.
    """
    out = data.copy()
    for stat in REL_STATS:
        col = f"PREV_{stat}"
        league_mean = out.groupby("PREV_SEASON")[col].transform("mean")
        out[f"{col}_REL"] = out[col] - league_mean
    return out
