# NBA preseason forecasting

Forecasts each NBA team's regular-season wins before the season starts, freezes the forecast, and scores it against what happens. The aim is a record of predictions made in advance, not a one-time backtest.

<!-- Add a Motivation section here, 2-3 sentences in your own words. -->

## Status

Version 1.0 is complete and frozen for the 2026-27 season. It uses a single input, last season's win percentage. Payroll, roster, playoff and betting-market work is planned and not yet done; it is listed under Future work and not described as a result.

## Research question

Can information available before a season starts predict that season's regular-season wins, and how much does each additional piece of information improve the forecast? Later versions will ask whether team payroll adds anything, and how the forecasts compare with betting-market expectations. I do not assume the model will beat the market.

## Data

- **Source:** team advanced statistics from stats.nba.com, retrieved with the unofficial `nba_api` package (`collect_data.py`). Regular season, 2010-11 through 2025-26, 30 teams per season: 480 team-seasons.
- **Shortened seasons:** 2011-12 (66 games), 2019-20 (64 to 75 games, unequal across teams) and 2020-21 (72 games). The target is win percentage scaled to 82 games, not raw wins. 2012-13 has two teams at 81 games.
- **Team identity:** teams are joined on `TEAM_ID`, not name. Franchise names change (the Charlotte Bobcats became the Hornets under the same ID).

## Method

**No look-ahead.** `build_dataset.py` builds one row per team-season. Columns prefixed `PREV_` come from the previous season's final statistics and are the only allowed inputs. Columns prefixed `ACTUAL_` describe the season being predicted and are used only as the target. This gives 450 rows (2011-12 to 2025-26). A unit test overwrites the target season's data and asserts that the inputs do not change.

**Walk-forward validation.** For each test season, models are fitted only on earlier seasons (at least five), then predict the test season. Test seasons are 2016-17 through 2025-26: 300 team-seasons. There is no random train/test split.

**Metrics.** MAE, RMSE and R-squared, with wins on an 82-game scale. The metrics are implemented by hand and cross-checked against scikit-learn.

## Results (walk-forward, 2016-17 to 2025-26)

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Everyone .500 | 10.00 | 12.02 | -0.00 |
| Naive: last season's win % | 8.45 | 10.99 | 0.16 |
| **v1.0: fitted line on last season's win %** | **7.79** | **9.79** | **0.34** |

- The fitted line beats the naive forecast in 9 of the 10 test seasons. The exception is 2023-24.
- The fitted slope is 0.604 (95% CI 0.550 to 0.658, standard errors clustered by team), so about 40% of a team's distance from .500 disappears the following season. The slope is far from 1 (t = -15), so regression to the mean is distinguishable from "teams repeat last year." This describes a pattern; it does not separate luck from real roster change.
- Adding net rating, an offense/defense split, or four-factor statistics (raw or league-relative) gave MAE 7.69 to 7.75 and RMSE 9.73 to 9.83. That is not distinguishable from 7.79 and 9.79 on 300 rows, so the one-variable model was kept.
- The league's average offensive rating rose from about 106 in 2010-11 to about 114 by 2023-24, and the average eFG% from .499 (2010-11) to .543 (2024-25). Converting inputs to league-relative values did not change forecast accuracy. The raw offense and defense coefficients came out equal and opposite (+0.019 and -0.019; test that they sum to zero, p = 0.89), so the shared drift cancels in the difference.
- Win percentage and net rating correlate at 0.97. Including both gave a VIF of 14.6 and a win-percentage coefficient whose confidence interval includes zero, while R-squared barely moved. Individual coefficients are not interpretable in that model.
- The slope is similar in three five-season eras: 0.644 (2011-2015), 0.571 (2016-2020) and 0.594 (2021-2025), with overlapping confidence intervals; a test that all three are equal gives p = 0.68. With 150 team-seasons per era this can only detect fairly large changes, so it does not prove stability. Out-of-sample error was higher in 2021-2025 than in 2016-2020 (fitted MAE 8.35 vs 7.23 wins; naive 8.84 vs 8.05), a difference of roughly 1.6 standard errors. I have not identified a cause.
- Seven of the eight largest out-of-sample misses are collapses (predicted well above actual). I have not yet checked the causes systematically.

Full detail is in [MODEL_HISTORY.md](MODEL_HISTORY.md).

## Frozen forecast

`predictions/2026_27_predictions_v1.0.csv` contains predicted wins and an 80% interval for all 30 teams, made before the 2026-27 season began. The interval comes from the 10th and 90th percentiles of past walk-forward errors (about -12.5 to +11.6 wins). The file records its own creation time (`CREATED_UTC`) and model parameters, and is never edited after creation. `predict_next_season.py` refuses to overwrite an existing prediction file.

## Limitations

- v1.0 sees only last season's win percentage. It ignores the offseason (trades, free agency, the draft), injuries and coaching changes.
- Roughly fifteen model variants were compared on the same 300 test rows, so the best-scoring one is optimistic, and differences of about 0.1 wins are not meaningful.
- The 80% intervals are estimated from the same walk-forward errors they describe. Out-of-sample coverage has not been checked.
- Ten test seasons of 30 teams each is a small sample, and rows from the same team are not independent.
- 2019-20 has unequal game counts across teams, which makes that season noisier.
- Only one forecast has been frozen so far. Whether it holds up is unknown until the 2026-27 season ends.

## Repository layout

```
collect_data.py          download team statistics from nba_api
build_dataset.py         leakage-safe team-season dataset
predict_next_season.py   fit v1.0 and write the frozen forecast
analysis/features.py     league-relative features
models/                  baseline, walk-forward validation, feature comparisons, regression
tests/                   pytest suite
data/raw, data/processed downloaded and derived data
predictions/             frozen forecasts (never overwritten)
figures/                 charts
MODEL_HISTORY.md         what each version contains and how it scored
```

## How to run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python collect_data.py
python build_dataset.py
python -m models.baseline
python -m models.walk_forward
python -m models.feature_models
python -m models.relative_features
python -m models.regression
python -m pytest
```

`predict_next_season.py` writes the frozen forecast once per season and model version.

## Annual update (currently manual)

1. After the season ends: raise `LAST_SEASON_START` in `collect_data.py`, then re-run `collect_data.py` and `build_dataset.py`.
2. Compare the frozen forecast with actual wins and record the result in `MODEL_HISTORY.md`. A scoring script for this step is not yet written.
3. Before the next season: update `TARGET_SEASON`, `LAST_COMPLETED` and `MODEL_VERSION` in `predict_next_season.py`, run it, and commit and push before the first game. Any model change gets a new version and a new file; earlier forecasts are not touched.

File names and season constants are hard-coded for now.

## Future work

- Team payroll as a share of the salary cap, with two separate timing rules: same-season payroll for the question "does spending buy wins," and opening-night payroll for forecasting.
- Roster continuity and star availability.
- Playoff and championship probabilities, with calibration checks.
- Comparison with preseason betting markets.
- A SQLite store and a small dashboard.

## Technology

Python, pandas, NumPy, scikit-learn, statsmodels, matplotlib, pytest, nba_api.
