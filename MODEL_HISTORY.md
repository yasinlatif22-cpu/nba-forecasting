# Model history

Every model version, what it contains, and how it scored. All numbers come from scripts in this repo.

## v1.0 (frozen before the 2026-27 season; see CREATED_UTC in the prediction file)

**Model:** next-season win % = intercept + slope x previous-season win %, fitted on all 450 team-seasons from 2011-12 to 2025-26. Slope 0.604 (95% CI 0.550 to 0.658, clustered by team), intercept 0.198. The slope is far from 1 (t = -15), so regression to the mean is statistically distinguishable from "teams repeat last year."

**Inputs:** previous-season win % only. It ignores the 2026 offseason (trades, free agency, draft) and injuries.

**Walk-forward results** (train on all earlier seasons, predict the next one; test seasons 2016-17 to 2025-26, 300 team-seasons; wins on an 82-game scale):

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Everyone .500 | 10.00 | 12.02 | -0.00 |
| Naive: last season's win % | 8.45 | 10.99 | 0.16 |
| **v1.0: win % on previous win %** | **7.79** | **9.79** | **0.34** |

**Tried and not clearly better:** net rating, an offense/defense split, extra four-factor stats, and league-relative versions of these. They scored MAE 7.69 to 7.75 and RMSE 9.73 to 9.83, versus 7.79 and 9.79 for v1.0, which is indistinguishable on 300 rows. The simpler model was kept.

**Prediction file:** `predictions/2026_27_predictions_v1.0.csv`. Never edited after creation.

**Largest out-of-sample misses** (win % x 82):

| Season | Team | Predicted | Actual | Miss |
|---|---|---:|---:|---:|
| 2019-20 | Golden State Warriors | 51.4 | 18.9 | -32.5 |
| 2018-19 | Cleveland Cavaliers | 46.9 | 19.0 | -27.9 |
| 2020-21 | Houston Rockets | 46.8 | 19.4 | -27.5 |
| 2025-26 | Indiana Pacers | 46.5 | 19.0 | -27.5 |
| 2025-26 | San Antonio Spurs | 36.7 | 62.0 | +25.3 |
| 2024-25 | New Orleans Pelicans | 45.9 | 21.0 | -24.9 |
| 2020-21 | Toronto Raptors | 53.5 | 30.8 | -22.7 |
| 2023-24 | Washington Wizards | 37.4 | 15.0 | -22.4 |

**2026-27 result:** to be filled in after the season.
