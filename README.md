# Expected Goals (xG) — calibrated model & Streamlit dashboard

A calibrated Expected Goals (xG) model built on StatsBomb open data — designed to be
**honest about its scope**, not to artificially beat a professional benchmark.
Results are explored in a five-page Streamlit dashboard.

**▶ Live demo: [xg-advanced.streamlit.app](https://xg-advanced.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3-3776AB) ![XGBoost](https://img.shields.io/badge/XGBoost-calibrated-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)

## TL;DR

- **Data**: StatsBomb open data, 2015/16 season — La Liga, Premier League, Serie A
  (train, 28,765 shots) → holdout test on Ligue 1 (8,723 shots).
- **Model**: XGBoost, Optuna-tuned (15 trials, log-loss objective, `GroupKFold(3)` on
  `match_id`), `max_depth` deliberately capped at 2–4 for explainability, then
  isotonic-calibrated (`CalibratedClassifierCV`, `GroupKFold(5)` on `match_id`).
- **Result**: **AUC 0.8024** on the Ligue 1 holdout vs. StatsBomb's own benchmark of
  **0.8041** — a gap of ~0.0017, left **deliberately unclosed**. The goal here isn't to
  beat a professional reference; it's to build a model that's honest about what it
  knows and doesn't know.

## Why calibration, not just accuracy

"Exact xG" doesn't exist. What can exist is a model that's well-calibrated on a defined
domain (professional European football, 2015/16 rules and tactics) and transparent
about every assumption. This project is built around that principle end-to-end.

## Data & scope decisions

| Role | Competition | Season | Shots |
|---|---|---|---|
| Train | La Liga | 2015/16 | included |
| Train | Premier League | 2015/16 | included |
| Train | Serie A | 2015/16 | included |
| Train (excluded) | Bundesliga | 2015/16 | 34 matches, single club — not representative |
| Test (holdout) | Ligue 1 | 2015/16 | 8,723 |

Bundesliga is excluded at model-loading time (`train.py`), not at collection — the raw
open-data file is under-collected (34 matches, ~13MB, one club only).

## Feature engineering

23 raw columns before encoding, including two geometrically-derived features:

- `distance`: Euclidean distance from shot location to goal center (120, 40).
- `angle`: angle subtended by the two posts (dot product of two vectors), posts fixed
  at y=36 / y=44.
- `defenders_in_cone`: number of opponents geometrically inside the shot-to-posts
  triangle (determinant-sign test, not a distance approximation).
- `gk_distance` / `gk_angle` / `gk_found`: goalkeeper freeze-frame features — **NaN
  when no freeze frame exists**, real values when the keeper is identified, and a
  documented fallback (goal-center coordinates) for the rare case where a freeze frame
  exists but the keeper isn't identified. **Update**: this fallback was corrected to
  honest NaN and the model retrained — measured effect: +0.00167 AUC
  (0.80072 → 0.80239), cutting the benchmark gap roughly in half (0.0033 → 0.0017).

**Methodological transparency note**: one-hot encoding (`pd.get_dummies`,
`drop_first=True`) is fit on train+test concatenated, then re-split. This guarantees no
missing category at test time but means the column *schema* (category names only —
not targets or numeric features) is built with knowledge of the test set's categorical
vocabulary. Common, defensible, disclosed here rather than hidden.

## Two real bugs found and fixed

1. **Pitch geometry**: penalty-arc direction was inverted between the left and right
   goals (`theta1`/`theta2` confusion in `matplotlib.patches.Arc`) — caught by manual
   geometric recalculation before execution, confirmed against the official docs.
2. **Feature collision**: `pass_technique`/`pass_height`/`pass_type` were silently
   falling back to `'None'` on 100% of shots due to a pandas `_x`/`_y` merge-suffix
   collision. Fixed by dropping those columns from the shots frame before the join.
   Real, measured gain: **+0.00136 AUC** (0.79935 → 0.80072).

## Three use cases

1. **Goalkeeper evaluation** — team-level "xG faced vs. goals conceded" (pre-shot
   proxy — see limitations below).
2. **Scouting** — shot volume vs. average shot quality, split by defensive pressure.
3. **Transfer diagnostics** — Delta xG (actual goals − cumulative xG) to flag
   over/underperforming finishers.

## Known limitations (disclosed on purpose)

- **No true PSxG**: the model is pre-shot (distance, angle, defenders in cone,
  goalkeeper geometry, pressure). A genuine post-shot xG (PSxG-GA, StatsBomb/FBref
  style) needs shot end-location as a feature — not currently in the pipeline.
- **No goalkeeper identity column**: `gk_distance`/`gk_angle`/`gk_found` are geometric,
  not nominative. Per-keeper attribution isn't possible without re-extracting keeper
  identity from raw freeze-frame data.
- **No minutes-played data**: shot volume can't currently be normalized per-90.

## Dashboard

Available online at **https://xg-advanced.streamlit.app** (the app may take a few seconds to wake up).

| Page | What it shows |
|---|---|
| Shot Explorer | Shot maps and Opta-style match reports |
| Goalkeeper xG Against | Team-level xG faced vs. goals conceded |
| Scouting Quadrant | Shot volume vs. average shot quality, split by pressure |
| Delta xG | Goals − cumulative xG, to flag over/under-performing finishers |
| Model Evaluation | ROC, precision-recall, calibration curve and SHAP importances |

Run it locally:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Reproducing the model

Run from the repository root:

```bash
pip install -r requirements.txt
python src/viz/data_collection.py   # StatsBomb open data → data/raw/
python src/viz/features.py          # feature engineering → data/processed/
python src/viz/train.py             # Optuna + XGBoost + calibration → models/, reports/
python scripts/score_all_matches.py # scores every shot for the dashboard
```

## Project structure

```
app/               Streamlit dashboard (entry point + 5 pages)
src/viz/           data collection, feature engineering, training, pitch & heatmap plots
src/analytics/     advanced metrics used by the dashboard
scripts/           maintenance scripts (re-scoring, refit, figure regeneration)
models/            calibrated model artifact
reports/           final metrics, test predictions, figures; history/ keeps pre-fix metrics
data/processed/    engineered features and scored shots
```

## Author

**Mohamed Mehdi El Jadaoui** — software engineer, data science & football analytics
[LinkedIn](https://www.linkedin.com/in/mehdi-el-jadaoui-2a7755407) · [GitHub](https://github.com/mohamed5mehdi)
