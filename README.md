# xg-advanced-portfolio

A calibrated Expected Goals (xG) model built on StatsBomb open data — designed to be
**honest about its scope**, not to artificially beat a professional benchmark.

## TL;DR

- **Data**: StatsBomb open data, 2015/16 season — La Liga, Premier League, Serie A
  (train, 28,765 shots) → holdout test on Ligue 1 (8,723 shots).
- **Model**: XGBoost, Optuna-tuned (15 trials, log-loss objective, `GroupKFold(3)` on
  `match_id`), `max_depth` deliberately capped at 2–4 for explainability, then
  isotonic-calibrated (`CalibratedClassifierCV`, `GroupKFold(5)` on `match_id`).
- **Result**: **AUC 0.8007** on the Ligue 1 holdout vs. StatsBomb's own benchmark of
  **0.8041** — a gap of ~0.0033, left **deliberately unclosed**. The goal here isn't to
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
  exists but the keeper isn't identified (32 / 37,488 shots, 0.085% — open decision:
  keep vs. correct to NaN + retrain).

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

## Reproducing this

```powershell
python src/data_collection.py
python src/features.py
python src/train.py
python score_all_matches.py
```

## Roadmap

- Phase 4: Streamlit dashboard MVP (`match_whitelist.json` as source of truth).
- Phase 5: document the shot-coordinate attack-direction normalization mechanism
  (confirmed empirically, not yet located in source) and the goalkeeper-fallback
  decision.

## Author

Mohamed Mehdi El Jadoui — software engineer, aspiring data scientist in football
analytics.