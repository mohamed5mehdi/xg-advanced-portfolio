import sys
import pandas as pd
import joblib

sys.path.insert(0, 'src/viz')  # ajuster si l'arborescence diffère
from features import process_events

# --- 1. Charger l'artefact modèle ---
artifact = joblib.load('models/xgb_calibrated_v1.joblib')
model = artifact['model']
feature_columns = artifact['feature_columns']
cat_cols = artifact['categorical_columns']

# --- 2. Construire le showcase (finale C1) via la même pipeline que le corpus ---
raw_showcase = pd.read_parquet('data/raw/showcase/comp16_match18243.parquet')
showcase_features = process_events(raw_showcase)
print(f"Finale C1 -> {len(showcase_features)} tirs après process_events() "
      f"(penalties, y compris séance de tab, exclus par construction — cohérent "
      f"avec le reste du corpus)")

# --- 3. Combiner train + test + showcase ---
train = pd.read_parquet('data/processed/train_features.parquet')
test = pd.read_parquet('data/processed/test_features.parquet')
all_shots = pd.concat([train, test, showcase_features], ignore_index=True)

# --- 4. Encoder et aligner strictement sur les colonnes d'entraînement ---
X = all_shots.copy()
for c in cat_cols:
    X[c] = X[c].astype(str)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

missing_cols = set(feature_columns) - set(X_encoded.columns)
extra_cols = set(X_encoded.columns) - set(feature_columns) - set(all_shots.columns)
print(f"Colonnes manquantes imputées à 0 (catégories absentes de ce sous-ensemble): {missing_cols}")
print(f"Colonnes inédites ignorées (catégories jamais vues à l'entraînement): {extra_cols}")

X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)

# --- 5. Scorer ---
all_shots['model_xg'] = model.predict_proba(X_aligned)[:, 1]

# --- 6. Sauver ---
all_shots.to_parquet('data/processed/all_shots_scored.parquet', index=False)

target_ids = {266310: "Deportivo-Barcelone", 3754214: "Aston Villa-Liverpool",
              3901184: "Marseille-Rennes", 18243: "Finale C1 (showcase)"}
for mid, name in target_ids.items():
    sub = all_shots[all_shots['match_id'] == mid]
    print(f"\n{name} (match_id={mid}) — {len(sub)} tirs")
    print(sub.groupby('team')[['statsbomb_xg', 'model_xg', 'is_goal']].agg(
        n=('is_goal', 'size'), buts=('is_goal', 'sum'),
        xg_statsbomb=('statsbomb_xg', 'sum'), xg_model=('model_xg', 'sum')
    ))
