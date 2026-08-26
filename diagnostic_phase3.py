import os
import pandas as pd

train = pd.read_parquet('data/processed/train_features.parquet')
test = pd.read_parquet('data/processed/test_features.parquet')

print("=== Répartition pass_technique / pass_height / pass_type (train) ===")
for col in ['pass_technique', 'pass_height', 'pass_type']:
    print(f"--- {col} ---")
    print(train[col].value_counts(dropna=False))

print("\n=== Sanity x/y (train) ===")
print(train[['x', 'y']].describe())
print("Null x:", train['x'].isna().sum(), " | Null y:", train['y'].isna().sum())
print("Hors bornes [0,120]x[0,80] :",
      ((train['x'] < 0) | (train['x'] > 120) | (train['y'] < 0) | (train['y'] > 80)).sum())

print("\n=== Équipes des 3 matchs MVP déjà dans le corpus ===")
target_ids = {266310: "Deportivo-Barcelone", 3754214: "Aston Villa-Liverpool", 3901184: "Marseille-Rennes"}
for df, label in [(train, "train"), (test, "test")]:
    for mid, name in target_ids.items():
        sub = df[df['match_id'] == mid]
        if len(sub):
            print(f"[{label}] {name} (match_id={mid}) -> équipes: {sub['team'].unique().tolist()}, {len(sub)} tirs")

print("\n=== Contenu de src/viz/pitch.py (ajuster le chemin si différent) ===")
pitch_path = 'src/viz/pitch.py'
if os.path.exists(pitch_path):
    with open(pitch_path, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print(f"Fichier '{pitch_path}' non trouvé (le module de visualisation n'est pas encore créé).")
