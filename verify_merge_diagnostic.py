import pandas as pd

df = pd.read_parquet('data/raw/train/comp11_season27.parquet')

print("=== Colonnes brutes contenant 'pass' ===")
print([c for c in df.columns if 'pass' in c.lower()])
print()

shots = df[df['type'] == 'Shot'].copy()
passes = df[df['type'] == 'Pass'][['id', 'pass_technique', 'pass_height', 'pass_type']].copy()
passes = passes.rename(columns={'id': 'shot_key_pass_id'})

print("=== Colonnes de 'shots' AVANT merge (contenant 'pass') ===")
print([c for c in shots.columns if 'pass' in c.lower()])
print()
print("=== Colonnes de 'passes' AVANT merge (contenant 'pass') ===")
print([c for c in passes.columns if 'pass' in c.lower()])
print()

merged = shots.merge(passes, on='shot_key_pass_id', how='left')

print("=== Colonnes de 'merged' APRES merge (contenant 'pass') ===")
pass_cols_merged = [c for c in merged.columns if 'pass' in c.lower()]
print(pass_cols_merged)
print()

for col in pass_cols_merged:
    print(f"--- value_counts({col}) ---")
    print(merged[col].value_counts(dropna=False).head(10))
    print()