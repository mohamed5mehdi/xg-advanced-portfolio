"""
Régénère les 4 PNG opta_style_<match_id>.png dans reports/
en utilisant generate_opta_style_match_report() (avec le fix "Saved to Post").
Affiche les valeurs "Cadrés" par équipe pour chaque match.
"""
import sys, os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent / "src" / "viz"))
from heatmap import generate_opta_style_match_report

SHOTS_PATH = Path("data/processed/all_shots_scored.parquet")
if not SHOTS_PATH.exists():
    # Fallback : concatène train + test
    train = pd.read_parquet("data/processed/train_features.parquet")
    test  = pd.read_parquet("data/processed/test_features.parquet")
    shots = pd.concat([train, test], ignore_index=True)
    print("[INFO] all_shots_scored.parquet absent — fallback sur train+test")
else:
    shots = pd.read_parquet(SHOTS_PATH)
    print(f"[INFO] {len(shots)} tirs chargés depuis all_shots_scored.parquet")

MATCHES = {
    266310:  "Barcelona 8-0 Deportivo (La Liga 2015/16)",
    3754214: "Aston Villa 0-6 Liverpool (PL 2015/16)",
    3901184: "Marseille 2-5 Rennes (L1 2015/16)",
    18243:   "Real Madrid vs Atl. Madrid — Finale C1 2015/16",
}

os.makedirs("reports", exist_ok=True)

for mid, label in MATCHES.items():
    save_path = f"reports/opta_style_{mid}.png"
    print(f"\n{'='*60}")
    print(f"Match {mid} — {label}")

    sub = shots[shots["match_id"] == mid]
    if sub.empty:
        print(f"  [SKIP] Aucun tir trouvé pour match_id={mid}")
        continue

    xg_col = "model_xg" if "model_xg" in sub.columns else "statsbomb_xg"
    print(f"  xg_col utilisé : {xg_col}  |  {len(sub)} tirs")

    # Affiche les valeurs shot_outcome présentes pour vérification
    if "shot_outcome" in sub.columns:
        outcomes = sub["shot_outcome"].value_counts().to_dict()
        print(f"  shot_outcome values : {outcomes}")
        # Calcul "Cadrés" avec le nouveau set
        known = {"Goal", "Saved", "Saved to Post"}
        for team, gdf in sub.groupby("team"):
            cadres = int(gdf["shot_outcome"].isin(known).sum())
            print(f"    {team:35s}  Cadrés = {cadres}")
    else:
        print("  [WARN] colonne shot_outcome absente")

    fig, _ = generate_opta_style_match_report(
        shots_df=shots,
        match_id=mid,
        xg_col=xg_col,
        save_path=save_path,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [OK] Sauvegarde -> {save_path}")

print("\nTerminé.")
