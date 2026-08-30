"""
generate_raw_predictions.py

Génération des prédictions brutes non calibrées (raw_pred) sur le jeu de test holdout.
Recharge X_train, y_train, X_test, y_test via load_and_prep_data() de src/viz/train.py,
ajuste un XGBClassifier avec les hyperparamètres exacts issus d'Optuna (reports/final_metrics.txt),
prédit les probabilités sur X_test et ajoute la colonne 'raw_pred' dans reports/test_predictions.csv.

Note méthodologique :
raw_pred est issu d'un modèle XGBoost ÉQUIVALENT (mêmes hyperparamètres optimaux) non calibré,
entraîné sur l'ensemble de X_train. Il ne s'agit pas du composant interne exact de
CalibratedClassifierCV (qui utilise 5 modèles en validation croisée GroupKFold pour ajuster
l'étalonnage isotonique), mais d'un estimateur représentatif de la fonction de décision brute
avant calibration.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Résolution des chemins
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.viz.train import load_and_prep_data


def main():
    print("=" * 60)
    print("Génération des prédictions non calibrées (raw_pred)...")
    print("=" * 60)
    
    # 1. Chargement des données
    X_train, y_train, groups_train, X_test, y_test, sb_xg_test = load_and_prep_data()
    print(f"Données chargées : Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # 2. Hyperparamètres exacts issus de reports/final_metrics.txt
    best_params = {
        'n_estimators': 138,
        'max_depth': 3,
        'learning_rate': 0.06784848062577233,
        'subsample': 0.7882960342255022,
        'colsample_bytree': 0.9950963787861274,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Entraînement du modèle XGBoost non calibré équivalent
    print(f"Ajustement de XGBClassifier avec best_params : {best_params}")
    raw_xgb = XGBClassifier(**best_params)
    raw_xgb.fit(X_train, y_train)
    
    # Prédictions sur le test set
    raw_preds = raw_xgb.predict_proba(X_test)[:, 1]
    
    # 3. Chargement et mise à jour de reports/test_predictions.csv
    csv_path = PROJECT_ROOT / "reports" / "test_predictions.csv"
    if not csv_path.exists():
        csv_path = Path("reports/test_predictions.csv")
    
    df_preds = pd.read_csv(csv_path)
    len_existing = len(df_preds)
    len_raw = len(raw_preds)
    
    print(f"\nVérification de la longueur :")
    print(f"  - Longueur test_predictions.csv existant : {len_existing}")
    print(f"  - Longueur raw_pred : {len_raw}")
    assert len_existing == len_raw, f"Erreur : dimensions incompatibles ({len_existing} vs {len_raw})"
    print("  -> Validation de dimension : OK")
    
    # Ajout sans écraser les autres colonnes
    df_preds['raw_pred'] = raw_preds
    df_preds.to_csv(csv_path, index=False)
    print(f"Colonnes enregistrées dans {csv_path} : {list(df_preds.columns)}")
    
    # 4. Calcul et comparaison des métriques
    auc_raw = roc_auc_score(y_test, raw_preds)
    auc_cal = roc_auc_score(y_test, df_preds['xgb_pred'])
    ll_raw = log_loss(y_test, raw_preds)
    ll_cal = log_loss(y_test, df_preds['xgb_pred'])
    brier_raw = brier_score_loss(y_test, raw_preds)
    brier_cal = brier_score_loss(y_test, df_preds['xgb_pred'])
    
    print("\n" + "=" * 60)
    print("RÉSULTATS COMPARATIFS (TEST SET) :")
    print("=" * 60)
    print(f"ROC-AUC    : Non Calibré (raw_pred) = {auc_raw:.6f} | Calibré (xgb_pred) = {auc_cal:.6f} (Diff: {auc_cal - auc_raw:+.6f})")
    print(f"Log-Loss   : Non Calibré (raw_pred) = {ll_raw:.6f} | Calibré (xgb_pred) = {ll_cal:.6f} (Gain: {ll_raw - ll_cal:+.6f})")
    print(f"Brier Loss : Non Calibré (raw_pred) = {brier_raw:.6f} | Calibré (xgb_pred) = {brier_cal:.6f} (Gain: {brier_raw - brier_cal:+.6f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
