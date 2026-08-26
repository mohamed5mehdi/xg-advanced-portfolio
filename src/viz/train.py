import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import optuna
import shap
import os
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)

cat_cols = ['shot_technique', 'play_pattern', 'pass_technique', 'pass_height', 'pass_type', 'shot_body_part']

def load_and_prep_data():
    train = pd.read_parquet('data/processed/train_features.parquet')
    test = pd.read_parquet('data/processed/test_features.parquet')
    
    len_before = len(train)
    # Exclude Bundesliga (competition_id == 9)
    train = train[train['competition_id'] != 9].copy()
    logging.info(f"Train shots BEFORE Bundesliga exclusion: {len_before}")
    logging.info(f"Train shots AFTER Bundesliga exclusion: {len(train)}")
    
    target = 'is_goal'
    cols_to_drop = [
        'match_id', 'competition_id', 'is_goal', 'statsbomb_xg', 'gk_found',
        'team', 'player', 'minute', 'x', 'y', 'shot_outcome',  # métadonnées Phase 3 — jamais des features
    ]
    
    cat_cols = ['shot_technique', 'play_pattern', 'pass_technique', 'pass_height', 'pass_type', 'shot_body_part']
    
    all_data = pd.concat([train, test], axis=0)
    # Convert categorical to string to avoid issues, though already strings
    for c in cat_cols:
        all_data[c] = all_data[c].astype(str)
        
    all_data_encoded = pd.get_dummies(all_data, columns=cat_cols, drop_first=True)
    
    # Ensure boolean columns from get_dummies are float/int for XGBoost/sklearn
    for c in all_data_encoded.columns:
        if all_data_encoded[c].dtype == 'bool':
            all_data_encoded[c] = all_data_encoded[c].astype(int)
            
    train_encoded = all_data_encoded.iloc[:len(train)].copy()
    test_encoded = all_data_encoded.iloc[len(train):].copy()
    
    X_train = train_encoded.drop(columns=cols_to_drop)
    y_train = train_encoded[target]
    groups_train = train_encoded['match_id']
    
    X_test = test_encoded.drop(columns=cols_to_drop)
    y_test = test_encoded[target]
    statsbomb_xg_test = test_encoded['statsbomb_xg']
    
    return X_train, y_train, groups_train, X_test, y_test, statsbomb_xg_test

def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train, y_train, groups_train, X_test, y_test, sb_xg_test = load_and_prep_data()
    
    # 1. Naive Baseline
    mean_rate = y_train.mean()
    naive_preds = np.full_like(y_test, mean_rate, dtype=float)
    naive_ll = log_loss(y_test, naive_preds)
    naive_brier = brier_score_loss(y_test, naive_preds)
    logging.info(f"Naive Baseline -> Log-Loss: {naive_ll:.4f}, Brier: {naive_brier:.4f}")
    
    # 2. Logistic Regression (Baseline TP1)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    gkf_lr = GroupKFold(n_splits=5)
    cv_splits_lr = list(gkf_lr.split(X_train_scaled, y_train, groups=groups_train))
    calibrated_lr = CalibratedClassifierCV(lr, cv=cv_splits_lr, method='isotonic')
    calibrated_lr.fit(X_train_scaled, y_train)
    lr_preds = calibrated_lr.predict_proba(X_test_scaled)[:, 1]
    
    lr_auc = roc_auc_score(y_test, lr_preds)
    lr_ll = log_loss(y_test, lr_preds)
    lr_brier = brier_score_loss(y_test, lr_preds)
    logging.info(f"Logistic Regression -> AUC: {lr_auc:.4f}, Log-Loss: {lr_ll:.4f}, Brier: {lr_brier:.4f}")
    
    # 3. XGBoost Tuning
    logging.info("Tuning XGBoost...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 4), # Very shallow for explainability
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        gkf = GroupKFold(n_splits=3)
        scores = []
        X_arr = X_train.values
        y_arr = y_train.values
        
        for train_idx, val_idx in gkf.split(X_arr, y_arr, groups_train):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va, y_va = X_arr[val_idx], y_arr[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(log_loss(y_va, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    
    # 4. Train & Calibrate XGBoost
    xgb = XGBClassifier(**best_params)
    gkf_xgb = GroupKFold(n_splits=5)
    cv_splits_xgb = list(gkf_xgb.split(X_train, y_train, groups=groups_train))
    calibrated_xgb = CalibratedClassifierCV(xgb, cv=cv_splits_xgb, method='isotonic')
    calibrated_xgb.fit(X_train, y_train)
    
    xgb_preds = calibrated_xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    xgb_ll = log_loss(y_test, xgb_preds)
    xgb_brier = brier_score_loss(y_test, xgb_preds)
    logging.info(f"XGBoost Calibrated -> AUC: {xgb_auc:.4f}, Log-Loss: {xgb_ll:.4f}, Brier: {xgb_brier:.4f}")
    
    # 4b. Persistance du modèle calibré
    model_artifact = {
        "model": calibrated_xgb,
        "feature_columns": list(X_train.columns),
        "categorical_columns": cat_cols,
        "trained_on": {
            "leagues": "La Liga (11) + Premier League (2) + Serie A (12), season_id=27",
            "excluded": "Bundesliga (competition_id=9) — fichier brut sous-collecté (34 matchs, ~13 Mo)",
            "test_league": "Ligue 1 (7), season_id=27",
        },
        "metrics_test": {
            "auc": float(xgb_auc),
            "log_loss": float(xgb_ll),
            "brier": float(xgb_brier),
        },
    }
    joblib.dump(model_artifact, "models/xgb_calibrated_v1.joblib")
    logging.info("Modèle persisté -> models/xgb_calibrated_v1.joblib")
    
    # 5. StatsBomb Benchmark
    sb_auc = roc_auc_score(y_test, sb_xg_test)
    sb_ll = log_loss(y_test, sb_xg_test)
    sb_brier = brier_score_loss(y_test, sb_xg_test)
    logging.info(f"StatsBomb xG -> AUC: {sb_auc:.4f}, Log-Loss: {sb_ll:.4f}, Brier: {sb_brier:.4f}")
    
    # 6. SHAP Analysis (on a single model fit to explainability)
    final_xgb = XGBClassifier(**best_params)
    final_xgb.fit(X_train, y_train)
    explainer = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X_train)
    
    # Get top 10 features by mean absolute SHAP value
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': shap_sum})
    importance_df = importance_df.sort_values('importance', ascending=False).head(10)
    
    logging.info("=== TOP 10 SHAP FEATURES ===")
    for _, row in importance_df.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")
        
    # Save predictions
    results = pd.DataFrame({
        'y_true': y_test,
        'xgb_pred': xgb_preds,
        'lr_pred': lr_preds,
        'sb_pred': sb_xg_test,
        'naive_pred': naive_preds
    })
    results.to_csv('reports/test_predictions.csv', index=False)
    
    with open('reports/final_metrics.txt', 'w') as f:
        f.write(f"Naive: LogLoss={naive_ll}, Brier={naive_brier}\n")
        f.write(f"Logistic: AUC={lr_auc}, LogLoss={lr_ll}, Brier={lr_brier}\n")
        f.write(f"XGBoost: AUC={xgb_auc}, LogLoss={xgb_ll}, Brier={xgb_brier}\n")
        f.write(f"StatsBomb: AUC={sb_auc}, LogLoss={sb_ll}, Brier={sb_brier}\n")
        f.write(f"\nBest XGBoost Params: {best_params}\n")
        f.write(f"\nTop 10 SHAP Features:\n")
        for _, row in importance_df.iterrows():
            f.write(f"  {row['feature']}: {row['importance']}\n")

if __name__ == "__main__":
    main()
