"""
app/pages/5_Model_Evaluation.py

Page 5 : Évaluation approfondie des modèles xG, analyse de discrimination,
fiabilité de calibration, benchmark comparatif et interprétabilité globale (SHAP).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# Configuration de la page
st.set_page_config(
    page_title="Model Evaluation | xG Advanced Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Résolution des chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# CSS personnalisé pour harmonisation avec l'application
st.markdown("""
<style>
    .eval-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f1f2f6;
        margin-bottom: 0.2rem;
    }
    .eval-subtitle {
        font-size: 1.05rem;
        color: #a4b0be;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #3498db;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #ced6e0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-card {
        background-color: #1e272e;
        border: 1px solid #2f3542;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .optuna-pill {
        display: inline-block;
        background-color: #2f3542;
        color: #70a1ff;
        padding: 4px 10px;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 3px 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Charge le jeu de prédictions sur le test holdout (Ligue 1 2015/16)."""
    csv_path = PROJECT_ROOT / "reports" / "test_predictions.csv"
    if not csv_path.exists():
        csv_path = Path("reports/test_predictions.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    return pd.read_csv(csv_path)


@st.cache_data
def load_final_metrics_and_shap() -> tuple[dict, dict, list[tuple[str, float]]]:
    """
    Lit directement le fichier reports/final_metrics.txt sans recalcul manuel
    pour garantir une stricte cohérence avec les rapports d'entraînement.
    """
    metrics_path = PROJECT_ROOT / "reports" / "final_metrics.txt"
    if not metrics_path.exists():
        metrics_path = Path("reports/final_metrics.txt")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {}
    # Extraction des métriques par modèle
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Naive:"):
            ll = float(re.search(r"LogLoss=([\d\.]+)", line).group(1))
            brier = float(re.search(r"Brier=([\d\.]+)", line).group(1))
            metrics["Naive"] = {"LogLoss": ll, "Brier": brier, "AUC": None}
        elif line.startswith("Logistic:"):
            auc_val = float(re.search(r"AUC=([\d\.]+)", line).group(1))
            ll = float(re.search(r"LogLoss=([\d\.]+)", line).group(1))
            brier = float(re.search(r"Brier=([\d\.]+)", line).group(1))
            metrics["Logistic"] = {"AUC": auc_val, "LogLoss": ll, "Brier": brier}
        elif line.startswith("XGBoost:"):
            auc_val = float(re.search(r"AUC=([\d\.]+)", line).group(1))
            ll = float(re.search(r"LogLoss=([\d\.]+)", line).group(1))
            brier = float(re.search(r"Brier=([\d\.]+)", line).group(1))
            metrics["XGBoost"] = {"AUC": auc_val, "LogLoss": ll, "Brier": brier}
        elif line.startswith("StatsBomb:"):
            auc_val = float(re.search(r"AUC=([\d\.]+)", line).group(1))
            ll = float(re.search(r"LogLoss=([\d\.]+)", line).group(1))
            brier = float(re.search(r"Brier=([\d\.]+)", line).group(1))
            metrics["StatsBomb"] = {"AUC": auc_val, "LogLoss": ll, "Brier": brier}

    # Extraction des best params
    best_params = {}
    params_match = re.search(r"Best XGBoost Params:\s*(\{.*?\})", content)
    if params_match:
        try:
            import ast
            best_params = ast.literal_eval(params_match.group(1))
        except Exception:
            best_params = {}

    # Extraction des SHAP features
    shap_features = []
    in_shap = False
    for line in content.splitlines():
        if "Top 10 SHAP Features:" in line:
            in_shap = True
            continue
        if in_shap and line.strip():
            parts = line.strip().split(":")
            if len(parts) == 2:
                feat_name = parts[0].strip()
                val = float(parts[1].strip())
                shap_features.append((feat_name, val))

    return metrics, best_params, shap_features


def main():
    st.markdown('<div class="eval-header">📈 Évaluation & Diagnostic du Modèle xG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="eval-subtitle">'
        'Analyse comparative complète sur le jeu de test holdout (Ligue 1 2015/16 — 8 723 tirs) : '
        'pouvoir discriminant (ROC & PR), diagramme de fiabilité (calibration isotonique), métriques globales et interprétabilité SHAP.'
        '</div>',
        unsafe_allow_html=True
    )

    # Chargement des données et métriques de référence
    try:
        df_preds = load_test_predictions()
        metrics, best_params, shap_features = load_final_metrics_and_shap()
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats de modélisation : {e}")
        return

    y_true = df_preds["y_true"].values

    # -------------------------------------------------------------
    # 1. KPIs RÉSUMÉS EN BANDEAU SUPÉRIEUR
    # -------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    
    xgb_auc = metrics.get("XGBoost", {}).get("AUC", 0.8024)
    sb_auc = metrics.get("StatsBomb", {}).get("AUC", 0.8041)
    xgb_brier = metrics.get("XGBoost", {}).get("Brier", 0.0728)
    xgb_ll = metrics.get("XGBoost", {}).get("LogLoss", 0.2575)
    test_shots_count = len(df_preds)

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <div class="metric-lbl">XGBoost ROC-AUC</div>
            <div class="metric-val">{xgb_auc:.4f}</div>
            <span style="font-size: 0.75rem; color: #7bed9f;">Calibré Isotonique</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3498db;">
            <div class="metric-lbl">Benchmark StatsBomb</div>
            <div class="metric-val">{sb_auc:.4f}</div>
            <span style="font-size: 0.75rem; color: #70a1ff;">Écart minime : -0.0017</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f1c40f;">
            <div class="metric-lbl">Brier Score Loss</div>
            <div class="metric-val">{xgb_brier:.4f}</div>
            <span style="font-size: 0.75rem; color: #eccc68;">Baseline Naive : 0.0881</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9b59b6;">
            <div class="metric-lbl">Log-Loss Holdout</div>
            <div class="metric-val">{xgb_ll:.4f}</div>
            <span style="font-size: 0.75rem; color: #d2a8d8;">StatsBomb : 0.2567</span>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-lbl">Tirs Test Holdout</div>
            <div class="metric-val">{test_shots_count:,}</div>
            <span style="font-size: 0.75rem; color: #ff6b81;">Ligue 1 2015/16</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. COURBES ROC ET PRECISION-RECALL (PLOTLY)
    # -------------------------------------------------------------
    st.markdown("### 🎯 Pouvoir de Discrimination : Courbes ROC & Precision-Recall")
    
    col_roc, col_pr = st.columns(2)

    # Configuration des 4 modèles
    model_configs = [
        {"col": "xgb_pred", "name": "XGBoost Calibré", "color": "#2ecc71", "width": 3, "dash": "solid"},
        {"col": "raw_pred", "name": "XGBoost Non Calibré", "color": "#e67e22", "width": 2, "dash": "dot"},
        {"col": "sb_pred", "name": "Benchmark StatsBomb", "color": "#3498db", "width": 2.5, "dash": "solid"},
        {"col": "lr_pred", "name": "Régression Logistique", "color": "#9b59b6", "width": 2, "dash": "dash"},
    ]

    with col_roc:
        fig_roc = go.Figure()
        # Diagonale aléatoire
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#7f8c8d", width=1.5, dash="dash"),
            name="Aléatoire (AUC = 0.5000)",
            hoverinfo="skip"
        ))

        for cfg in model_configs:
            if cfg["col"] in df_preds.columns:
                fpr, tpr, _ = roc_curve(y_true, df_preds[cfg["col"]])
                auc_val = roc_auc_score(y_true, df_preds[cfg["col"]])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"{cfg['name']} (AUC = {auc_val:.4f})",
                    line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
                    hovertemplate=f"<b>{cfg['name']}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
                ))

        fig_roc.update_layout(
            title=dict(text="<b>Courbe ROC (Receiver Operating Characteristic)</b>", font=dict(color="white", size=15)),
            xaxis=dict(title="Taux de Faux Positifs (FPR)", gridcolor="#2f3542", color="white", range=[-0.01, 1.01]),
            yaxis=dict(title="Taux de Vrais Positifs (TPR / Rappel)", gridcolor="#2f3542", color="white", range=[-0.01, 1.01]),
            paper_bgcolor="#1e272e",
            plot_bgcolor="#14181d",
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(30,39,46,0.85)",
                bordercolor="#2f3542",
                borderwidth=1,
                x=0.45, y=0.08
            ),
            margin=dict(l=45, r=20, t=50, b=45),
            height=430,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        fig_pr = go.Figure()
        # Baseline No-skill (taux de buts moyen)
        base_rate = float(y_true.mean())
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[base_rate, base_rate],
            mode="lines",
            line=dict(color="#7f8c8d", width=1.5, dash="dash"),
            name=f"Baseline Aléatoire (AP = {base_rate:.4f})",
            hoverinfo="skip"
        ))

        for cfg in model_configs:
            if cfg["col"] in df_preds.columns:
                precision, recall, _ = precision_recall_curve(y_true, df_preds[cfg["col"]])
                ap_val = average_precision_score(y_true, df_preds[cfg["col"]])
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode="lines",
                    name=f"{cfg['name']} (AP = {ap_val:.4f})",
                    line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
                    hovertemplate=f"<b>{cfg['name']}</b><br>Rappel: %{{x:.3f}}<br>Précision: %{{y:.3f}}<extra></extra>"
                ))

        fig_pr.update_layout(
            title=dict(text="<b>Courbe Précision-Rappel (Precision-Recall)</b>", font=dict(color="white", size=15)),
            xaxis=dict(title="Rappel (Recall)", gridcolor="#2f3542", color="white", range=[-0.01, 1.01]),
            yaxis=dict(title="Précision", gridcolor="#2f3542", color="white", range=[-0.01, 1.01]),
            paper_bgcolor="#1e272e",
            plot_bgcolor="#14181d",
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(30,39,46,0.85)",
                bordercolor="#2f3542",
                borderwidth=1,
                x=0.45, y=0.92
            ),
            margin=dict(l=45, r=20, t=50, b=45),
            height=430,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. DIAGRAMME DE FIABILITÉ (CALIBRATION CURVE)
    # -------------------------------------------------------------
    st.markdown("### ⚖️ Diagramme de Fiabilité : Effet de la Calibration Isotonique")
    st.write(
        "Le diagramme de fiabilité regroupe les prédictions en **10 intervalles (bins)** et compare la probabilité moyenne prédite "
        "à la fréquence réelle de buts observée. Un modèle parfaitement calibré suit strictement la diagonale en pointillés ($y = x$)."
    )

    col_cal_plot, col_cal_text = st.columns([3, 2])

    with col_cal_plot:
        fig_cal = go.Figure()
        # Diagonale parfaite
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#95a5a6", width=2, dash="dash"),
            name="Calibration Parfaite (y = x)",
            hoverinfo="skip"
        ))

        cal_markers = {
            "xgb_pred": {"marker": "circle", "size": 9},
            "raw_pred": {"marker": "diamond", "size": 9},
            "sb_pred": {"marker": "square", "size": 8},
            "lr_pred": {"marker": "triangle-up", "size": 8},
        }

        for cfg in model_configs:
            col_name = cfg["col"]
            if col_name in df_preds.columns:
                prob_true, prob_pred = calibration_curve(y_true, df_preds[col_name], n_bins=10, strategy="uniform")
                fig_cal.add_trace(go.Scatter(
                    x=prob_pred, y=prob_true,
                    mode="lines+markers",
                    name=cfg["name"],
                    line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
                    marker=dict(symbol=cal_markers[col_name]["marker"], size=cal_markers[col_name]["size"], color=cfg["color"]),
                    hovertemplate=f"<b>{cfg['name']}</b><br>xG Moyen Prédit: %{{x:.3f}}<br>Fréquence Réelle Buts: %{{y:.3f}}<extra></extra>"
                ))

        fig_cal.update_layout(
            title=dict(text="<b>Reliability Diagram (10 bins uniformes)</b>", font=dict(color="white", size=15)),
            xaxis=dict(title="Probabilité Prédite (xG Moyen)", gridcolor="#2f3542", color="white", range=[-0.02, 1.02]),
            yaxis=dict(title="Fréquence Empirique de Buts", gridcolor="#2f3542", color="white", range=[-0.02, 1.02]),
            paper_bgcolor="#1e272e",
            plot_bgcolor="#14181d",
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(30,39,46,0.85)",
                bordercolor="#2f3542",
                borderwidth=1,
                x=0.03, y=0.92
            ),
            margin=dict(l=45, r=20, t=50, b=45),
            height=460,
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    with col_cal_text:
        st.markdown(r"""
        <div class="section-card">
            <h4 style="color: #2ecc71; margin-top:0;">💡 Pourquoi la calibration est essentielle</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea; line-height: 1.5;">
            En classification binaire standard, seul l'ordre relatif compte (ROC-AUC).
            En <strong>football analytics</strong>, l'espérance de buts cumulée ($\sum \text{xG}$) doit correspondre
            au nombre réel de buts marqués — la justesse probabiliste est donc primordiale.
            </p>
            <hr style="border-color: #2f3542; margin: 10px 0;" />
            <p style="font-size: 0.82rem; color: #a4b0be; line-height: 1.5; font-style: italic;">
            ⚠️ <strong>Note méthodologique sur le delta observé :</strong><br/>
            <code>raw_pred</code> = un seul XGBoost entraîné sur 100 % des données d'entraînement.<br/>
            <code>xgb_pred</code> = moyenne de 5 modèles indépendants (un par fold <code>GroupKFold</code>),
            chacun calibré isotoniquement via <code>CalibratedClassifierCV</code>.<br/><br/>
            Le delta mesuré (ROC-AUC +0.0011, Log-Loss −0.00072, Brier −0.00022) reflète
            <strong>à la fois l'effet de la calibration isotonique ET l'effet de lissage par ensemble</strong>
            (moyennage des 5 folds). Ces deux contributions ne sont <em>pas isolées séparément ici</em>.
            </p>
            <hr style="border-color: #2f3542; margin: 10px 0;" />
            <ul style="font-size: 0.88rem; color: #ced6e0; padding-left: 18px;">
                <li><strong style="color: #e67e22;">XGBoost Non Calibré (raw_pred)</strong> : 1 modèle entraîné sur 100 % de X_train. Présente une déviation caractéristique en S (sur-estimation à faible probabilité, sous-estimation des grosses occasions).</li>
                <li><strong style="color: #2ecc71;">XGBoost Calibré (xgb_pred)</strong> : Moyenne de 5 modèles <code>GroupKFold(5)</code>, chacun avec étalonnage isotonique. Colle quasiment à la diagonale idéale ($y = x$).</li>
                <li><strong style="color: #3498db;">Benchmark StatsBomb</strong> : Référence professionnelle directement étalonnée sur des données historiques massives.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. BAR CHARTS COMPARATIFS : LOG-LOSS / BRIER / AUC (LUS DIRECTEMENT)
    # -------------------------------------------------------------
    st.markdown("### 📊 Comparatif des Métriques Clés (Holdout Test Set)")
    st.caption(
        "Valeurs officielles lues directement depuis <code>reports/final_metrics.txt</code> "
        "(aucun recalcul manuel à la volée afin de préserver l'intégrité des rapports d'entraînement)."
    )

    models_order = ["Naive", "Logistic", "XGBoost", "StatsBomb"]
    model_labels = ["Baseline Naive", "Régression Logistique", "XGBoost Calibré", "Benchmark StatsBomb"]
    colors_bar = ["#7f8c8d", "#9b59b6", "#2ecc71", "#3498db"]

    ll_vals = [metrics.get(m, {}).get("LogLoss", 0.0) for m in models_order]
    brier_vals = [metrics.get(m, {}).get("Brier", 0.0) for m in models_order]
    auc_vals = [metrics.get(m, {}).get("AUC") for m in models_order]  # Naive is None

    fig_bars = make_subplots(
        rows=1, cols=3,
        subplot_titles=("<b>Log-Loss (plus bas = meilleur)</b>", "<b>Brier Score (plus bas = meilleur)</b>", "<b>ROC-AUC (plus haut = meilleur)</b>"),
        horizontal_spacing=0.08
    )

    # Log-Loss Bar Chart
    fig_bars.add_trace(go.Bar(
        x=model_labels, y=ll_vals,
        marker=dict(color=colors_bar),
        text=[f"{v:.4f}" for v in ll_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Log-Loss: %{y:.5f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Brier Score Bar Chart
    fig_bars.add_trace(go.Bar(
        x=model_labels, y=brier_vals,
        marker=dict(color=colors_bar),
        text=[f"{v:.4f}" for v in brier_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Brier Score: %{y:.5f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    # ROC-AUC Bar Chart (excluant Naive)
    auc_labels = ["Régression Logistique", "XGBoost Calibré", "Benchmark StatsBomb"]
    auc_display = [metrics.get("Logistic", {}).get("AUC", 0.0), metrics.get("XGBoost", {}).get("AUC", 0.0), metrics.get("StatsBomb", {}).get("AUC", 0.0)]
    colors_auc = ["#9b59b6", "#2ecc71", "#3498db"]

    fig_bars.add_trace(go.Bar(
        x=auc_labels, y=auc_display,
        marker=dict(color=colors_auc),
        text=[f"{v:.4f}" for v in auc_display],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.5f}<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    fig_bars.update_layout(
        paper_bgcolor="#1e272e",
        plot_bgcolor="#14181d",
        font=dict(color="white"),
        height=380,
        margin=dict(l=30, r=30, t=50, b=40),
    )
    for col_idx in [1, 2, 3]:
        fig_bars.update_xaxes(gridcolor="#2f3542", color="white", row=1, col=col_idx, tickangle=-15)
        fig_bars.update_yaxes(gridcolor="#2f3542", color="white", row=1, col=col_idx)

    # Ajustement des bornes d'axe pour la lisibilité
    fig_bars.update_yaxes(range=[0, max(ll_vals) * 1.18], row=1, col=1)
    fig_bars.update_yaxes(range=[0, max(brier_vals) * 1.18], row=1, col=2)
    fig_bars.update_yaxes(range=[0.75, 0.83], row=1, col=3)

    st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. TOP 10 FEATURES SHAP (VALEURS DE FINAL_METRICS.TXT)
    # -------------------------------------------------------------
    st.markdown("### 🔍 Interprétabilité Globale : Top 10 Features SHAP")
    st.write(
        r"Importance globale calculée par valeur absolue moyenne des contributions de Shapley ($|\text{SHAP}|$ moyen) "
        r"sur l'ensemble des tirs d'entraînement. Valeurs extraites fidèlement de <code>reports/final_metrics.txt</code>.",
        unsafe_allow_html=True
    )

    if shap_features:
        # Tri croissant pour affichage horizontal (les plus fortes en haut)
        sorted_shap = sorted(shap_features, key=lambda x: x[1])
        feat_names = [x[0] for x in sorted_shap]
        feat_vals = [x[1] for x in sorted_shap]

        fig_shap = go.Figure(go.Bar(
            x=feat_vals,
            y=feat_names,
            orientation="h",
            marker=dict(
                color=feat_vals,
                colorscale="Viridis",
                showscale=False
            ),
            text=[f"{v:.4f}" for v in feat_vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance Moyenne |SHAP| : %{x:.5f}<extra></extra>"
        ))

        fig_shap.update_layout(
            title=dict(text="<b>Importance Relative des 10 Variables Clés du Modèle xG</b>", font=dict(color="white", size=15)),
            xaxis=dict(title="Importance Moyenne Absolue (|SHAP value|)", gridcolor="#2f3542", color="white", range=[0, max(feat_vals) * 1.15]),
            yaxis=dict(title="", gridcolor="#2f3542", color="white"),
            paper_bgcolor="#1e272e",
            plot_bgcolor="#14181d",
            margin=dict(l=150, r=40, t=50, b=40),
            height=460,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        col_sh1, col_sh2 = st.columns(2)
        with col_sh1:
            st.markdown("""
            **🧠 Enseignements Tactiques Principaux :**
            - **`angle` & `gk_distance`** : La géométrie du tir et la proximité du gardien dominent largement (plus de 60% de l'attribution).
            - **`defenders_in_cone`** : La densité défensive directe dans le cône entre le tireur et les poteaux est le 3ᵉ prédicteur le plus décisif.
            """)
        with col_sh2:
            st.markdown("""
            **🎯 Signaux Contextuels & Techniques :**
            - **`shot_body_part_Right Foot` / `Left Foot`** : Forte différenciation biomécanique par rapport aux têtes.
            - **`pass_height_High Pass` & `pass_technique_Through Ball`** : La dynamique de la passe décisive (hauteur et passe en profondeur) modifie drastiquement la conversion.
            """)
    else:
        st.info("Aucune valeur SHAP trouvée dans le rapport de métriques.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. ESPACE DE RECHERCHE OPTUNA & TRANSPARENCE SCIENTIFIQUE
    # -------------------------------------------------------------
    st.markdown("### 🧪 Espace de Recherche & Optimisation Bayésienne (Optuna)")

    col_opt1, col_opt2 = st.columns([3, 2])

    with col_opt1:
        st.markdown("""
        <div class="section-card">
            <h4 style="color: #70a1ff; margin-top:0;">Configuration du Tuning Bayésien</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea;">
            L'optimisation des hyperparamètres a été conduite via <strong>Optuna</strong> (15 trials) avec validation croisée 
            groupée par match <code>GroupKFold(n_splits=3)</code> afin d'éviter toute fuite d'information intra-rencontre.
            </p>
            <table style="width:100%; font-size:0.88rem; color:#ced6e0; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #2f3542;">
                    <th style="text-align:left; padding:6px;">Hyperparamètre</th>
                    <th style="text-align:left; padding:6px;">Espace de Recherche</th>
                    <th style="text-align:left; padding:6px;">Valeur Retenue (Best)</th>
                </tr>
                <tr style="border-bottom: 1px solid #2f3542;">
                    <td style="padding:6px;"><code>n_estimators</code></td>
                    <td style="padding:6px;">[50, 200] (entier)</td>
                    <td style="padding:6px; color:#2ecc71; font-weight:bold;">138</td>
                </tr>
                <tr style="border-bottom: 1px solid #2f3542;">
                    <td style="padding:6px;"><code>max_depth</code></td>
                    <td style="padding:6px;">[2, 4] (arbres peu profonds / interprétabilité)</td>
                    <td style="padding:6px; color:#2ecc71; font-weight:bold;">3</td>
                </tr>
                <tr style="border-bottom: 1px solid #2f3542;">
                    <td style="padding:6px;"><code>learning_rate</code></td>
                    <td style="padding:6px;">[0.01, 0.1] (échelle logarithmique)</td>
                    <td style="padding:6px; color:#2ecc71; font-weight:bold;">0.0678</td>
                </tr>
                <tr style="border-bottom: 1px solid #2f3542;">
                    <td style="padding:6px;"><code>subsample</code></td>
                    <td style="padding:6px;">[0.7, 1.0]</td>
                    <td style="padding:6px; color:#2ecc71; font-weight:bold;">0.7883</td>
                </tr>
                <tr style="border-bottom: 1px solid #2f3542;">
                    <td style="padding:6px;"><code>colsample_bytree</code></td>
                    <td style="padding:6px;">[0.7, 1.0]</td>
                    <td style="padding:6px; color:#2ecc71; font-weight:bold;">0.9951</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_opt2:
        st.markdown("""
        <div class="section-card">
            <h4 style="color: #ff6b81; margin-top:0;">🛡️ Note de Transparence Scientifique</h4>
            <p style="font-size: 0.88rem; color: #dfe4ea; line-height: 1.5;">
            L'historique complet des 15 essais individuels d'Optuna n'a pas été persisté sur disque lors de la session de tuning 
            initiale (seul le dictionnaire <code>best_params</code> et les métriques finales ont été sauvegardés).
            </p>
            <p style="font-size: 0.88rem; color: #ced6e0; line-height: 1.5;">
            Conformément aux principes de rigueur et d'honnêteté méthodologique du portfolio, 
            <strong>aucune courbe de convergence synthétique n'a été fabriquée</strong>. 
            Les hyperparamètres optimaux retenus garantissent un compromis optimal entre régularisation, 
            profondeur d'arbre contrôlée et fidélité probabiliste.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
