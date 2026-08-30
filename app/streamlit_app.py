"""
app/streamlit_app.py

Page principale d'accueil du portfolio xG Calibrated Analytics.
Présentation du modèle, métriques clés, showcase des matchs whitelistés
et accès aux outils d'analyse avancés.
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="xG Advanced Portfolio | Calibrated Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Résolution des imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.data_loader import load_shots_data, load_whitelist, render_sidebar_filters
from src.viz.heatmap import generate_opta_style_match_report

# CSS personnalisé pour un rendu moderne et soigné
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f2f6;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #a4b0be;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #3498db;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #ced6e0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .methodo-card {
        background-color: #1e272e;
        border: 1px solid #2f3542;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">⚽ Calibrated Expected Goals (xG)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Modèle xG XGBoost avec calibration isotonique sur données StatsBomb Open Data (2015/16). '
        'Conçu pour la transparence méthodologique et l\'honnêteté du périmètre.</div>',
        unsafe_allow_html=True
    )

    # Chargement des données
    try:
        shots_df = load_shots_data()
        whitelist_matches = load_whitelist()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return

    # Filtres latéraux globaux
    filtered_df = render_sidebar_filters(shots_df, default_all=True)

    # KPIs principaux
    st.markdown("### 📊 Métriques Générales du Corpus")
    c1, c2, c3, c4 = st.columns(4)

    total_shots = len(shots_df)
    total_goals = int(shots_df["is_goal"].sum())
    total_xg = shots_df["model_xg"].sum()
    holdout_auc = 0.8024
    statsbomb_benchmark = 0.8041

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3498db;">
            <div class="metric-lbl">Volume Total de Tirs</div>
            <div class="metric-val">{total_shots:,}</div>
            <span style="font-size: 0.8rem; color: #70a1ff;">5 Ligues + Finale C1</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <div class="metric-lbl">Total Buts Marqués</div>
            <div class="metric-val">{total_goals:,}</div>
            <span style="font-size: 0.8rem; color: #7bed9f;">{total_xg:,.1f} xG cumulés</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f1c40f;">
            <div class="metric-lbl">ROC-AUC Test Holdout</div>
            <div class="metric-val">{holdout_auc:.4f}</div>
            <span style="font-size: 0.8rem; color: #eccc68;">Benchmark SB : {statsbomb_benchmark:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-lbl">Écart Benchmark Volontaire</div>
            <div class="metric-val">0.0017</div>
            <span style="font-size: 0.8rem; color: #ff6b81;">Non sur-optimisé</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section Showcase : Rapports de Match Whitelistés
    st.markdown("### 🏆 Showcase — Matchs de Référence Confirmés")
    st.write(
        "Sélectionnez un match parmi la liste des matchs de référence du portfolio "
        "pour générer le rapport tactique de tirs façon Opta."
    )

    if whitelist_matches:
        match_options = {m["match_id"]: f"{m['label']} (ID: {m['match_id']})" for m in whitelist_matches}
        selected_match_id = st.selectbox(
            "Sélectionner un match du showcase :",
            options=list(match_options.keys()),
            format_func=lambda x: match_options[x]
        )

        match_shots = shots_df[shots_df["match_id"] == selected_match_id]
        if not match_shots.empty:
            with st.spinner("Génération du rapport de match..."):
                fig, _ = generate_opta_style_match_report(
                    shots_df,
                    match_id=selected_match_id,
                    xg_col="model_xg"
                )
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning(f"Aucun tir trouvé pour le match {selected_match_id}.")
    else:
        st.info("Aucun match whitelisté configuré.")

    st.markdown("---")

    # Piliers Méthodologiques & Décisions d'Ingénierie
    st.markdown("### 🔬 Rigueur Méthodologique & Décisions Clés")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="methodo-card">
            <h4 style="color: #70a1ff; margin-top:0;">1. Calibration Isotonique</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea;">
            Un modèle xG n'est pas un classifieur binaire standard : la justesse de la probabilité prédite 
            est primordiale. Utilisation de <code>CalibratedClassifierCV(GroupKFold(5))</code> avec 
            <code>max_depth</code> contrôlé (2–4) pour éviter le surapprentissage.
            </p>
        </div>
        <div class="methodo-card">
            <h4 style="color: #7bed9f; margin-top:0;">2. Gestion Honnête des Valeurs Manquantes</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea;">
            En l'absence de freeze-frame du gardien, les variables <code>gk_distance</code> et <code>gk_angle</code> 
            sont conservées en <strong>NaN honnête</strong> plutôt que substituées par les coordonnées du centre du but. 
            Gain mesuré : <strong>+0.00167 AUC</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="methodo-card">
            <h4 style="color: #eccc68; margin-top:0;">3. Correction de Collision de Schéma</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea;">
            Identification d'un bug subtil où <code>pass_technique</code>, <code>pass_height</code> et <code>pass_type</code> 
            étaient écrasées en 'None' lors d'un merge pandas. Résolution nette : <strong>+0.00136 AUC</strong>.
            </p>
        </div>
        <div class="methodo-card">
            <h4 style="color: #ff6b81; margin-top:0;">4. Transparence sur les Limites Métier</h4>
            <p style="font-size: 0.9rem; color: #dfe4ea;">
            - <strong>Proxy pré-tir</strong> : pas de PSxG déguisé sans coordonnées d'arrivée.<br>
            - <strong>Pas de stats /90 artificielles</strong> sans total réel de minutes jouées.<br>
            - <strong>Bundesliga exclue de l'entraînement</strong> (échantillon partiel 34 matchs).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "💡 *Naviguez dans le menu latéral pour explorer les 4 pages spécialisées : "
        "**Shot Explorer**, **Goalkeeper xG Against**, **Scouting Quadrant** et **Delta xG**.*"
    )


if __name__ == "__main__":
    main()
