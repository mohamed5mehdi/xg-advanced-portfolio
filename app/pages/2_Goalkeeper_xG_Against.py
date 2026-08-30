"""
app/pages/2_Goalkeeper_xG_Against.py

Page 2 : Analyse de la solidité défensive et xG Against (proxy pré-tir équipe).
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.data_loader import load_shots_data, render_sidebar_filters
from src.analytics.advanced_metrics import compute_team_xg_against, plot_xg_against_diff

st.set_page_config(page_title="Goalkeeper xG Against | xG Portfolio", page_icon="🧤", layout="wide")

st.title("🧤 Solidité Défensive & xG Against (Proxy Pré-Tir)")
st.markdown(
    "Évaluez la performance défensive globale par équipe : comparaison entre le volume d'xG concédé "
    "et les buts réellement encaissés."
)

# Avertissement méthodologique strict
st.info(
    "ℹ️ **Note méthodologique & limites documentées** : "
    "Cet indicateur est un **proxy pré-tir calculé au niveau ÉQUIPE**, car aucune colonne d'identité "
    "de gardien ni coordonnée d'arrivée de tir (post-shot) n'existe dans le schéma de base. "
    "Ce n'est **pas un PSxG-GA** au sens StatsBomb/FBref."
)

try:
    shots_df = load_shots_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

filtered_df = render_sidebar_filters(shots_df)

st.sidebar.subheader("⚙️ Paramètres d'analyse")
min_matches = st.sidebar.slider("Nombre minimum de matchs joués :", min_value=1, max_value=38, value=5)

# Calcul des métriques
try:
    team_agg = compute_team_xg_against(
        filtered_df,
        xg_col="model_xg",
        goal_col="is_goal",
        team_col="team",
        match_col="match_id"
    )
except Exception as e:
    st.error(f"Erreur de calcul : {e}")
    st.stop()

filtered_agg = team_agg[team_agg["matches"] >= min_matches].sort_values("xg_against_diff", ascending=False)

if filtered_agg.empty:
    st.warning("Aucune équipe ne correspond aux critères de filtrage (augmentez les compétitions ou baissez le seuil de matchs).")
else:
    # KPIs Top / Flop défensifs
    top_team = filtered_agg.iloc[0]
    bottom_team = filtered_agg.iloc[-1]

    k1, k2, k3 = st.columns(3)
    k1.metric("Équipes analysées", len(filtered_agg))
    k2.metric(
        f"Meilleure surperformance : {top_team['team']}",
        f"+{top_team['xg_against_diff']:.2f} xG sauvés",
        f"{top_team['goals_conceded']} buts encaissés pour {top_team['xg_faced']:.1f} xG"
    )
    k3.metric(
        f"Plus forte vulnérabilité : {bottom_team['team']}",
        f"{bottom_team['xg_against_diff']:.2f} xG",
        f"{bottom_team['goals_conceded']} buts encaissés pour {bottom_team['xg_faced']:.1f} xG",
        delta_color="inverse"
    )

    st.markdown("---")

    # Visualisation Bar Chart
    st.subheader("📊 Différentiel xG Concédé − Buts Encaissés")
    st.write(
        "🟢 **Vert (positif)** : L'équipe a encaissé moins de buts que prévu par le modèle xG (défense/gardien solides ou réussite adverse faible).<br>"
        "🔴 **Rouge (négatif)** : L'équipe a encaissé plus de buts que prévu par le modèle xG.",
        unsafe_allow_html=True
    )

    with st.spinner("Génération du graphique de solidité défensive..."):
        fig, _ = plot_xg_against_diff(filtered_agg, min_matches=min_matches)
        st.pyplot(fig)
        plt.close(fig)

    # Tableau récapitulatif
    with st.expander("📋 Tableau des statistiques défensives détaillées", expanded=True):
        display_df = filtered_agg.copy()
        display_df["xg_faced"] = display_df["xg_faced"].round(2)
        display_df["xg_against_diff"] = display_df["xg_against_diff"].round(2)
        display_df.columns = ["Équipe", "xG Concédé Total", "Buts Encaissés", "Tirs Subis", "Matchs", "Différentiel xG Sauvés"]
        st.dataframe(
            display_df.sort_values("Différentiel xG Sauvés", ascending=False),
            use_container_width=True
        )
