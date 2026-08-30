"""
app/pages/1_Shot_Explorer.py

Page 1 : Explorateur interactif de tirs et cartographies tactiques (KDE & Buts).
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
from src.viz.heatmap import plot_shot_map, generate_opta_style_match_report

st.set_page_config(page_title="Shot Explorer | xG Portfolio", page_icon="🎯", layout="wide")

st.title("🎯 Shot Explorer & Cartographies Tactiques")
st.markdown(
    "Explorez les tirs individuels, les cartes de chaleur xG pondérées (KDE) "
    "ou les rapports de match binationaux façon Opta."
)

try:
    shots_df = load_shots_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

# Filtres de compétition
filtered_df = render_sidebar_filters(shots_df)

# Filtres supplémentaires
st.sidebar.subheader("🔍 Filtres Spécifiques")

tab_mode = st.radio(
    "Mode d'analyse :",
    ["Cartographie par Joueur / Équipe", "Rapport de Match (Opta Style)"],
    horizontal=True
)

if tab_mode == "Cartographie par Joueur / Équipe":
    c1, c2, c3 = st.columns(3)
    
    with c1:
        teams = ["Toutes les équipes"] + sorted(filtered_df["team"].dropna().unique().tolist())
        selected_team = st.selectbox("Équipe :", teams)
    
    subset_df = filtered_df if selected_team == "Toutes les équipes" else filtered_df[filtered_df["team"] == selected_team]
    
    with c2:
        players = ["Tous les joueurs"] + sorted(subset_df["player"].dropna().unique().tolist())
        selected_player = st.selectbox("Joueur :", players)
        
    if selected_player != "Tous les joueurs":
        subset_df = subset_df[subset_df["player"] == selected_player]
        
    with c3:
        map_mode = st.selectbox("Type d'affichage :", ["all (Heatmap KDE + Tirs)", "goals (Buts annotés uniquement)"])
        actual_mode = "all" if "all" in map_mode else "goals"

    st.markdown("---")
    
    # Métriques du sous-ensemble
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tirs filtrés", len(subset_df))
    m2.metric("Buts marqués", int(subset_df["is_goal"].sum()) if not subset_df.empty else 0)
    m3.metric("xG Total Modèle", f"{subset_df['model_xg'].sum():.2f}" if not subset_df.empty else "0.00")
    m4.metric("xG Moyen / Tir", f"{subset_df['model_xg'].mean():.3f}" if not subset_df.empty else "0.000")

    if subset_df.empty:
        st.warning("Aucun tir ne correspond aux critères sélectionnés.")
    else:
        title_str = f"Cartographie : {selected_player if selected_player != 'Tous les joueurs' else selected_team}"
        with st.spinner("Rendu de la cartographie..."):
            fig, _ = plot_shot_map(
                subset_df,
                mode=actual_mode,
                xg_col="model_xg",
                title=title_str,
                annotate_summary=True
            )
            st.pyplot(fig)
            plt.close(fig)

        # Tableau des tirs
        with st.expander("📋 Consulter les données détaillées des tirs", expanded=False):
            cols_to_show = [
                "match_id", "team", "player", "minute", "is_goal", "model_xg",
                "statsbomb_xg", "distance", "angle", "defenders_in_cone",
                "gk_found", "under_pressure", "shot_technique", "shot_outcome"
            ]
            available_cols = [c for c in cols_to_show if c in subset_df.columns]
            st.dataframe(
                subset_df[available_cols].sort_values("model_xg", ascending=False),
                use_container_width=True
            )

else:
    # Mode Rapport Opta
    available_matches = sorted(filtered_df["match_id"].unique())
    selected_mid = st.selectbox(
        "Sélectionnez un match_id à analyser :",
        available_matches,
        format_func=lambda x: f"Match ID {x} ({', '.join(filtered_df[filtered_df['match_id'] == x]['team'].unique())})"
    )
    
    match_shots = filtered_df[filtered_df["match_id"] == selected_mid]
    
    if len(match_shots["team"].unique()) != 2:
        st.warning("Ce match ne contient pas exactement 2 équipes dans le corpus.")
    else:
        with st.spinner("Génération du rapport de match complet..."):
            fig, _ = generate_opta_style_match_report(
                match_shots,
                match_id=selected_mid,
                xg_col="model_xg"
            )
            st.pyplot(fig)
            plt.close(fig)
