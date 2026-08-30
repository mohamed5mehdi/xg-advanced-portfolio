"""
app/pages/3_Scouting_Quadrant.py

Page 3 : Matrice de scouting — Volume de tirs vs Qualité moyenne (xG / tir).
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
from src.analytics.advanced_metrics import compute_shot_creation_quality, plot_creation_quality_quadrant

st.set_page_config(page_title="Scouting Quadrant | xG Portfolio", page_icon="🔍", layout="wide")

st.title("🔍 Matrice de Scouting : Volume vs Qualité de Tir")
st.markdown(
    "Cartographie des attaquants et créateurs d'occasions : évaluation conjointe de la quantité d'occasions "
    "obtenues et de leur dangerosité moyenne par tir."
)

st.info(
    "ℹ️ **Limite de données documentée** : Le schéma ne contient pas le temps de jeu exact des joueurs. "
    "Le volume d'occasions est donc exprimé en **nombre brut de tirs** (non normalisé par 90 minutes)."
)

try:
    shots_df = load_shots_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

filtered_df = render_sidebar_filters(shots_df)

st.sidebar.subheader("⚙️ Paramètres de Scouting")
min_shots = st.sidebar.slider("Seuil minimum de tirs :", min_value=5, max_value=100, value=20, step=5)
label_top_n = st.sidebar.slider("Nombre de joueurs labellisés :", min_value=4, max_value=20, value=8)

# Calcul des indicateurs
try:
    scout_agg = compute_shot_creation_quality(
        filtered_df,
        xg_col="model_xg",
        player_col="player",
        pressure_col="under_pressure",
        min_shots=min_shots
    )
except Exception as e:
    st.error(f"Erreur de calcul : {e}")
    st.stop()

if scout_agg.empty:
    st.warning("Aucun joueur n'atteint le seuil minimum de tirs sélectionné.")
else:
    st.markdown("### 🎯 Quadrant d'Analyse (Volume X vs Qualité Y)")
    st.write(
        "La couleur des points indique la **part des tirs pris sous pression défensive** "
        "(rouge = forte pression, vert = tirs plus souvent ouverts)."
    )

    with st.spinner("Génération du scatter 4-quadrants..."):
        fig, _ = plot_creation_quality_quadrant(
            scout_agg,
            player_col="player",
            label_top_n=label_top_n
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Table interactive
    st.subheader("📋 Classement des Attaquants & Buteurs")
    search_player = st.text_input("Filtrer par nom de joueur (ex: Ronaldo, Messi, Benzema) :", "")
    
    display_df = scout_agg.copy()
    if search_player:
        display_df = display_df[display_df["player"].str.contains(search_player, case=False, na=False)]

    display_df["avg_xg_per_shot"] = display_df["avg_xg_per_shot"].round(3)
    display_df["total_xg"] = display_df["total_xg"].round(2)
    display_df["pct_under_pressure"] = (display_df["pct_under_pressure"] * 100).round(1).astype(str) + "%"

    display_df = display_df.rename(columns={
        "player": "Joueur",
        "total_shots": "Tirs Totaux",
        "avg_xg_per_shot": "xG Moyen / Tir",
        "total_xg": "xG Cumulé",
        "goals": "Buts Marqués",
        "pct_under_pressure": "% Sous Pression"
    })

    st.dataframe(display_df, use_container_width=True)
