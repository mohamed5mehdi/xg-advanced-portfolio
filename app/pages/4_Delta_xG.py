"""
app/pages/4_Delta_xG.py

Page 4 : Diagnostic de sur/sous-performance à la finition (Delta xG).
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
from src.analytics.advanced_metrics import compute_delta_xg, plot_delta_xg_diagonal

st.set_page_config(page_title="Delta xG | xG Portfolio", page_icon="⚡", layout="wide")

st.title("⚡ Diagnostic de Finition : Delta xG (Buts − xG Cumulé)")
st.markdown(
    "Mesure de l'efficacité devant le but : identification des attaquants qui surperforment "
    "ou sous-performent significativement leurs occasions attendues."
)

try:
    shots_df = load_shots_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

filtered_df = render_sidebar_filters(shots_df)

st.sidebar.subheader("⚙️ Paramètres d'Analyse")
min_shots = st.sidebar.slider("Seuil minimum de tirs :", min_value=5, max_value=80, value=15, step=5)
label_top_n = st.sidebar.slider("Nombre de joueurs annotés :", min_value=4, max_value=16, value=6, step=2)

# Calcul du Delta xG
try:
    delta_df = compute_delta_xg(
        filtered_df,
        xg_col="model_xg",
        goal_col="is_goal",
        player_col="player",
        min_shots=min_shots
    )
except Exception as e:
    st.error(f"Erreur de calcul : {e}")
    st.stop()

if delta_df.empty:
    st.warning("Aucun joueur n'atteint le seuil de tirs sélectionné.")
else:
    # Highlights
    top_finisher = delta_df.iloc[0]
    under_finisher = delta_df.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Joueurs analysés", len(delta_df))
    c2.metric(
        f"Top Surperformance : {top_finisher['player']}",
        f"+{top_finisher['delta_xg']:.2f} Buts",
        f"{int(top_finisher['total_goals'])} buts pour {top_finisher['total_xg']:.1f} xG ({int(top_finisher['total_shots'])} tirs)"
    )
    c3.metric(
        f"Forte Sous-performance : {under_finisher['player']}",
        f"{under_finisher['delta_xg']:.2f} Buts",
        f"{int(under_finisher['total_goals'])} buts pour {under_finisher['total_xg']:.1f} xG ({int(under_finisher['total_shots'])} tirs)",
        delta_color="inverse"
    )

    st.markdown("---")

    # Graphique diagonal
    st.subheader("📈 Diagramme de Dispersion y = x (xG vs Buts Réels)")
    st.write(
        "🟢 **Au-dessus de la diagonale (vert)** : Buteurs cliniques ayant marqué plus que prévu par l'xG.<br>"
        "🔴 **En-dessous de la diagonale (rouge)** : Attaquants en déficit d'efficacité ou malchanceux.",
        unsafe_allow_html=True
    )

    with st.spinner("Génération du graphique Delta xG..."):
        fig, _ = plot_delta_xg_diagonal(
            delta_df,
            player_col="player",
            label_top_n=label_top_n
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Table de diagnostic
    st.subheader("📋 Tableau Récapitulatif Delta xG")
    search_player = st.text_input("Rechercher un joueur (ex: Cristiano Ronaldo dos Santos Aveiro, Messi, Ibrahimovic) :", "")
    
    display_df = delta_df.copy()
    if search_player:
        display_df = display_df[display_df["player"].str.contains(search_player, case=False, na=False)]

    display_df["total_xg"] = display_df["total_xg"].round(2)
    display_df["delta_xg"] = display_df["delta_xg"].round(2)
    display_df["total_goals"] = display_df["total_goals"].astype(int)

    display_df = display_df.rename(columns={
        "player": "Joueur",
        "total_shots": "Tirs Totaux",
        "total_xg": "xG Cumulé",
        "total_goals": "Buts Marqués",
        "delta_xg": "Delta xG (Buts − xG)"
    })

    st.dataframe(display_df, use_container_width=True)
