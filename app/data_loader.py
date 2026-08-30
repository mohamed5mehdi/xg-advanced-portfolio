"""
app/data_loader.py

Module centralisé de chargement des données et de gestion des filtres
pour le dashboard Streamlit xg-advanced-portfolio.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Assurer l'accès aux modules src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

COMPETITION_MAP = {
    11: "La Liga (2015/16)",
    2: "Premier League (2015/16)",
    12: "Serie A (2015/16)",
    7: "Ligue 1 (2015/16 - Test Holdout)",
    9: "Bundesliga (2015/16 - Hors périmètre)",
    16: "Champions League (2015/16 - Showcase Finale)",
}

OUT_OF_SCOPE_COMPETITION_IDS = {9, 16}


@st.cache_data
def load_shots_data() -> pd.DataFrame:
    """
    Charge le dataset all_shots_scored.parquet enrichi des prédictions du modèle xG.
    Ajoute les libellés de compétition et le flag out_of_training_scope.
    """
    data_path = PROJECT_ROOT / "data" / "processed" / "all_shots_scored.parquet"
    if not data_path.exists():
        # Fallback chemin relatif direct
        data_path = Path("data/processed/all_shots_scored.parquet")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    df = pd.read_parquet(data_path)
    
    # Mapping des noms de compétition
    df["competition_name"] = df["competition_id"].map(COMPETITION_MAP).fillna("Autre Compétition")
    
    # Flag explicite de périmètre d'entraînement (Bundesliga=9, Showcase C1=16)
    df["out_of_training_scope"] = df["competition_id"].isin(OUT_OF_SCOPE_COMPETITION_IDS)
    
    return df


@st.cache_data
def load_whitelist() -> list[dict]:
    """Charge la liste des matchs vérifiés et confirmés du portfolio."""
    whitelist_path = PROJECT_ROOT / "app" / "match_whitelist.json"
    if not whitelist_path.exists():
        whitelist_path = Path("app/match_whitelist.json")
    
    if not whitelist_path.exists():
        return []
    
    with open(whitelist_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("matches", [])


def check_and_render_scope_warnings(selected_comp_ids: list[int] | set[int]) -> None:
    """
    Affiche un avertissement visible dans la barre latérale si une compétition
    hors périmètre d'entraînement (ex: Bundesliga) est sélectionnée.
    """
    if 9 in selected_comp_ids:
        st.sidebar.warning(
            "⚠️ **Bundesliga (2015/16)** : Échantillon partiel dans StatsBomb Open Data "
            "(34 matchs, un seul club). Exclu du périmètre d'entraînement du modèle. "
            "Les valeurs xG sont affichées à titre indicatif et exploratoire."
        )
    if 16 in selected_comp_ids:
        st.sidebar.info(
            "ℹ️ **Champions League** : Match showcase hors distribution (Finale 2015/16 Real vs Atlético). "
            "Tirs scorés via le même pipeline (séance de tirs au but exclue)."
        )


def render_sidebar_filters(df: pd.DataFrame, default_all: bool = True) -> pd.DataFrame:
    """
    Rend les filtres communs dans st.sidebar et retourne le dataframe filtré
    après avoir vérifié et affiché les avertissements de périmètre.
    """
    st.sidebar.header("🎯 Filtres Globaux")
    
    available_comps = df[["competition_id", "competition_name"]].drop_duplicates().sort_values("competition_id")
    comp_options = dict(zip(available_comps["competition_id"], available_comps["competition_name"]))
    
    # Sélection des compétitions
    default_selection = list(comp_options.keys()) if default_all else [11, 2, 12, 7]
    selected_comp_ids = st.sidebar.multiselect(
        "Compétitions",
        options=list(comp_options.keys()),
        format_func=lambda x: comp_options.get(x, str(x)),
        default=default_selection,
        help="Filtrer les tirs par compétition et saison 2015/16."
    )
    
    # Exécution de l'avertissement visible st.sidebar.warning()
    check_and_render_scope_warnings(selected_comp_ids)
    
    filtered_df = df[df["competition_id"].isin(selected_comp_ids)].copy()
    
    return filtered_df
