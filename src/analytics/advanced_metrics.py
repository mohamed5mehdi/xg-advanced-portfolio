"""
Indicateurs métier avancés — xg-advanced-portfolio.

Architecture défensive : chaque fonction valide la présence des colonnes
attendues et lève une erreur explicite plutôt que de produire un résultat
silencieusement faux (principe déjà appliqué ailleurs dans le pipeline :
NaN honnête > valeur fabriquée).

Colonnes réelles utilisées (confirmées par le rapport exhaustif, Phases 1 & 3) :
match_id, team, player, minute, x, y, is_goal, under_pressure, model_xg
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _require_columns(df: pd.DataFrame, cols: set, fn_name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[{fn_name}] Colonnes manquantes : {sorted(missing)}. "
            f"Vérifie le schéma réel de all_shots_scored.parquet avant d'appeler "
            f"cette fonction — ne pas deviner un nom de colonne approximant."
        )


# ---------------------------------------------------------------------------
# 1. "xG Against" — PROXY PRÉ-SHOT (voir note méthodologique dans le docstring)
# ---------------------------------------------------------------------------

def compute_team_xg_against(
    shots_df: pd.DataFrame,
    xg_col: str = "model_xg",
    goal_col: str = "is_goal",
    team_col: str = "team",
    match_col: str = "match_id",
) -> pd.DataFrame:
    """
    xG concédé (proxy pré-tir) par équipe, vs buts réellement encaissés.

    ATTENTION MÉTHODOLOGIQUE — À RESPECTER DANS TOUTE PRÉSENTATION :
    Ceci N'EST PAS un PSxG-GA au sens StatsBomb/FBref. Un vrai PSxG nécessite
    un modèle post-shot entraîné sur shot.end_location, donnée absente des
    23 colonnes documentées de features.py. Notre modèle est pré-shot.
    Nommer ce graphique "xG Against (proxy pré-tir)", jamais "PSxG-GA".

    Agrégation au niveau ÉQUIPE, pas gardien : aucune colonne d'identité de
    gardien n'existe dans le schéma actuel.
    """
    _require_columns(shots_df, {xg_col, goal_col, team_col, match_col}, "compute_team_xg_against")

    rows = []
    for match_id, match_shots in shots_df.groupby(match_col):
        teams = match_shots[team_col].unique()
        if len(teams) != 2:
            continue  # données partielles — exclues explicitement, pas devinées
        for team in teams:
            opponent_shots = match_shots[match_shots[team_col] != team]
            rows.append({
                "match_id": match_id,
                "team": team,
                "xg_faced": opponent_shots[xg_col].sum(),
                "goals_conceded": opponent_shots[goal_col].sum(),
                "shots_faced": len(opponent_shots),
            })

    agg = pd.DataFrame(rows).groupby("team", as_index=False).agg(
        xg_faced=("xg_faced", "sum"),
        goals_conceded=("goals_conceded", "sum"),
        shots_faced=("shots_faced", "sum"),
        matches=("match_id", "nunique"),
    )
    agg["xg_against_diff"] = agg["xg_faced"] - agg["goals_conceded"]
    return agg.sort_values("xg_against_diff")


def plot_xg_against_diff(agg_df: pd.DataFrame, min_matches: int = 1, save_path: str | None = None):
    """Bar chart horizontal centré sur 0 — vert (droite) = mieux que prévu, rouge (gauche) = pire."""
    df = agg_df[agg_df["matches"] >= min_matches].sort_values("xg_against_diff")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df["xg_against_diff"]]
    ax.barh(df["team"], df["xg_against_diff"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="white", linewidth=1)
    ax.set_xlabel("xG Against (proxy pré-tir) − Buts encaissés", color="white")
    ax.set_title("Solidité défensive — xG concédé vs buts réels", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.text(0.5, -0.02, "⚠ Proxy pré-tir — pas un PSxG-GA au sens strict.", ha="center", color="#999999", fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Scouting — Qualité de création vs Volume
# ---------------------------------------------------------------------------

def compute_shot_creation_quality(
    shots_df: pd.DataFrame,
    xg_col: str = "model_xg",
    player_col: str = "player",
    pressure_col: str = "under_pressure",
    min_shots: int = 10,
) -> pd.DataFrame:
    """
    LIMITE DE DONNÉES À DOCUMENTER : pas de minutes jouées par joueur dans le
    schéma actuel (`minute` = instant du tir, pas un total de minutes jouées).
    Impossible de normaliser par 90 min sans source supplémentaire. Renvoie
    donc un VOLUME BRUT de tirs, pas un volume /90.
    """
    _require_columns(shots_df, {xg_col, player_col}, "compute_shot_creation_quality")
    has_pressure = pressure_col in shots_df.columns

    grouped = shots_df.groupby(player_col)
    agg = grouped.agg(
        total_shots=(xg_col, "size"),
        avg_xg_per_shot=(xg_col, "mean"),
        total_xg=(xg_col, "sum"),
    ).reset_index()

    if "is_goal" in shots_df.columns:
        agg = agg.merge(grouped["is_goal"].sum().reset_index(name="goals"), on=player_col)

    if has_pressure:
        pressure_share = grouped[pressure_col].apply(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0).mean()
        ).reset_index(name="pct_under_pressure")
        agg = agg.merge(pressure_share, on=player_col)
    else:
        agg["pct_under_pressure"] = np.nan

    return agg[agg["total_shots"] >= min_shots].sort_values("total_xg", ascending=False)


def plot_creation_quality_quadrant(agg_df: pd.DataFrame, player_col: str = "player", label_top_n: int = 8, save_path: str | None = None):
    """Scatter 4 quadrants — Volume (X) vs xG moyen/tir (Y), couleur = % sous pression."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    x, y, c = agg_df["total_shots"], agg_df["avg_xg_per_shot"], agg_df["pct_under_pressure"]
    sc = ax.scatter(x, y, c=c, cmap="RdYlGn_r", s=70, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axvline(x.median(), color="#666666", linestyle="--", linewidth=1)
    ax.axhline(y.median(), color="#666666", linestyle="--", linewidth=1)

    for _, row in agg_df.nlargest(label_top_n, "total_xg").iterrows():
        ax.annotate(row[player_col], (row["total_shots"], row["avg_xg_per_shot"]),
                    color="white", fontsize=8, xytext=(4, 4), textcoords="offset points")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Part des tirs sous pression défensive", color="white")

    ax.set_xlabel("Volume de tirs (brut — pas de normalisation /90 possible, voir note)", color="white")
    ax.set_ylabel("Qualité moyenne du tir (xG modèle / tir)", color="white")
    ax.set_title("Scouting — Créateurs d'occasions : volume vs qualité", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Delta xG — Diagnostic sur/sous-performance
# ---------------------------------------------------------------------------

def compute_delta_xg(shots_df: pd.DataFrame, xg_col: str = "model_xg", goal_col: str = "is_goal",
                      player_col: str = "player", min_shots: int = 10) -> pd.DataFrame:
    _require_columns(shots_df, {xg_col, goal_col, player_col}, "compute_delta_xg")
    agg = shots_df.groupby(player_col).agg(
        total_shots=(xg_col, "size"), total_xg=(xg_col, "sum"), total_goals=(goal_col, "sum"),
    ).reset_index()
    agg = agg[agg["total_shots"] >= min_shots]
    agg["delta_xg"] = agg["total_goals"] - agg["total_xg"]
    return agg.sort_values("delta_xg", ascending=False)


def plot_delta_xg_diagonal(agg_df: pd.DataFrame, player_col: str = "player", label_top_n: int = 6, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    x, y = agg_df["total_xg"], agg_df["total_goals"]
    max_val = max(x.max(), y.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], color="#666666", linestyle="--", linewidth=1, label="y = x")

    colors = np.where(agg_df["delta_xg"] >= 0, "#2ecc71", "#e74c3c")
    ax.scatter(x, y, c=colors, s=70, edgecolor="white", linewidth=0.5, alpha=0.9)

    labels = pd.concat([agg_df.nlargest(label_top_n // 2, "delta_xg"), agg_df.nsmallest(label_top_n // 2, "delta_xg")])
    for _, row in labels.iterrows():
        ax.annotate(row[player_col], (row["total_xg"], row["total_goals"]),
                    color="white", fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_xlabel("xG cumulé (modèle)", color="white")
    ax.set_ylabel("Buts réels marqués", color="white")
    ax.set_title("Delta xG — Sur/Sous-performance", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", edgecolor="#444444")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig, ax