"""
src/viz/heatmap.py

Génération de heatmaps pondérées par les xG (KDE - Kernel Density Estimation)
sur terrain StatsBomb (120x80).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    from src.viz.pitch import draw_statsbomb_pitch, PITCH_LENGTH, PITCH_WIDTH
except ImportError:
    from pitch import draw_statsbomb_pitch, PITCH_LENGTH, PITCH_WIDTH


def filter_in_pitch_shots(df_shots, x_col='x', y_col='y', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """
    Exclut tout tir dont x est hors [0, pitch_length] ou y hors [0, pitch_width].
    Retourne (df_filtre, nombre_tirs_exclus).
    """
    if df_shots.empty:
        return df_shots.copy(), 0

    valid_mask = (
        (df_shots[x_col] >= 0) & (df_shots[x_col] <= pitch_length) &
        (df_shots[y_col] >= 0) & (df_shots[y_col] <= pitch_width) &
        df_shots[x_col].notna() & df_shots[y_col].notna()
    )
    excluded_count = int((~valid_mask).sum())
    return df_shots[valid_mask].copy(), excluded_count


def draw_xg_weighted_heatmap(
    df_shots,
    ax=None,
    x_col='x',
    y_col='y',
    xg_col='model_xg',
    cmap='hot',
    alpha=0.65,
    bw_method=0.25,
    grid_size=200,
    levels=40,
    show_pitch=True,
    scatter_shots=True,
    face_color="#1a1a1a",
    line_color="white"
):
    """
    Trace une heatmap xG-pondérée (Gaussian KDE) sur un terrain StatsBomb.
    
    Contrainte stricte : Les tirs hors [0, 120] x [0, 80] sont exclus
    AVANT tout calcul KDE et pondération.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    if show_pitch:
        draw_statsbomb_pitch(ax, line_color=line_color, face_color=face_color)

    # 1. Filtrage strict pré-KDE
    valid_shots, excluded_count = filter_in_pitch_shots(
        df_shots, x_col=x_col, y_col=y_col, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH
    )

    if len(valid_shots) == 0:
        return ax, None, excluded_count

    # 2. Préparation des coordonnées et poids xG
    x = valid_shots[x_col].values
    y = valid_shots[y_col].values
    
    if xg_col in valid_shots.columns and valid_shots[xg_col].notna().any():
        weights = valid_shots[xg_col].fillna(0.0).values
        # Si la somme des poids est nulle, bascule sur poids uniformes
        if weights.sum() <= 0:
            weights = np.ones_like(x, dtype=float)
    else:
        weights = np.ones_like(x, dtype=float)

    # 3. Calcul du KDE si au moins 2 points non identiques
    cf = None
    if len(x) >= 2:
        try:
            # Vérifier variance minimale pour éviter matrice de covariance singulière
            if np.var(x) > 1e-4 and np.var(y) > 1e-4:
                kde = gaussian_kde(np.vstack([x, y]), weights=weights, bw_method=bw_method)
                
                # Grille régulière bornée au terrain
                xi = np.linspace(0, PITCH_LENGTH, grid_size)
                yi = np.linspace(0, PITCH_WIDTH, grid_size)
                X, Y = np.meshgrid(xi, yi)
                grid_coords = np.vstack([X.ravel(), Y.ravel()])
                
                Z = kde(grid_coords).reshape(X.shape)
                
                # Masquage des densités quasi-nulles pour lisibilité
                threshold = Z.max() * 0.05
                Z_masked = np.ma.masked_where(Z < threshold, Z)
                
                cf = ax.contourf(X, Y, Z_masked, levels=levels, cmap=cmap, alpha=alpha, zorder=3)
        except Exception:
            pass

    # 4. Affichage optionnel des tirs individuels
    if scatter_shots and len(valid_shots) > 0:
        goals = valid_shots[valid_shots['is_goal'] == 1] if 'is_goal' in valid_shots.columns else pd.DataFrame()
        non_goals = valid_shots[valid_shots['is_goal'] == 0] if 'is_goal' in valid_shots.columns else valid_shots

        if not non_goals.empty:
            sizes = non_goals[xg_col] * 350 + 20 if xg_col in non_goals.columns else 40
            ax.scatter(
                non_goals[x_col], non_goals[y_col],
                s=sizes, c="#3498db", edgecolors="white", alpha=0.8,
                label="Tir (non-but)", zorder=6
            )

        if not goals.empty:
            sizes = goals[xg_col] * 350 + 40 if xg_col in goals.columns else 70
            ax.scatter(
                goals[x_col], goals[y_col],
                s=sizes, c="#e74c3c", marker="*", edgecolors="white", lw=1.2,
                label="But", zorder=7
            )

    return ax, cf, excluded_count


def generate_match_heatmap(df_match_shots, match_id, output_path=None, title=None):
    """
    Génère et sauvegarde la heatmap complète pour un match donné.
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a1a")
    
    ax, cf, excluded = draw_xg_weighted_heatmap(
        df_match_shots,
        ax=ax,
        x_col='x',
        y_col='y',
        xg_col='model_xg' if 'model_xg' in df_match_shots.columns else 'statsbomb_xg',
        cmap='hot',
        alpha=0.65,
        scatter_shots=True
    )
    
    total_xg = df_match_shots['model_xg'].sum() if 'model_xg' in df_match_shots.columns else df_match_shots.get('statsbomb_xg', pd.Series([0])).sum()
    goals = df_match_shots['is_goal'].sum() if 'is_goal' in df_match_shots.columns else 0
    total_shots = len(df_match_shots)
    
    default_title = f"xG-Weighted Shot Heatmap (Match ID: {match_id})\nTirs: {total_shots} | Buts: {goals} | xG Total: {total_xg:.2f}"
    ax.set_title(title or default_title, color="white", fontsize=14, pad=12, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower left', facecolor="#2a2a2a", edgecolor="white", labelcolor="white", fontsize=10)

    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path, excluded
    
    return fig, excluded
