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


def plot_shot_map(
    shots_df,
    mode="all",
    ax=None,
    xg_col="model_xg",
    title=None,
    annotate_summary=True,
    save_path=None,
    face_color="#1a1a1a",
    line_color="white",
):
    """
    mode="all"   -> délègue à draw_xg_weighted_heatmap() (fonction existante,
                    déjà validée sur les 4 matchs MVP) : heatmap KDE pondérée
                    + nuage de tirs (non-buts cercle, buts étoile).
    mode="goals" -> terrain nu (draw_statsbomb_pitch) + uniquement les buts,
                    avec annotation textuelle par but (joueur, minute, xG).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=face_color)
    else:
        fig = ax.get_figure()

    if mode == "all":
        ax, cf, excluded = draw_xg_weighted_heatmap(
            shots_df, ax=ax, xg_col=xg_col,
            face_color=face_color, line_color=line_color,
        )
        if excluded > 0:
            import logging
            logging.warning(f"plot_shot_map(mode='all') : {excluded} tirs exclus (hors bornes terrain)")

    elif mode == "goals":
        draw_statsbomb_pitch(ax, line_color=line_color, face_color=face_color)
        goals = shots_df[shots_df["is_goal"] == 1].copy()
        ax.scatter(goals["x"], goals["y"], marker="*", s=320, color="#f1c40f",
                   edgecolor="#e74c3c", linewidth=1.4, zorder=6)
        for _, row in goals.iterrows():
            xg_val = row[xg_col] if xg_col in row and pd.notna(row[xg_col]) else float("nan")
            label = f"{row.get('player', '?')} {int(row['minute'])}' (xG={xg_val:.2f})"
            ax.annotate(label, (row["x"], row["y"]), color="white", fontsize=8,
                        xytext=(6, 6), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a", ec="#f1c40f", alpha=0.85))
    else:
        raise ValueError(f"mode inconnu : {mode!r} (attendu 'all' ou 'goals')")

    if annotate_summary:
        add_summary_textbox(ax, shots_df, xg_col=xg_col)
    if title:
        ax.set_title(title, color="white", fontweight="bold", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=face_color)
    return fig, ax


def add_summary_textbox(ax, shots_df, xg_col="model_xg"):
    total_xg = shots_df[xg_col].sum() if xg_col in shots_df.columns else 0.0
    total_goals = shots_df["is_goal"].sum() if "is_goal" in shots_df.columns else 0
    n_shots = len(shots_df)
    text = f"Tirs : {n_shots}   |   xG cumulé : {total_xg:.2f}   |   Buts réels : {int(total_goals)}"
    ax.text(0.5, -0.06, text, transform=ax.transAxes, ha="center", color="white",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#444444"))


def add_tactical_inset(*args, **kwargs):
    """
    NON IMPLÉMENTÉ dans cette phase — scope confirmé par l'audit 4.0, pas bloqué.
    Le freeze_frame brut (positions x,y, teammate) EXISTE dans data/raw/*.parquet
    (colonne shot_freeze_frame), mais N'EST PAS conservé dans all_shots_scored.parquet.
    Implémentation nécessite : jointure match_id + identifiant du tir vers le
    fichier raw correspondant (train/comp{N}_season27.parquet selon competition_id,
    ou raw/showcase/comp16_match18243.parquet pour le showcase) pour récupérer
    le freeze_frame du tir concerné. Hors scope de cette phase — à scoper séparément.
    """
    raise NotImplementedError("add_tactical_inset : nécessite jointure vers data/raw/ (scope confirmé, non implémenté ici).")


def _assign_pitch_sides(shots_df, team_col="team"):
    """
    Assigne un côté d'affichage (gauche/droite) à chaque équipe du match.

    LIMITE DE DONNÉES : aucun flag home/away n'existe dans le schéma confirmé.
    Le côté gauche/droite est une CONVENTION D'AFFICHAGE (ordre alphabétique),
    pas une donnée factuelle domicile/extérieur — à mentionner dans toute légende.
    """
    teams = sorted(shots_df[team_col].dropna().unique())
    if len(teams) != 2:
        raise ValueError(f"Attendu exactement 2 équipes, trouvé {len(teams)} : {teams}")
    return teams[0], teams[1]  # team_left, team_right


def _mirror_shots_for_side(shots_df, side, x_col="x", pitch_length=120):
    """
    Retourne une copie du dataframe avec x inversé (x' = 120 - x) si side == "left".

    RAPPEL MÉTHODOLOGIQUE : les coordonnées sont normalisées par sens d'attaque
    PROPRE À CHAQUE ÉQUIPE (rapport du 22/08, vérifié sur Sporting Gijón,
    match 3825739 : x moyen quasi identique period=1 vs period=2). Résultat :
    toutes les équipes pointent par défaut vers x=120. Cette fonction inverse
    uniquement l'AFFICHAGE de l'équipe assignée "gauche" — aucune donnée
    sous-jacente n'est modifiée, seule une copie locale l'est.
    """
    df = shots_df.copy()
    if side == "left":
        df[x_col] = pitch_length - df[x_col]
    return df


def generate_opta_style_match_report(
    shots_df,
    match_id,
    xg_col="model_xg",
    x_col="x", y_col="y",
    save_path=None,
    face_color="#1a1a1a",   # inchangé, comme demandé
    line_color="white",     # inchangé, comme demandé
    header_color="#0d0d0d",
):
    """
    Rapport de match façon Opta : terrain unique avec les deux équipes en
    miroir gauche/droite, bandeau d'en-tête, comparatif de stats central.

    LIMITES DOCUMENTÉES (ne pas contourner) :
    - Côté gauche/droite = convention alphabétique, pas domicile/extérieur.
    - Pas de possession — absente du schéma, jamais affichée.
    - "Tirs cadrés" affiché SEULEMENT si shot_outcome contient bien les
      catégories 'Goal'/'Saved' après vérification réelle — sinon omis.
    - Statut toujours "FULL TIME" : données 2015/16 déjà terminées, jamais du direct.
    """
    match_shots = shots_df[shots_df["match_id"] == match_id].copy()
    if match_shots.empty:
        raise ValueError(f"Aucun tir trouvé pour match_id={match_id}")

    team_left, team_right = _assign_pitch_sides(match_shots)

    left_raw = match_shots[match_shots["team"] == team_left]
    right_raw = match_shots[match_shots["team"] == team_right]
    left_mirrored = _mirror_shots_for_side(left_raw, "left", x_col)
    display_shots = pd.concat([left_mirrored, right_raw], ignore_index=True)

    def _team_stats(team_df):
        stats = {
            "goals": int(team_df["is_goal"].sum()),
            "xg": team_df[xg_col].sum(),
            "shots": len(team_df),
        }
        if "shot_outcome" in team_df.columns:
            known_on_target = {"Goal", "Saved", "Saved to Post"}
            present = set(team_df["shot_outcome"].dropna().unique())
            if known_on_target & present:
                stats["on_target"] = int(team_df["shot_outcome"].isin(known_on_target).sum())
        return stats

    stats_l, stats_r = _team_stats(left_raw), _team_stats(right_raw)

    fig = plt.figure(figsize=(13, 10), facecolor=header_color)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 0.9, 5.5], hspace=0.05)
    ax_header, ax_stats, ax_pitch = (fig.add_subplot(gs[i]) for i in range(3))

    for ax in (ax_header, ax_stats):
        ax.set_facecolor(header_color)
        ax.axis("off")

    ax_header.text(0.02, 0.6, team_left, color="white", fontsize=20, fontweight="bold", ha="left", va="center")
    ax_header.text(0.5, 0.6, f"{stats_l['goals']} - {stats_r['goals']}", color="white",
                    fontsize=28, fontweight="bold", ha="center", va="center")
    ax_header.text(0.98, 0.6, team_right, color="white", fontsize=20, fontweight="bold", ha="right", va="center")
    ax_header.text(0.5, 0.1, f"Match {match_id} — FULL TIME", color="#999999", fontsize=11, ha="center", va="center")

    rows = [("xG", stats_l["xg"], stats_r["xg"], ".2f"), ("Tirs", stats_l["shots"], stats_r["shots"], "d")]
    if "on_target" in stats_l and "on_target" in stats_r:
        rows.append(("Cadrés", stats_l["on_target"], stats_r["on_target"], "d"))

    for i, (label, val_l, val_r, fmt) in enumerate(rows):
        y = 1 - (i + 0.5) / len(rows)
        total = (val_l + val_r) or 1
        frac_l = val_l / total
        ax_stats.barh(y, frac_l, left=0, height=0.35, color="#5dade2")
        ax_stats.barh(y, 1 - frac_l, left=frac_l, height=0.35, color="#e74c3c")
        ax_stats.text(0.02, y, f"{val_l:{fmt}}", color="white", fontsize=11, fontweight="bold", va="center", ha="left")
        ax_stats.text(0.98, y, f"{val_r:{fmt}}", color="white", fontsize=11, fontweight="bold", va="center", ha="right")
        ax_stats.text(0.5, y, label, color="#cccccc", fontsize=10, va="center", ha="center")
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)

    draw_statsbomb_pitch(ax_pitch, line_color=line_color, face_color=face_color)
    ax_pitch, cf, excluded = draw_xg_weighted_heatmap(
        display_shots, ax=ax_pitch, xg_col=xg_col,
        face_color=face_color, line_color=line_color, show_pitch=False,
    )
    if excluded > 0:
        import logging
        logging.warning(f"generate_opta_style_match_report(match_id={match_id}) : {excluded} tirs exclus (hors bornes)")

    ax_pitch.text(0.02, -0.03, f"{team_left} — xG {stats_l['xg']:.2f}", transform=ax_pitch.transAxes,
                  color="white", fontsize=11, ha="left")
    ax_pitch.text(0.98, -0.03, f"{team_right} — xG {stats_r['xg']:.2f}", transform=ax_pitch.transAxes,
                  color="white", fontsize=11, ha="right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=header_color)
    return fig, (ax_header, ax_stats, ax_pitch)