"""
src/viz/pitch.py

Convention StatsBomb : pitch 120 (longueur, x) x 80 (largeur, y).
Le but attaqué par convention "vers la droite" est en x=120.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def draw_statsbomb_pitch(ax, line_color="white", face_color="#1a1a1a"):
    ax.set_facecolor(face_color)

    ax.plot([0, 0, PITCH_LENGTH, PITCH_LENGTH, 0],
             [0, PITCH_WIDTH, PITCH_WIDTH, 0, 0], color=line_color, lw=1.5, zorder=5)
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color=line_color, lw=1.5, zorder=5)
    ax.add_patch(plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15,
                              color=line_color, fill=False, lw=1.5, zorder=5))
    ax.plot(PITCH_LENGTH / 2, PITCH_WIDTH / 2, marker="o", color=line_color, markersize=2, zorder=5)

    for gx, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([gx, gx + direction * 18, gx + direction * 18, gx],
                 [18, 18, 62, 62], color=line_color, lw=1.5, zorder=5)
        ax.plot([gx, gx + direction * 6, gx + direction * 6, gx],
                 [30, 30, 50, 50], color=line_color, lw=1.5, zorder=5)
        pen_x = gx + direction * 12
        ax.plot(pen_x, PITCH_WIDTH / 2, marker="o", color=line_color, markersize=2, zorder=5)
        half_angle_deg = np.degrees(np.arccos(6 / 9.15))
        # L'arc doit bulger vers l'exterieur de la surface : +x (theta~0) pour le but
        # gauche (direction=1), -x (theta~180) pour le but droit (direction=-1).
        if direction == 1:
            theta1, theta2 = -half_angle_deg, half_angle_deg
        else:
            theta1, theta2 = 180 - half_angle_deg, 180 + half_angle_deg
        ax.add_patch(Arc((pen_x, PITCH_WIDTH / 2), height=18.3, width=18.3,
                          angle=0, theta1=theta1, theta2=theta2, color=line_color, lw=1.5, zorder=5))
        goal_depth = 2
        ax.plot([gx - direction * goal_depth, gx - direction * goal_depth],
                 [36, 44], color=line_color, lw=1.5, zorder=5)
        ax.plot([gx, gx - direction * goal_depth], [36, 36], color=line_color, lw=1.5, zorder=5)
        ax.plot([gx, gx - direction * goal_depth], [44, 44], color=line_color, lw=1.5, zorder=5)

    ax.set_xlim(-3, PITCH_LENGTH + 3)
    ax.set_ylim(-3, PITCH_WIDTH + 3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
