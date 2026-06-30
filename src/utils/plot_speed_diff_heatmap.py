"""
One-time script: Plot the track map colored by speed difference
between vlada1 (high aero) and vlada2 (low aero).

Usage:
    python -m src.utils.plot_speed_diff_heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json

def main():
    d1 = np.load("results/vlada1.npz", allow_pickle=True)
    d2 = np.load("results/vlada2.npz", allow_pickle=True)

    x = d1["x"]
    y = d1["y"]
    v1 = d1["v"]
    v2 = d2["v"]

    dv = np.abs(v1 - v2) * 3.6  # km/h

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    dv_seg = (dv[:-1] + dv[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 8))
    lc = LineCollection(segments, cmap="turbo", norm=plt.Normalize(0, 20), linewidths=3)
    lc.set_array(dv_seg)
    ax.add_collection(lc)

    ax.set_xlim(x.min() - 10, x.max() + 10)
    ax.set_ylim(y.min() - 10, y.max() + 10)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Speed difference")

    cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label("Δv [km/h]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
