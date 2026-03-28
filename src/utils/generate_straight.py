"""
src/utils/generate_straight.py

Generates a straight track .npz file compatible with Track.load().

Usage:
    python -m src.utils.generate_straight --length 200 --ds 0.1 --name straight_200m
    python -m src.utils.generate_straight --length 75  --ds 0.5 --name my_straight

Output:
    data/tracks/<name>.npz

Track layout:
    - x = 0        (no lateral displacement)
    - y = 0 -> L   (forward direction, matches arc-length)
    - kappa = 0    (no curvature)
    - z = 0        (flat)
    - psi = pi/2   (heading in +y direction)

    Saves only the 6 mandatory keys (s, x, y, z, kappa, psi),
    matching the format of the original straight_50m.npz.
    Optional telemetry (drs, alpha, speed, brake, power) is omitted;
    track.py handles their absence gracefully.
"""

import argparse
import os
import sys

# Allow running directly: python src/utils/generate_straight.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np


def generate_straight(length: float, ds: float = 0.1, output_dir: str = "data/tracks") -> dict:
    """
    Build arrays for a flat, zero-curvature straight of the given length.

    Parameters
    ----------
    length : float
        Total length of the straight [m].
    ds : float
        Spatial step size [m]. Defaults to 0.1 m.
    output_dir : str
        Directory where the .npz file will be saved.

    Returns
    -------
    dict with keys: s, x, y, z, kappa, psi
    """
    N = max(2, int(round(length / ds)) + 1)
    s = np.linspace(0.0, length, N)

    x     = np.zeros(N)               # no lateral displacement
    y     = s.copy()                  # forward position = arc-length
    z     = np.zeros(N)               # flat track
    kappa = np.zeros(N)               # zero curvature
    psi   = np.full(N, np.pi / 2)    # heading: +y direction

    return dict(s=s, x=x, y=y, z=z, kappa=kappa, psi=psi)


def save_straight(name: str, length: float, ds: float = 0.1,
                  output_dir: str = "data/tracks") -> str:
    """
    Generate and save a straight track as <output_dir>/<name>.npz.

    Returns the full path to the saved file.
    """
    data = generate_straight(length=length, ds=ds)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.npz")
    np.savez(filepath, **data)
    print(f"[generate_straight] Saved '{name}' ({length} m, ds={ds} m, "
          f"{len(data['s'])} points) -> {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a straight track .npz file for optiSim."
    )
    parser.add_argument(
        "--length", "-L", type=float, required=True,
        help="Length of the straight [m]."
    )
    parser.add_argument(
        "--ds", type=float, default=0.1,
        help="Spatial resolution [m] (default: 0.1)."
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None,
        help="Output file name without extension (default: straight_<length>m)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/tracks",
        help="Directory to save the .npz file (default: data/tracks)."
    )
    args = parser.parse_args()

    track_name = args.name if args.name else f"straight_{int(args.length)}m"
    save_straight(
        name=track_name,
        length=args.length,
        ds=args.ds,
        output_dir=args.output_dir,
    )
