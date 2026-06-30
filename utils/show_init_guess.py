
import os
import sys
import ctypes


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from src.core.car import Car
from src.core.track import Track
from src.utils.config_manager import ConfigManager
from src.optimization.solvers.init_guess import calculate_initial_guess
from src.optimization.solvers.base_solver import SolverResults, InitConditions
from src.ui.results_window import ResultsWindow
from src.ui.style_manager import StyleManager


# ── helpers ───────────────────────────────────────────────────────────────────

def _first_track() -> str:
    import glob
    files = glob.glob(os.path.join(ROOT, "data", "tracks", "*.npz"))
    if not files:
        raise FileNotFoundError("No tracks found in data/tracks/")
    return os.path.splitext(os.path.basename(sorted(files)[0]))[0]


def _first_vehicle() -> str:
    vehicles = ConfigManager.get_available_vehicles()
    if not vehicles:
        raise FileNotFoundError("No vehicle profiles found.")
    return vehicles[0]


def build_results_from_init_guess(car: Car, track: Track, ds: float) -> SolverResults:
    """
    Calls calculate_initial_guess and wraps its output in a SolverResults so
    the ResultsWindow can display it directly.
    """
    E_start = car.acp_capacity * (1 - car.soc_start)

    if track.speed is not None:
        init_cond = InitConditions(v=float(track.speed[0]), T=0.0, E=E_start)
    else:
        init_cond = InitConditions(v=10.0, T=0.0, E=E_start)

    guess = calculate_initial_guess(car, track, init_cond, ds=ds)

    v   = np.asarray(guess["v"])
    T   = np.asarray(guess["T"])
    E   = np.asarray(guess["E"])
    F_a = np.asarray(guess["F_acc"])
    F_b = np.asarray(guess["F_brake"])
    N   = len(v)  # N+1 points

    # s-axis: same resolution as the guess
    total_length = track.length
    s = np.linspace(0, total_length, N)

    # lap time is the last cumulative time value
    lap_time = float(T[-1] - T[0])

    # recuperation: negative-force portions of F_brake
    rho = 1.225
    v_mid = 0.5 * (v[:-1] + v[1:])
    recuperation = np.maximum(-F_b, 0.0) * v_mid * car.regen_eff

    # DRS usage: all zeros (init guess doesn't model DRS)
    drs_usage = np.zeros(N - 1)

    # SoC: convert E array from kWh to % (E is cumulative energy used)
    capacity_kwh = car.acp_capacity        # total battery capacity [kWh]
    E_arr = np.asarray(E)
    soc_pct = np.clip(
        (1.0 - E_arr / capacity_kwh) * 100.0,
        0.0, 100.0
    )

    res = SolverResults(
        success=True,
        lap_time=lap_time,
        t=T,
        v=v,
        s=s,
        f_acc=F_a,
        f_brake=F_b,
        recuperation=recuperation,
        E=E_arr,
        drs_usage=drs_usage,
        SoC=soc_pct,
    )
    return res


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # --- resolve track / vehicle from CLI args or defaults ---
    track_name   = sys.argv[1] if len(sys.argv) > 1 else _first_track()
    vehicle_name = sys.argv[2] if len(sys.argv) > 2 else _first_vehicle()
    ds = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    print(f"[show_init_guess] track='{track_name}'  vehicle='{vehicle_name}'  ds={ds}")

    # --- Qt app setup ---
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("optisim.initguess")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet(StyleManager.get_global_stylesheet())
    icon_path = os.path.join(ROOT, "data", "logo.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    app.setStyle("Fusion")

    # --- load car & track ---
    default_params = {
        "acp_capacity": 6.0, "weight": 280.0, "rotating_mass": 22.0,
        "wb": 1.53, "tw": 1.25, "wd": 0.53, "cogz": 0.3, "i": 11.46,
        "M_max": 30.0, "cd": 1.83, "cd_drs": 1.17, "cl": 5.64,
        "cl_drs": 4.2, "cop": 0.53, "power_max": 80_000.0,
        "regen_eff": 0.8, "drs_allowed": True, "soc_start": 0.67,
        "model_type": "Mass point (Rk4)",
    }

    raw_params = ConfigManager.load_vehicle(vehicle_name, default_params)

    # convert the UI-stored power_limit → power_max if needed
    if "power_limit" in raw_params and "power_max" not in raw_params:
        raw_params["power_max"] = raw_params.pop("power_limit") * 1000.0
    elif "power_limit" in raw_params:
        raw_params.pop("power_limit")

    # drop keys that Car.__init__ doesn't accept
    for key in ("laps", "ds", "mass"):
        raw_params.pop(key, None)

    try:
        car = Car(**raw_params)
    except Exception as e:
        QMessageBox.critical(None, "Car error", f"Could not create car:\n{e}")
        sys.exit(1)

    try:
        track = Track.load(track_name)
    except Exception as e:
        QMessageBox.critical(None, "Track error", f"Could not load track '{track_name}':\n{e}")
        sys.exit(1)

    # --- compute init guess ---
    print("[show_init_guess] computing initial guess …")
    try:
        res = build_results_from_init_guess(car, track, ds)
    except Exception as e:
        QMessageBox.critical(None, "Init guess error", f"Failed:\n{e}")
        sys.exit(1)

    print(f"[show_init_guess] done – predicted lap time: {res.lap_time:.3f} s")

    # --- open results window on the 'Dynamics' tab (index 0, default) ---
    win = ResultsWindow(res, car, track, ds)
    # Jump straight to the SoC tab if available, otherwise stay on Dynamics
    # (Tab index: 0=Dynamics, 1=Powertrain, 2=Forces, 3=Sensitivity,
    #             4=Power Map, 5=DRS Map, 6=Battery)
    # Uncomment the line below to start on a specific tab:
    # win.tabs.setCurrentIndex(6)  # Battery / SoC tab

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
