
import os
import sys
import time
import json
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers import MassPointSolver
from src.optimization.solvers.base_solver import InitConditions


# ── Configuration ────────────────────────────────────────────────────────────
TRACKS = ["FSCZ"]
SOC_POWER_MODE = "lookup_table"  # "optimum" or "lookup_table"
DS = 1.0

SOC_GRID   = np.linspace(0.10, 1.0, 10)
ENERGY_GRID = np.linspace(0.15, 0.50, 10)
REGEN_GRID = np.linspace(0.25, 0.95, 10)

# Base car parameters 
BASE_CAR_PARAMS = {
    "name":           "CTU26",
    "weight":         290,
    "rotating_mass":  22.0,
    "cd":             2.06,
    "cl":             5.64,
    "cd_drs":         1.17,
    "cl_drs":         4.2,
    "drs_allowed":    False,
    "power_max":      80000,
    "regen_eff":      0.94,
    "soc_start":      1.0,       # overwritten per sweep point
    "M_max":          30.0,
    "model_type":     "mass_point",
    "energy_limit":   None,      # overwritten per sweep point
    
    # Chassis parameters required by Car
    "cop":            0.53,
    "wb":             1.53,
    "tw":             1.20,
    "wd":             0.53,
    "cogz":           0.25,
    "i":              11.46,
}

BATTERIES = [
    {
        "label":        "ep95",
        "battery_type": "ep95",  # Defaults to from_npz parsing
        "npz_file":     os.path.join("src", "core", "vehicles","battery", "EP95_2.npz"),
    },
]



def run_single(car, track, ds, soc_power_mode):
    """Run a single-lap solve and return lap time (or NaN on failure)."""
    solver = MassPointSolver(car, track)
    solver.simulate_full_battery = True
    solver.soc_power_mode = soc_power_mode

    #E_init = car.acp_capacity * (1.0 - car.soc_start)

    if track.speed is not None:
        init_cond = InitConditions(v=track.speed[0], T=0.0, E=0.00)
    else:
        init_cond = InitConditions(v=10.0, T=0.0, E=0.00)

    res = solver.solve(init_cond, ds=ds)

    if res.success:
        return res.lap_time
    else:
        return np.nan


def sweep_battery(battery_cfg, track, soc_grid, energy_grid, regen_grid, soc_power_mode):


    n_soc = len(soc_grid)
    n_energy = len(energy_grid)
    n_regen = len(regen_grid)
    total = n_soc * n_energy * n_regen
    lap_times = np.full((n_soc, n_energy, n_regen), np.nan)

    print(f"\n{'='*60}")
    print(f"  Battery: {battery_cfg['label']}")
    print(f"  Grid:    {n_soc} SoC  ×  {n_energy} Energy  ×  {n_regen} Regen  =  {total} runs")
    print(f"{'='*60}")

    t0 = time.time()
    count = 0

    for i, soc in enumerate(soc_grid):
        for j, e_lim in enumerate(energy_grid):
            for l, r_eff in enumerate(regen_grid):
                count += 1
                tag = f"[{count:3d}/{total}]"

                # Build a fresh car for each point
                params = BASE_CAR_PARAMS.copy()
                params["soc_start"] = float(soc-1e-3)
                params["energy_limit"] = float(e_lim)
                params["regen_eff"] = float(r_eff)
                car = Car(**params)

                # Load battery1
                car.load_battery(battery_cfg["npz_file"],
                                battery_type=battery_cfg["battery_type"])

                try:
                    lt = run_single(car, track, DS, soc_power_mode)
                    lap_times[i, j, l] = lt
                    status = f"{lt:.3f} s" if not np.isnan(lt) else "FAILED"
                except Exception as exc:
                    lap_times[i, j, l] = np.nan
                    status = f"ERROR: {exc}"

                print(f"  {tag}  SoC={soc:.2f}  E_lim={e_lim:.3f} kWh  Regen={r_eff:.2f}  ->  {status}")

    elapsed = time.time() - t0
    valid = np.count_nonzero(~np.isnan(lap_times))
    print(f"\n  Done in {elapsed:.1f} s  ({valid}/{total} valid)")

    return lap_times


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "data", "tracks", "lapmaps")
    os.makedirs(out_dir, exist_ok=True)
    mode_suffix = "opt" if SOC_POWER_MODE == "optimum" else "lookup"

    for track_name in TRACKS:
        # Load track
        try:
            track = Track.load(track_name)
            print(f"\nTrack loaded: {track.name}  (length={track.length:.1f} m)")
        except FileNotFoundError:
            print(f"Track '{track_name}' not found in data/tracks/")
            continue

        for bat_cfg in BATTERIES:
            lap_times = sweep_battery(bat_cfg, track, SOC_GRID, ENERGY_GRID, REGEN_GRID, SOC_POWER_MODE)

            params_to_save = BASE_CAR_PARAMS.copy()
            params_to_save["battery_name"] = bat_cfg["label"]
            params_to_save["soc_power_mode"] = SOC_POWER_MODE

            # get acp capacity from the loaded battery
            temp_car = Car(**BASE_CAR_PARAMS)
            temp_car.load_battery(bat_cfg["npz_file"], battery_type=bat_cfg["battery_type"])
            params_to_save["acp_capacity"] = temp_car.acp_capacity

            # Pop parameters we don't want to save in npz
            for key in ["cop", "wb", "tw", "wd", "cogz", "i"]:
                params_to_save.pop(key, None)

            filename = f"{track.name.lower()}_{bat_cfg['label'].lower()}_{mode_suffix.lower()}.npz"
            out_file = os.path.join(out_dir, filename)
            np.savez(
                out_file,
                soc_grid=SOC_GRID,
                energy_grid=ENERGY_GRID,
                regen_grid=REGEN_GRID,
                lap_times=lap_times,
                battery=bat_cfg["label"],
                track=track_name,
                car_params=json.dumps(params_to_save),
                ds=DS,
            )
            print(f"\n  Saved -> {out_file}")

    print("\nDone. All done.")


if __name__ == "__main__":
    main()
