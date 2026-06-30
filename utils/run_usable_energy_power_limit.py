"""
Standalone script to calculate usable energy from battery models
under different maximum power limits (80, 60, 40 kW) on the FSCZ track.

Combines exact SoC interpolation (stopping exactly at 8.0% SoC) with 
achievable distance and energy economy metrics. Compares ep95 and melasta.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers import MassPointSolver
from src.optimization.solvers.base_solver import InitConditions

def main():
    # ─── Configuration ────────────────────────────────────────────
    TRACK_NAME = "FSCZ"
    BATTERY_MODELS = ["ep95", "melasta_ad04"]
    MAX_LAPS = 40
    DS = 1.0

    power_limits_kw = [80.0, 60.0, 40.0]
    all_results = {bm: [] for bm in BATTERY_MODELS}

    # ─── Base car configuration ───────────────────────────────────
    base_car_params = {
        "name": "CTU26",
        "weight": 290.0,
        "rotating_mass": 22.0,
        "wb": 1.53,
        "tw": 1.20,
        "wd": 0.53,
        "cogz": 0.25,
        "cd": 1.83,
        "cl": 5.64,
        "cop": 0.53,
        "cd_drs": 1.17,
        "cl_drs": 4.20,
        "M_max": 30.0,
        "i": 12.0,
        "regen_eff": 0.8,
        "drs_allowed": False,
    }

    battery_dir = os.path.join(os.path.dirname(__file__), "..", "src", "core", "vehicles", "battery")

    try:
        track = Track.load(TRACK_NAME)
        print(f"Track: {track.name}, length: {track.length:.0f} m\n")
    except Exception as e:
        print(f"Error loading track {TRACK_NAME}: {e}")
        return

    # ─── Sweep over battery models and power limits ───────────────
    for battery_model in BATTERY_MODELS:
        if battery_model == "melasta_ad04":
            npz_path = os.path.join(battery_dir, "melasta_AD04.npz")
        else:
            npz_path = os.path.join(battery_dir, "EP95_2.npz")

        print(f"\n{'#'*60}")
        print(f"EVALUATING BATTERY: {battery_model.upper()}")
        print(f"{'#'*60}")

        for p_limit_kw in power_limits_kw:
            p_limit_w = p_limit_kw * 1000.0
            print(f"\n{'='*55}")
            print(f"RUNNING POWER LIMIT: {p_limit_kw} kW")
            print(f"{'='*55}")

            car_params = base_car_params.copy()
            car_params["power_max"] = p_limit_w
            car_params["soc_start"] = 1.0
            
            car = Car(**car_params)
            car.load_battery(npz_path, battery_type=battery_model)
            
            solver = MassPointSolver(car, track)
            solver.simulate_full_battery = True
            solver.soc_power_mode = "optimum"

            total_E_pack = 0.0
            total_distance = 0.0
            current_soc = 1.0
            prev_guess = None
            v0 = track.speed[0] if track.speed is not None else 10.0

            global_soc = []
            global_epack = []
            global_dist = []

            for lap_idx in range(MAX_LAPS):
                car.soc_start = current_soc
                init_cond = InitConditions(v=v0, T=0.0, E=0.0)

                try:
                    res = solver.solve(init_cond, ds=DS, init_guess=prev_guess)
                except Exception as e:
                    print(f"  Lap {lap_idx+1}: solver exception — {e}")
                    break

                if not res.success:
                    print(f"  Lap {lap_idx+1}: solver did not converge")
                    break

                soc_array = np.array(res.SoC) / 100.0
                epack_array = np.array(res.E_pack)
                
                N_steps = len(epack_array) - 1
                dist_array = np.linspace(0, N_steps * DS, N_steps + 1)
                
                global_soc.extend(soc_array)
                global_epack.extend(epack_array + total_E_pack)
                global_dist.extend(dist_array + total_distance)

                soc_end = soc_array[-1]
                E_pack_lap = epack_array[-1] if len(epack_array) > 0 else 0.0
                
                print(f"  Lap {lap_idx+1:2d}: t={res.lap_time:.2f}s  SoC {current_soc*100:5.1f}% → {soc_end*100:5.1f}%  E_pack={E_pack_lap:.3f} kWh")

                total_E_pack += E_pack_lap
                total_distance += N_steps * DS

                if soc_end < 0.08:
                    print(f"\n  >> SOC dropped below 8.0%. Stopping simulation.")
                    break

                current_soc = soc_end
                v0 = res.v[-1]
                prev_guess = {
                    'v': res.v,
                    'F_acc': res.f_acc * car.power_eff,
                }

            global_soc = np.array(global_soc)
            global_epack = np.array(global_epack)
            global_dist = np.array(global_dist)

            idx_below = np.where(global_soc <= 0.08)[0]
            
            if len(idx_below) > 0:
                i = idx_below[0]
                if i == 0:
                    exact_usable_energy = global_epack[0]
                    exact_distance = global_dist[0]
                else:
                    soc_prev = global_soc[i-1]
                    soc_curr = global_soc[i]
                    e_prev = global_epack[i-1]
                    e_curr = global_epack[i]
                    d_prev = global_dist[i-1]
                    d_curr = global_dist[i]
                    
                    fraction = (soc_prev - 0.08) / (soc_prev - soc_curr)
                    exact_usable_energy = e_prev + fraction * (e_curr - e_prev)
                    exact_distance = d_prev + fraction * (d_curr - d_prev)
            else:
                exact_usable_energy = global_epack[-1]
                exact_distance = global_dist[-1]
                print("  >> Warning: Did not reach 8.0% SoC within MAX_LAPS.")

            exact_distance_km = exact_distance / 1000.0
            economy_wh_km = (exact_usable_energy * 1000.0) / exact_distance_km if exact_distance_km > 0 else 0

            print(f"--> Exact Usable Energy (to 8.0% SoC): {exact_usable_energy:.3f} kWh")
            print(f"--> Exact Achievable Distance:         {exact_distance_km:.2f} km")
            print(f"--> Energy Economy:                    {economy_wh_km:.1f} Wh/km")

            all_results[battery_model].append({
                'power_kw': p_limit_kw,
                'usable_energy': exact_usable_energy,
                'distance_km': exact_distance_km,
                'economy_wh_km': economy_wh_km
            })

    # ─── Plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Batterkos usable energy comparison')
    
    colors = {"ep95": "#e11d48", "melasta_ad04": "#3b82f6"}
    markers = {"ep95": "o", "melasta_ad04": "s"}
    labels = {"ep95": "EP95", "melasta_ad04": "Melasta"}

    for bm in BATTERY_MODELS:
        res_sorted = sorted(all_results[bm], key=lambda x: x['power_kw'])
        x_vals = [r['power_kw'] for r in res_sorted]
        y_energy = [r['usable_energy'] for r in res_sorted]
        y_dist = [r['distance_km'] for r in res_sorted]
        y_econ = [r['economy_wh_km'] for r in res_sorted]
        
        # Plot 1: Usable Energy
        axes[0].plot(x_vals, y_energy, marker=markers[bm], linestyle='-', color=colors[bm], label=labels[bm], markersize=8)
        
        # Plot 2: Achievable Distance
        axes[1].plot(x_vals, y_dist, marker=markers[bm], linestyle='-', color=colors[bm], label=labels[bm], markersize=8)
        
        # Plot 3: Energy Economy
        axes[2].plot(x_vals, y_econ, marker=markers[bm], linestyle='-', color=colors[bm], label=labels[bm], markersize=8)

        # Annotate points
        for i, txt in enumerate(y_energy):
            axes[0].annotate(f"{txt:.3f}", (x_vals[i], y_energy[i]), textcoords="offset points", xytext=(0,10), ha='center', color=colors[bm])
        for i, txt in enumerate(y_dist):
            axes[1].annotate(f"{txt:.2f}", (x_vals[i], y_dist[i]), textcoords="offset points", xytext=(0,10), ha='center', color=colors[bm])
        for i, txt in enumerate(y_econ):
            axes[2].annotate(f"{txt:.1f}", (x_vals[i], y_econ[i]), textcoords="offset points", xytext=(0,10), ha='center', color=colors[bm])

    # Formatting Plot 1
    axes[0].set_xlabel('Max Power Limit [kW]')
    axes[0].set_ylabel('Usable Capacity [kWh]')
    axes[0].set_title('Total Usable Energy')
    axes[0].grid(True, alpha=0.4, linestyle='--')
    axes[0].set_xticks(power_limits_kw)
    axes[0].legend()

    # Formatting Plot 2
    axes[1].set_xlabel('Max Power Limit [kW]')
    axes[1].set_ylabel('Distance [km]')
    axes[1].set_title('Achievable Distance')
    axes[1].grid(True, alpha=0.4, linestyle='--')
    axes[1].set_xticks(power_limits_kw)
    axes[1].legend()

    # Formatting Plot 3
    axes[2].set_xlabel('Max Power Limit [kW]')
    axes[2].set_ylabel('Economy [Wh/km]')
    axes[2].set_title('Energy Economy')
    axes[2].grid(True, alpha=0.4, linestyle='--')
    axes[2].set_xticks(power_limits_kw)
    axes[2].legend()

    # Expand y-limits slightly to fit annotations
    # Compute global min/max across all batteries
    all_e = [r['usable_energy'] for b in BATTERY_MODELS for r in all_results[b]]
    all_d = [r['distance_km'] for b in BATTERY_MODELS for r in all_results[b]]
    all_ec = [r['economy_wh_km'] for b in BATTERY_MODELS for r in all_results[b]]
    
    if all_e: axes[0].set_ylim(min(all_e) - 0.2, max(all_e) + 0.3)
    if all_d: axes[1].set_ylim(min(all_d) - 1.0, max(all_d) + 1.5)
    if all_ec: axes[2].set_ylim(min(all_ec) - 10, max(all_ec) + 15)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "usable_capacity_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
