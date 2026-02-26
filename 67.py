import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tkinter import Tk, filedialog
import time
from src.core.car import Car

# --- IMPORTS FROM YOUR PROJECT ---
# Ujisti se, že cesty sedí k tvé struktuře složek
from src.core.trackGenerator import TrackGenerator
from src.optimization.solver  import MassPointSolver, InitConditions
from src.core.car import Car 

# ==========================================
# 1. HELPER CLASS FOR SOLVER COMPATIBILITY
# ==========================================
class TrackWrapper:
    """
    Solver očekává objekt 'track', který má atribut .length a metodu .get_curvature(s).
    TrackGenerator vrací slovník (dict), takže to musíme zabalit.
    """
    def __init__(self, track_data):
        self.s = track_data['s']
        self.kappa = track_data['kappa']
        self.length = self.s[-1]
    
    def get_curvature(self, s_query):
        # Solver se ptá na křivost v libovolném bodě 's', musíme interpolovat
        return np.interp(s_query, self.s, self.kappa)

# ==========================================
# 2. RMSE CALCULATION
# ==========================================
def calculate_rmse(s_real, v_real, s_sim, v_sim):
    """
    Vypočítá chybu mezi realitou a simulací.
    Protože osy 's' nesedí přesně (simulace může mít jiný krok), interpolujeme simulaci na body reality.
    """
    # Omezíme se na rozsah, kde se překrývají
    s_min = max(s_real[0], s_sim[0])
    s_max = min(s_real[-1], s_sim[-1])
    
    # Vybereme body reality v tomto rozsahu
    mask = (s_real >= s_min) & (s_real <= s_max)
    s_eval = s_real[mask]
    v_real_eval = v_real[mask]
    
    # Interpolace simulace na tyto body
    f_sim = interp1d(s_sim, v_sim, kind='linear', bounds_error=False, fill_value="extrapolate")
    v_sim_eval = f_sim(s_eval)
    
    # RMSE
    mse = np.mean((v_real_eval - v_sim_eval)**2)
    return np.sqrt(mse)

# ==========================================
# 3. MAIN GRID SEARCH LOGIC
# ==========================================
def main():
    
    car = Car(
    mass=280.0,
    cd=1.9,
    cl=5.6,
    mu_x=1.2,
    mu_y=1.2,
    energy_limit=None,     # kWh (nebo None)
    acp_capacity=7.2,     # kWh
    drs_allowed=False)

    # --- B. NAČTENÍ DAT (MAT FILE) ---
    root = Tk()
    root.withdraw()
    print("Vyber .mat soubor s daty...")
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    if not file_path:
        print("Nevybrán soubor.")
        return

    all_data = TrackGenerator.filter_data(file_path)
    
    # --- C. VÝBĚR KOLA (MANUAL LAP SELECTOR) ---
    print("Vyber jedno čisté kolo pro Grid Search...")
    lap_info = TrackGenerator.run_lap_selector(all_data)
    
    if not lap_info:
        print("Nevybráno žádné kolo.")
        return

    # --- D. PŘÍPRAVA REALITY (Extrakce dat z výběru) ---
    s_idx, e_idx = lap_info['indices']
    
    # Extrakce výřezu
    raw_lat = all_data['Latitude'][s_idx:e_idx]
    raw_lon = all_data['Longitude'][s_idx:e_idx]
    raw_alt = all_data['Altitude'][s_idx:e_idx]
    raw_speed = all_data['Speed'][s_idx:e_idx] # m/s

    # Převod na metry (to se nemění s parametry r/w, uděláme to jednou)
    x_m, y_m = TrackGenerator.latlon_to_meters(raw_lat, raw_lon)
    
    # Výpočet reálné osy 's' (ujetá vzdálenost)
    # Pozn: Toto je "hrubá" vzdálenost z GPS, smoothing ji může trochu změnit,
    # ale pro RMSE je to naše reference.
    s_real = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))))

    # Počáteční rychlost pro solver (aby startoval stejně jako realita)
    v0_real = raw_speed[0]

    # --- E. DEFINICE GRIDU ---
    # Uprav rozsahy podle potřeby
    r_values = [100, 500, 1000, 2000, 3000, 4000, 5000]       # Smoothing (Regularizace)
    w_values = [1, 10, 50, 100, 200, 400, 500]             # Position Weight (Věrnost GPS)
    
    results = []
    
    print(f"\nSTARTING GRID SEARCH: {len(r_values)} x {len(w_values)} = {len(r_values)*len(w_values)} iterations")
    print("-" * 60)
    print(f"{'r':<8} {'w_pos':<8} {'RMSE [km/h]':<15} {'Time Diff [s]':<15} {'Status'}")
    print("-" * 60)

    start_time_total = time.time()

    for r in r_values:
        for w in w_values:
            try:
                # 1. Vygenerování hladké tratě (smooth_track2)
                # Používáme tvou novou metodu smooth_track2
                track_data = TrackGenerator.smooth_track2(
                    x_m, y_m, raw_alt, 
                    ds=0.5,         # Rozlišení sítě
                    w_pos=w,        # Grid Param
                    r=r,            # Grid Param
                    is_closed=True # Pro single lap raději False, pokud to není 100% uzavřené
                )
                
                if track_data is None:
                    print(f"{r:<8} {w:<8} {'FAILED (Gen)':<30}")
                    continue

                # 2. Zabalení do objektu pro solver
                track_obj = TrackWrapper(track_data)

                # 3. Setup Solveru
                solver = MassPointSolver(car, track_obj, endurance=False)
                
                # Init podmínky (startujeme s rychlostí reality)
                init_cond = InitConditions(v=v0_real-1, T=0.0, E=0.0)

                # 4. Spuštění simulace
                sol = solver.solveLap(init_cond, ds=1) # ds pro solver může být hrubší než pro track gen

                if not sol.success:
                    print(f"{r:<8} {w:<8} {'FAILED (Sol)':<30}")
                    continue

                # 5. Vyhodnocení (RMSE)
                rmse_val = calculate_rmse(s_real, raw_speed, sol.s, sol.v)
                rmse_kmh = rmse_val * 3.6
                time_diff = sol.lap_time - lap_info['time']
                combined_score = (rmse_kmh ** 2) * abs(time_diff)

                # Uložení
                res = {
                    'r': r,
                    'w': w,
                    'rmse': rmse_kmh,
                    'sim_time': sol.lap_time,
                    'real_time': lap_info['time'],
                    'time_diff': abs(time_diff),
                    'score': combined_score,
                    'sol_s': sol.s,
                    'sol_v': sol.v
                }
                results.append(res)

                print(f"{r:<8} {w:<8} {rmse_kmh:<15.3f} {time_diff:<15.3f} OK")

            except Exception as e:
                 print(f"{r:<8} {w:<8} ERROR: {str(e)}")

    # --- F. VÝSLEDKY ---
    if not results:
        print("No successful runs.")
        return

    # Seřazení podle RMSE
    df = pd.DataFrame(results)
    df = df.sort_values(by='rmse')

    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS:")
    print(df[['r', 'w', 'rmse', 'time_diff', 'score']].head(10).to_string(index=False))
    print("="*60)

    best = df.iloc[0]
    print(f"\nWINNER: r={best['r']}, w={best['w']} with RMSE={best['rmse']:.3f} km/h")

    # --- G. PLOT BEST FIT ---
    plt.figure(figsize=(12, 6))
    
    # Realita
    plt.plot(s_real, raw_speed * 3.6, 'k-', label='Real Telemetry', alpha=0.6, linewidth=1.5)
    
    # Simulace (Best)
    plt.plot(best['sol_s'], best['sol_v'] * 3.6, 'r--', label=f'Sim (r={best["r"]}, w={best["w"]})', linewidth=2)
    
    plt.title(f"Model Correlation Result (RMSE: {best['rmse']:.2f} km/h)")
    plt.xlabel("Distance [m]")
    plt.ylabel("Speed [km/h]")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()