import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers import MassPointSolver
from src.optimization.tasks import SingleLapTask

# 1. Battery model options
SIMULATE_BATTERY = False              # Set to True to enable the 2RC battery model simulation
BATTERY_MODEL_PATH = "src/core/vehicles/battery/EP95_2.npz"
BATTERY_TYPE = "ep95"

# 2. Power & SoC Limits (only relevant if simulate battery is true)
SOC_POWER_MODE = "lookup_table"       # "optimum" (optimal cell constraints) or "lookup_table"
SOC_START = 1.0

# 3. Energy Limitation
ENERGY_LIMIT = None                   # Set to a value (in kWh) to limit energy consumption per lap, or None

# 4. Solver Options
USE_WARM_START = False                 # keep false, change for true ONLY if the solver fails to converge
DS_STEP = 1.0                         # Distance step size (m) for discretization (smaller is more accurate but slower)


def configure_car_and_solver(car, solver):

    #setup the car and solver
    car.energy_limit = ENERGY_LIMIT
    car.soc_start = SOC_START

    if SIMULATE_BATTERY:
        # Resolve absolute battery path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_bat_path = os.path.join(project_root, BATTERY_MODEL_PATH)
        car.load_battery(abs_bat_path, battery_type=BATTERY_TYPE)
        
        # Apply solver battery simulation flags
        if hasattr(solver, 'simulate_full_battery'):
            solver.simulate_full_battery = True
        if hasattr(solver, 'soc_power_mode'):
            solver.soc_power_mode = SOC_POWER_MODE
    else:
        if hasattr(solver, 'simulate_full_battery'):
            solver.simulate_full_battery = False


def main():
    # Load track
    try:
        track = Track.load("FSG")
        print(f"Loaded track: {track.name}")
    except FileNotFoundError:
        print("Track not found. Please ensure baked tracks are available in data/tracks/")
        return
        
    # Create base car parameters 
    car_params = {
        "name": "Base Car",
        "weight": 280.0,
        "cd": 1.9,
        "cl": 4.7,
        "cd_drs": 0.8,
        "cl_drs": 1.5,
        "drs_allowed": True,
        "power_max": 80000,
        "regen_eff": 0.9,
        "soc_start": 1.0,
        "cop": 0.5,
        "wb": 1.53,
        "tw": 1.2,
        "wd": 0.5,
        "cogz": 0.3,
        "i": 11.0,
        "M_max": 250.0,
        "rotating_mass": 20.0,
        "model_type": "mass_point"
    }
    
    # Define grid points for the sweep
    mass_grid = np.linspace(0, 3, 10)     # Mass change: 0 to 3 kg
    cd_grid = np.linspace(1.9, 1.4, 10)   # CD from 1.9 down to 1.4
    
    n_mass = len(mass_grid)
    n_cd = len(cd_grid)
    total_runs = n_mass * n_cd
    
    time_deltas = np.zeros((n_mass, n_cd))
    
    # 1. Run base car
    print("Calculating base lap time...")
    base_car = Car(**car_params)
    base_solver = MassPointSolver(base_car, track)
    configure_car_and_solver(base_car, base_solver)
    
    base_task = SingleLapTask(base_solver, base_car, track, ds=DS_STEP, use_warm_start=USE_WARM_START)
    try:
        base_payload = base_task.execute(lambda curr, tot: None)
        base_time = base_payload.data.lap_time
        print(f"Base laptime: {base_time:.3f} s")
    except Exception as e:
        print(f"Base run failed: {e}")
        return
        
    print(f"Executing Mass-CD Grid Sweep ({total_runs} runs)...")
    
    count = 0
    for i, m_change in enumerate(mass_grid):
        for j, cd_val in enumerate(cd_grid):
            count += 1
            
            # Create a fresh car with modified weight and cd
            params = car_params.copy()
            params["weight"] = car_params["weight"] + m_change
            params["cd"] = cd_val
            
            car = Car(**params)
            solver = MassPointSolver(car, track)
            configure_car_and_solver(car, solver)
            
            task = SingleLapTask(solver, car, track, ds=DS_STEP, use_warm_start=USE_WARM_START)
            
            try:
                payload = task.execute(lambda curr, tot: None)
                lt = payload.data.lap_time
                delta = lt - base_time
            except Exception as e:
                print(f"Run failed for Mass change={m_change:.2f} kg, CD={cd_val:.2f}: {e}")
                delta = np.nan
                
            time_deltas[i, j] = delta
            
            if count % 10 == 0 or count == total_runs:
                print(f"Progress: {count}/{total_runs} ({(count/total_runs)*100:.1f}%)")

    # Save data to .npz for post-processing
    save_data_path = os.path.join(os.path.dirname(__file__), "mass_cd_grid_data.npz")
    np.savez(save_data_path, mass_grid=mass_grid, cd_grid=cd_grid, time_deltas=time_deltas, base_time=base_time)
    print(f"Data successfully saved to {save_data_path}")
    
    # Prepare data for plotting
    X, Y = np.meshgrid(mass_grid, cd_grid)
    # time_deltas is [mass_idx, cd_idx], X and Y are [cd_idx, mass_idx], so we need to transpose Z
    Z = time_deltas.T
    
    # Figure 1: 3D Surface
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', alpha=0.9)
    
    ax1.set_xlabel('Mass Change [kg]')
    ax1.set_ylabel('CD [-]')
    ax1.set_zlabel('Time Delta [s]')
    ax1.set_title(f'3D Sensitivity: Mass vs CD on {track.name}\n(Base: {base_time:.2f} s)')
    
    # All axes start at zero from the same corner (for Mass)
    ax1.set_xlim(left=0)
    ax1.view_init(elev=30, azim=225) 
    
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Lap Time Delta [s]')
    
    # Save figure 1
    save_path1 = os.path.join(os.path.dirname(__file__), "mass_cd_grid_3d.png")
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"3D Plot successfully saved to {save_path1}")
    
    # Figure 2: Heatmap
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    im = ax2.imshow(Z, extent=[mass_grid[0], mass_grid[-1], cd_grid[0], cd_grid[-1]], 
                    aspect='auto', cmap='plasma', interpolation='bilinear')
    
    ax2.set_xlabel('Mass Change [kg]')
    ax2.set_ylabel('CD [-]')
    ax2.xaxis.set_label_position('top') 
    ax2.xaxis.tick_top()
    ax2.set_title(f'Heatmap Sensitivity: Mass vs CD', pad=40)
    
    fig2.colorbar(im, ax=ax2, label='Lap Time Delta [s]')
    
    # Save figure 2
    save_path2 = os.path.join(os.path.dirname(__file__), "mass_cd_grid_heatmap.png")
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Heatmap Plot successfully saved to {save_path2}")
    
    # Show figures
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")

if __name__ == "__main__":
    main()
