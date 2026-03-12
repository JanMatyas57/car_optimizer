import os
import time
import numpy as np
import json
from dataclasses import asdict
from scipy.interpolate import griddata
from src.optimization.solvers.base_solver import InitConditions
from .base_task import SimulationTask, SolverPayload

class LaptimeMapTask(SimulationTask):
    def __init__(self, solver, car, track, ds, laps):
        self.solver = solver
        self.car = car
        self.track = track
        self.ds = ds
        self.laps = laps

    def execute(self, progress_callback) -> SolverPayload:
        start = time.time()
        
        acp_targets = np.linspace(3.5, 9.0, num=10, endpoint=True)
        e_targets = np.round(acp_targets / self.laps, 2)
        regen_targets = np.linspace(0.0, 1.1, num=10, endpoint=True)
        soc_targets = np.linspace(0.1, 1.05, num=10, endpoint=True)

        total_steps = len(e_targets) * len(regen_targets) * len(soc_targets)
        results_log = []
        flat_lap_times = []
        current_step = 0
        last_slowest_lap = 40
        prev_res = None

        for soc in soc_targets:
            for e_target in e_targets:
                for regen in regen_targets:
                    current_step += 1
                    progress_callback(current_step, total_steps)
                    
                    if soc * self.car.acp_capacity < e_target:
                        flat_lap_times.append(np.nan)
                        continue
                    
                    self.car.regen_eff = regen
                    self.car.energy_limit = e_target
                    E_start = np.round(self.car.acp_capacity * (1-soc), 2)
                    init_cond = InitConditions(v=self.track.speed[0]-2, T=0.0, E=E_start)
                    
                    res = self.solver.solve(init_cond=init_cond, ds=self.ds, init_guess=prev_res)
                    
                    if res.success:
                        laptime = res.lap_time
                        flat_lap_times.append(laptime)
                        prev_res = {
                            'v': res.v,
                            'F_acc': res.f_acc 
                        }
                        if laptime > last_slowest_lap:
                            last_slowest_lap = laptime

                        res.meta_soc = soc
                        res.meta_energy = e_target
                        res.meta_regen = regen
                        res.meta_mass = self.car.weight
                        results_log.append(res)
                    else:
                        flat_lap_times.append(np.nan) 

        flat_lap_times = np.array(flat_lap_times, dtype=float)
        
        if np.isnan(flat_lap_times).any():
            S, E_grid, R = np.meshgrid(soc_targets, e_targets, regen_targets, indexing='ij')
            points = np.column_stack((S.ravel(), E_grid.ravel(), R.ravel()))
            values = flat_lap_times
            valid_mask = ~np.isnan(values)
            
            if valid_mask.any():
                valid_points = points[valid_mask]
                valid_values = values[valid_mask]
                filled_values = griddata(valid_points, valid_values, points, method='nearest')
                flat_lap_times = filled_values

        end = time.time()
        print(f"Time elapsed: {end - start:.6f} s")

        save_dir = os.path.join("data", "tracks", "lapmaps")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{self.track.name}_laptime_map.npz")

        # Extract car config as json string (selected parameters only)
        allowed_keys = ["name", "weight", "cd", "cl", "acp_capacity", "drs_allowed", "power_max"]
        car_dict = {k: getattr(self.car, k) for k in allowed_keys if hasattr(self.car, k)}
        car_json = json.dumps(car_dict)

        np.savez_compressed(
            file_path,
            soc_targets=np.array(soc_targets),
            e_targets=np.array(e_targets),
            regen_targets=np.array(regen_targets),
            flat_lap_times=np.array(flat_lap_times), 
            laps=np.array(self.laps),
            acp_capacity=np.array(self.car.acp_capacity),
            car_config=np.array([car_json])
        )

        return SolverPayload(
            kind="acp_test",
            data=results_log,
            meta={"laps": self.laps, "car": self.car, "track": self.track}
        )