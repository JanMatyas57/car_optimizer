import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .base_task import SimulationTask, SolverPayload

class StrategyOptTask(SimulationTask):
    def __init__(self, solver, car):
        self.solver = solver
        self.car = car

    def execute(self, progress_callback) -> SolverPayload:
        ef_min_res = self.solver.solve(0,0,objective="efficiency")
        ef_min = ef_min_res["ef_score"] if ef_min_res["success"] else self.solver.ef_guess(4)

        t_min_res = self.solver.solve(0,0,objective="time")
        t_min = t_min_res["total_time"] if t_min_res["success"] else self.solver.best_T_guess(0)

        regen_targets = np.linspace(0.15, 1.0, num=10, endpoint=True)
        delta_targets = np.linspace(0.0, 0.15, num=10, endpoint=True)

        e_opt_grid         = np.zeros((len(regen_targets), len(delta_targets)))
        soc_opt_grid       = np.zeros((len(regen_targets), len(delta_targets)))
        total_time_grid    = np.zeros((len(regen_targets), len(delta_targets)))

        current_step = 0
        total_steps = len(regen_targets) * len(delta_targets)

        for i, regen_target in enumerate(regen_targets):
            for j, delta_target in enumerate(delta_targets):
                current_step += 1
                progress_callback(current_step, total_steps)
                
                self.car.regen_eff = regen_target
                t_min_adj = t_min * (1 - delta_target)
              
                res = self.solver.solve(ef_min, t_min_adj, objective="points")
                
                if res and res.get("success", False):
                    e_opt_grid[i, j]      = np.sum(res["e_targets"])
                    soc_opt_grid[i, j]    = res["soc_start"]
                    total_time_grid[i, j] = res["total_time"]
                else:
                    e_opt_grid[i, j]      = np.nan
                    soc_opt_grid[i, j]    = np.nan
                    total_time_grid[i, j] = np.nan

        e_interpolant = RegularGridInterpolator((regen_targets, delta_targets), e_opt_grid, method='linear', bounds_error=False, fill_value=np.nan)
        soc_interpolant = RegularGridInterpolator((regen_targets, delta_targets), soc_opt_grid, method='linear', bounds_error=False, fill_value=np.nan)

        # ── persist the grid so it can be plotted standalone ──────────────
        save_dir = os.path.join("data", "tracks", "lapmaps")
        os.makedirs(save_dir, exist_ok=True)
        car_name = getattr(self.car, 'name', 'car').replace(' ', '_')
        save_path = os.path.join(save_dir, f"{car_name}_strategy_map.npz")
        np.savez_compressed(
            save_path,
            regen_targets=regen_targets,
            delta_targets=delta_targets,
            e_opt_grid=e_opt_grid,
            soc_opt_grid=soc_opt_grid,
            total_time_grid=total_time_grid,
            ef_min=np.array(ef_min),
            t_min=np.array(t_min),
            laps=np.array(self.solver.laps),
        )
        print(f"[StrategyOptTask] Grid saved → {save_path}")
        # ──────────────────────────────────────────────────────────────────

        return SolverPayload(
            kind="endurance_map",
            data=None,
            meta={
                "ef_min": ef_min, "t_min": t_min,
                "e_interpolant": e_interpolant, "soc_interpolant": soc_interpolant,  
                "grid_regen": regen_targets, "grid_delta": delta_targets,
                "z_min_e": np.nanmin(e_opt_grid), "z_max_e": np.nanmax(e_opt_grid),
                "car": self.car, "track": None
            }
        )

class DeploymentOptTask(SimulationTask):
    def __init__(self, solver, car):
        self.solver = solver
        self.car = car

    def execute(self, progress_callback) -> SolverPayload:
        ef_min_res = self.solver.solve(0,0,objective="efficiency")
        ef_min = ef_min_res["ef_score"] if ef_min_res["success"] else self.solver.ef_guess(4)

        t_min_res = self.solver.solve(0,0,objective="time")
        t_min = t_min_res["total_time"] if t_min_res["success"] else self.solver.best_T_guess(0)

        t_min_adj = t_min * (1 - self.car.performance_delta)

        res_opt = self.solver.solve(ef_min, t_min_adj, objective="points")
        res_full = self.solver.solve(ef_min, t_min_adj, objective="points", enforce_full_soc=True)

        # ── persist the full-SoC deployment so it can be plotted standalone ──
        if res_full and res_full.get("success", False):
            save_dir = os.path.join("data", "opt_deployments")
            os.makedirs(save_dir, exist_ok=True)
            car_name  = getattr(self.car, 'name', 'car').replace(' ', '_')
            regen_pct = int(round(self.car.regen_eff * 100))
            delta_pct = int(round(self.car.performance_delta * 100))
            save_path = os.path.join(
                save_dir,
                f"{car_name}_deployment_r{regen_pct}_d{delta_pct}.npz"
            )
            np.savez_compressed(
                save_path,
                lap_times  = np.array(res_full["lap_times"]),
                e_targets  = np.array(res_full["e_targets"]),
                soc_profile= np.array(res_full["soc_profile"]),
                total_time = np.array(res_full["total_time"]),
                points     = np.array(res_full["points"]),
                ef_score   = np.array(res_full["ef_score"]),
                laps       = np.array(self.solver.laps),
                regen_eff  = np.array(self.car.regen_eff),
                delta      = np.array(self.car.performance_delta),
            )
            print(f"[DeploymentOptTask] Full-SoC result saved → {save_path}")
        # ─────────────────────────────────────────────────────────────────────

        return SolverPayload(
            kind="endurance_deployment",
            data={"opt_soc": res_opt, "full_soc": res_full},
            meta={
                "ef_min": ef_min, "t_min": t_min,
                "regen_eff": self.car.regen_eff, "delta": self.car.performance_delta,
                "car": self.car, "track": None
            }
        )