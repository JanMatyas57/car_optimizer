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

        e_opt_grid = np.zeros((len(regen_targets), len(delta_targets)))
        soc_opt_grid = np.zeros((len(regen_targets), len(delta_targets)))

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
                    e_opt_grid[i, j] = np.sum(res["e_targets"])
                    soc_opt_grid[i, j] = res["soc_start"]
                else:
                    e_opt_grid[i, j] = np.nan
                    soc_opt_grid[i, j] = np.nan

        e_interpolant = RegularGridInterpolator((regen_targets, delta_targets), e_opt_grid, method='linear', bounds_error=False, fill_value=np.nan)
        soc_interpolant = RegularGridInterpolator((regen_targets, delta_targets), soc_opt_grid, method='linear', bounds_error=False, fill_value=np.nan)
     
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

        return SolverPayload(
            kind="endurance_deployment",
            data={"opt_soc": res_opt, "full_soc": res_full},
            meta={
                "ef_min": ef_min, "t_min": t_min,
                "regen_eff": self.car.regen_eff, "delta": self.car.performance_delta,
                "car": self.car, "track": None
            }
        )