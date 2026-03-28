import optuna
from dataclasses import dataclass
from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers.quasi_steady import QuasiSteadySolver
from src.optimization.solvers.base_solver import InitConditions
from src.optimization.tasks.base_task import SimulationTask, SolverPayload


@dataclass
class OptimizationPayload:
    best_params: dict
    best_lap_time: float
    study: optuna.Study
    base_vehicle_name: str
    base_car_params: dict  
    tracks: list           

class OptimizationTask(SimulationTask):
    def __init__(self, car_params: dict, ds: float, vehicle_name: str, tracks: list, track_weights: dict, bounds: dict, trials: int = 100):
        super().__init__()
        self.base_car_params = car_params
        self.ds = ds
        self.vehicle_name = vehicle_name
        self.track_weights = track_weights
        self.bounds = bounds
       
        self.n_trials = trials
        
        self.loaded_tracks = [Track.load(t_name) for t_name in tracks]

    def _run_baseline(self):
        """Run the base car on all tracks to get reference lap times for normalization."""
        base_car = Car(**self.base_car_params)
        base_solver = QuasiSteadySolver(base_car, self.loaded_tracks[0])
        base_solver.generate_ggv()

        baseline_times = {}
        for track in self.loaded_tracks:
            if track.speed is not None:
                init_cond = InitConditions(v=track.speed[0], T=0.0, E=0.0)
            else:
                init_cond = InitConditions(v=5, T=0.0, E=0.0)

            solver = QuasiSteadySolver(base_car, track)
            solver.set_ggv(
                df=base_solver.df,
                slices_ax_pos=base_solver.slices_ax_pos,
                slices_ax_neg=base_solver.slices_ax_neg,
                slices_ay_throttle=base_solver.slices_ay_throttle,
                slices_ay_brake=base_solver.slices_ay_brake
            )
            res = solver.solve(ds=self.ds, init_cond=init_cond)
            baseline_times[track.name] = res.lap_time if res.success else 1.0
            print(f"[Baseline] {track.name}: {baseline_times[track.name]:.3f} s")

        return baseline_times

    def execute(self, progress_callback=None):

        # --- Baseline pre-run ---
        # Simulate the base car on all tracks once to get reference times.
        # These are used to normalize each track's contribution to the objective,
        # so longer tracks don't dominate shorter ones.
        print("[OptimizationTask] Running baseline to get reference lap times...")
        baseline_times = self._run_baseline()

        # Normalize weights so they sum to 1
        total_weight = sum(self.track_weights.get(t.name, 1.0) for t in self.loaded_tracks)

        def objective(trial):
            
            current_params = self.base_car_params.copy()
            for param_name, (p_min, p_max) in self.bounds.items():
                val = trial.suggest_float(param_name, p_min, p_max)
                current_params[param_name] = val

            car = Car(**current_params)

            base_solver = QuasiSteadySolver(car, self.loaded_tracks[0])
            base_solver.generate_ggv()
            
            # Weighted normalized objective: sum of w_i * (T_i / T_i_ref)
            objective_value = 0.0
            
            for track in self.loaded_tracks:
                if track.speed is not None:
                    init_cond = InitConditions(v=track.speed[0], T=0.0, E=0.0)
                else:
                    init_cond = InitConditions(v=5, T=0.0, E=0.0)

                solver = QuasiSteadySolver(car, track)
                solver.set_ggv(
                    df=base_solver.df,
                    slices_ax_pos=base_solver.slices_ax_pos,
                    slices_ax_neg=base_solver.slices_ax_neg,
                    slices_ay_throttle=base_solver.slices_ay_throttle,
                    slices_ay_brake=base_solver.slices_ay_brake
                )
                
                res = solver.solve(ds=self.ds, init_cond=init_cond)

                ref_time = baseline_times.get(track.name, 1.0)
                w = self.track_weights.get(track.name, 1.0) / total_weight
                objective_value += w * (res.lap_time / ref_time)
                
            return objective_value

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        
        
        def optuna_callback(study, trial):
            if progress_callback:
                
                progress_callback(len(study.trials), self.n_trials)


        study.optimize(objective, n_trials=self.n_trials, callbacks=[optuna_callback])
        
        
        results = OptimizationPayload(
            best_params=study.best_params,
            best_lap_time=study.best_value,
            study=study,
            base_vehicle_name=self.vehicle_name,
            base_car_params=self.base_car_params,        
            tracks=[t.name for t in self.loaded_tracks]
        )

        return SolverPayload(
            kind="optimization",
            data=results,
            meta={
                    "car": self.vehicle_name,
                    "tracks": [t.name for t in self.loaded_tracks],   
                }
        )