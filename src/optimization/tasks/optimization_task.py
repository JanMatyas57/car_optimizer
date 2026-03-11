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
    def __init__(self, car_params: dict, ds: float, vehicle_name: str, tracks: list, bounds: dict):
        super().__init__()
        self.base_car_params = car_params
        self.ds = ds
        self.vehicle_name = vehicle_name
        self.bounds = bounds
       
        self.n_trials = 10
        
        self.loaded_tracks = [Track.load(t_name) for t_name in tracks]

    def execute(self, progress_callback=None):
        

        def objective(trial):
            
            current_params = self.base_car_params.copy()
            for param_name, (p_min, p_max) in self.bounds.items():
               
                val = trial.suggest_float(param_name, p_min, p_max)
                current_params[param_name] = val

          
            car = Car(**current_params)

            base_solver = QuasiSteadySolver(car, self.loaded_tracks[0])
            base_solver.generate_ggv()
            
            total_lap_time = 0.0
            
          
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
                
          
                total_lap_time += res.lap_time
                
            return total_lap_time

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