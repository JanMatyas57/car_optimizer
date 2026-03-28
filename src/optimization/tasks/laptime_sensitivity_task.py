import time
import copy
from src.optimization.tasks.base_task import SimulationTask, SolverPayload
from src.optimization.solvers.base_solver import InitConditions

class LaptimeSensitivityTask(SimulationTask):
    def __init__(self, solver_class, car, tracks, ds, parameters):
        """
        solver_class: The class of the solver to instantiate (e.g. QuasiSteadySolver)
        car: The base car object
        tracks: A list of Track objects
        ds: The distance step
        parameters: A dictionary mapping parameter names to their step sizes e.g. {"weight": {"dx": 0.01}}
        """
        self.solver_class = solver_class
        self.base_car = car
        self.tracks = tracks
        self.ds = ds
        self.parameters = parameters

    def execute(self, progress_callback) -> SolverPayload:
        start_time = time.time()
        
        # Structure to hold results: {track_name: {param_name: sensitivity}}
        results = {}
        
        total_steps = len(self.tracks) * len(self.parameters) * 2 # 2 runs (plus and minus) per parameter per track
        current_step = 0

        for track in self.tracks:
            track_name = track.name
            results[track_name] = {}

            if track.speed is not None:
                init_cond = InitConditions(v=track.speed[0], T=0.0, E=0.0)
            else:
                init_cond = InitConditions(v=5, T=0.0, E=0.0)
            
            for param_name, param_info in self.parameters.items():
                dx_rel = param_info.get("dx", 0.01) # Default 1%
                
                # Get base value
                base_value = self._get_nested_attr(self.base_car, param_name)
                
                # If the base value is very close to 0, a relative step size doesn't work well.
                # In that case, we could use an absolute step, but for now we follow the 1% rule.
                # If base_value is 0, we can't do relative step. Let's add a small absolute epsilon if it's 0.
                if base_value == 0:
                    dx_abs = dx_rel  # just use dx_rel as absolute if base is 0
                else:
                    dx_abs = base_value * dx_rel
                    
                val_plus = base_value + dx_abs
                val_minus = base_value - dx_abs
                
                # Run +dx
                car_plus = copy.deepcopy(self.base_car)
                self._set_nested_attr(car_plus, param_name, val_plus)
                solver_plus = self.solver_class(car_plus, track)
                
                current_step += 1
                progress_callback(current_step, total_steps)
                
                res_plus = solver_plus.solve(ds=self.ds,init_cond=init_cond)
                t_plus = res_plus.lap_time if res_plus.success else float('inf')
                
                # Run -dx
                car_minus = copy.deepcopy(self.base_car)
                self._set_nested_attr(car_minus, param_name, val_minus)
                solver_minus = self.solver_class(car_minus, track)
                
                current_step += 1
                progress_callback(current_step, total_steps)

            
                res_minus = solver_minus.solve(ds=self.ds, init_cond=init_cond)
                t_minus = res_minus.lap_time if res_minus.success else float('inf')
                
                # Calculate sensitivity (Central Difference)
                # Sensitivity = dT / dP
                if t_plus != float('inf') and t_minus != float('inf'):
                    # if base_value is 0, denominator is 2 * dx_abs
                    delta_p = val_plus - val_minus
                    if delta_p != 0:
                        sensitivity = (t_plus - t_minus) / delta_p
                    else:
                        sensitivity = 0.0
                else:
                    # If any solver failed
                    sensitivity = float('nan')
                    
                # We store both the raw sensitivity and the relative sensitivity (dT / (dP/P)) %
                # Often it's more intuitive to see "Lap time change per 1% change in parameter"
                # which is simply (t_plus - t_minus) / 2
                time_diff = (t_plus - t_minus) / 2.0
                
                results[track_name][param_name] = {
                    "base_value": base_value,
                    "val_plus": val_plus,
                    "t_plus": t_plus,
                    "val_minus": val_minus,
                    "t_minus": t_minus,
                    "sensitivity": sensitivity,
                    "time_diff_per_dx": time_diff,
                    "dx_rel": dx_rel
                }

        end_time = time.time()
        print(f"Sensitivity analysis time elapsed: {end_time - start_time:.6f} s")

        return SolverPayload(
            kind="sensitivity_analysis",
            data=results,
            meta={"tracks": self.tracks, "parameters": self.parameters}
        )

    def _get_nested_attr(self, obj, attr_str):
        """Helper to get nested attributes like 'aerodynamics.cl'"""
        attrs = attr_str.split('.')
        for attr in attrs:
            if isinstance(obj, dict):
                obj = obj.get(attr)
            else:
                obj = getattr(obj, attr)
        return obj

    def _set_nested_attr(self, obj, attr_str, value):
        """Helper to set nested attributes like 'aerodynamics.cl'"""
        attrs = attr_str.split('.')
        for attr in attrs[:-1]:
            if isinstance(obj, dict):
                obj = obj.get(attr)
            else:
                obj = getattr(obj, attr)
                
        if isinstance(obj, dict):
            obj[attrs[-1]] = value
        else:
            setattr(obj, attrs[-1], value)
