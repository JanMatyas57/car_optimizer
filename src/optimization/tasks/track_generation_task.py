from .base_task import SimulationTask, SolverPayload
from src.core.trackGenerator import generate_smoothed_track

class TrackGenerationTask(SimulationTask):
    def __init__(self, data, lap_infos, is_closed, r_penalty, w_penalty):
        self.data = data
        self.lap_infos = lap_infos
        self.is_closed = is_closed
        self.r_penalty = r_penalty
        self.w_penalty = w_penalty

    def execute(self, progress_callback) -> SolverPayload:
        # emit start progress (casadi takes over the terminal, so we just use an indefinite loader in ui)
        progress_callback(0, 1)
        
        result_data = generate_smoothed_track(
            self.data, 
            self.lap_infos, 
            self.is_closed, 
            self.r_penalty,
            self.w_penalty
        )
        
        if result_data is None:
            raise RuntimeError("track smoothing failed to converge.")
            
        progress_callback(1, 1)
        
        return SolverPayload(
            kind="track_generation",
            data=result_data,
            meta={}
        )