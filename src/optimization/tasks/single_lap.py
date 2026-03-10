import time
from src.optimization.solvers.base_solver import InitConditions
from .base_task import SimulationTask, SolverPayload

class SingleLapTask(SimulationTask):
    def __init__(self, solver, car, track, ds):
        self.solver = solver
        self.car = car
        self.track = track
        self.ds = ds

    def execute(self, progress_callback) -> SolverPayload:
        start = time.time()

        E = self.car.acp_capacity * (1 - self.car.soc_start)
        
        if self.track.speed is not None:
            init_cond = InitConditions(v=self.track.speed[0], T=0.0, E=E)
        else:
            init_cond = InitConditions(v=5, T=0.0, E=E)

       # init_guess = calculate_initial_guess(self.car, self.track, init_cond, ds=self.ds)
        
        #ggv = self.solver.qs_car.find_GGV(delta_max=0.2, beta_max=0.07, kappa_max=0.15, tol=0.1, N=9)
        res = self.solver.solve(init_cond, ds=self.ds, init_guess=None)
        
        end = time.time()
        print(f"Time elapsed: {end - start:.2f} s")
        
        if not res.success:
            raise Exception("Optimization failed for single lap.")
        
        return SolverPayload(
            kind="single_lap",
            data=res,
            meta={"ds": self.ds, "car": self.car, "track": self.track}
        )