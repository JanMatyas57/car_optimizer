from PyQt6.QtCore import QObject, pyqtSignal
from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers import MassPointSolver, StrategyMaker
from src.optimization.solvers import InitConditions, calculate_initial_guess
from src.optimization.work_thread import SimulationWorker

# import tasks
from src.optimization.tasks import SingleLapTask, LaptimeMapTask, StrategyOptTask, DeploymentOptTask, TrackGenerationTask


# import requests
from src.handlers.requests import (
    SingleLapRequest, EnduranceRequest, LapMapTestRequest, 
    StrategyOptRequest, DeploymentOptRequest, TrackGenerationRequest
)

class SimulationController(QObject):
    # central signals for ui updates
    simulationStarted = pyqtSignal()
    simulationProgress = pyqtSignal(int, int)
    simulationFinished = pyqtSignal(object) 
    simulationError = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.active_worker = None

    def _create_solver(self, car: Car, track: Track, laps: int):
       
        if car.model_type == "mass_point_rk4":
            return MassPointSolver(car, track)
        else:
            return MassPointSolver(car, track)

    def handle_request(self, req):
        # single entry point for all incoming simulation requests
        self.simulationStarted.emit()

        try:
            # build the core car object shared across all requests
            car = Car(**req.car_params)
            task = None

            # route logic based on the specific type of request
            if isinstance(req, SingleLapRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(car, track, laps=1)
                task = SingleLapTask(solver, car, track, req.ds)
                
            elif isinstance(req, EnduranceRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(car, track, laps=req.laps)
                # task = EnduranceTask(solver, car, track, req.ds, req.laps)
                raise NotImplementedError("endurance task not implemented yet")

            elif isinstance(req, LapMapTestRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(car, track, laps=req.laps)
                task = LaptimeMapTask(solver, car, track, req.ds, req.laps)

            elif isinstance(req, TrackGenerationRequest):
                # track generation task doesnt use the car object
                task = TrackGenerationTask(
                    req.data, req.lap_infos, req.is_closed, req.r_penalty, req.w_penalty
                )

            elif isinstance(req, (StrategyOptRequest, DeploymentOptRequest)):
                # update car with strategy specific modifiers
                car.regen_eff = req.regen_val
                car.performance_delta = req.delta_val
                
                laptime_interpolant = req.map_data['interpolant']
                solver = StrategyMaker(
                    car, laptime_interpolant, req.laps, 
                    req.min_e_target + 0.01, req.max_e_target - 0.01
                )
                
                if type(req) is StrategyOptRequest:
                    task = StrategyOptTask(solver, car)
                elif type(req) is DeploymentOptRequest:
                    task = DeploymentOptTask(solver, car)
            else:
                # catch unknown requests
                raise TypeError(f"unknown request type: {type(req).__name__}")

            # delegate the task to the worker thread
            self.active_worker = SimulationWorker(task)
            
            # connect signals
            self.active_worker.progress.connect(self.simulationProgress.emit)
            self.active_worker.error.connect(self._on_worker_error)
            self.active_worker.finished.connect(self.simulationFinished.emit)
            self.active_worker.finished.connect(self._cleanup_worker)

            # start computation
            self.active_worker.start()

        except Exception as e:
            self.simulationError.emit(f"failed to initialize task: {str(e)}")

    def _on_worker_error(self, err_msg):
        # emit error and clean up
        self.simulationError.emit(err_msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        # safely delete the c++ thread object
        if self.active_worker:
            self.active_worker.deleteLater()
            self.active_worker = None