from PyQt6.QtCore import QObject, pyqtSignal
from src.core.car import Car
from src.core.track import Track
from src.optimization.solvers import MassPointSolver, StrategyMaker, QuasiSteadySolver
from src.optimization.solvers import InitConditions, calculate_initial_guess
from src.optimization.work_thread import SimulationWorker

# import tasks
from src.optimization.tasks import SingleLapTask, StrategyOptTask, DeploymentOptTask, TrackGenerationTask, OptimizationTask, LaptimeSensitivityTask


# import requests
from src.handlers.requests import (
    SingleLapRequest, EnduranceRequest, LapMapTestRequest, 
    StrategyOptRequest, DeploymentOptRequest, TrackGenerationRequest, OptimizationRequest, SensitivityRequest
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

    def _create_solver(self, car: Car, track: Track, laps: int,
                        simulate_full_battery: bool = False,
                        soc_power_mode: str = "lookup_table",
                        battery_model: str = "ep85"):
        import src.optimization.solvers as solvers
        from src.optimization.solvers import get_available_solvers
        
        available = get_available_solvers()
        model_type = car.model_type

        solver_class = available.get(model_type)
        if not solver_class:
            solver_class = available.get("Mass point (Rk4)")
            
        solver = solver_class(car, track)
        
        # Apply configurations if the solver supports them
        if hasattr(solver, 'simulate_full_battery'):
            solver.simulate_full_battery = simulate_full_battery
        if hasattr(solver, 'soc_power_mode'):
            solver.soc_power_mode = soc_power_mode
            
        return solver


    def handle_request(self, req):
        # single entry point for all incoming simulation requests
        self.simulationStarted.emit()

        try:
            # build the core car object shared across all requests
            car = Car(**req.car_params)

            # load battery model dynamically if present on the request
            battery_model = getattr(req, "battery_model", None)
            if battery_model:
                import os
                battery_dir = os.path.normpath(os.path.join(
                    os.path.dirname(__file__),
                    "..", "core", "vehicles", "battery"
                ))
                npz_name = f"{battery_model}.npz"
                npz_path = os.path.join(battery_dir, npz_name)
                if os.path.isfile(npz_path):
                    car.load_battery(npz_path, battery_type=battery_model)
                else:
                    print(f"[Controller] Battery file not found: {npz_path}")

            task = None

            # route logic based on the specific type of request
            if isinstance(req, SingleLapRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(
                    car, track, laps=1,
                    simulate_full_battery=req.simulate_full_battery,
                    soc_power_mode=req.soc_power_mode,
                    battery_model=req.battery_model
                )
                task = SingleLapTask(solver, car, track, req.ds, use_warm_start=req.use_warm_start)
                
            elif isinstance(req, EnduranceRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(car, track, laps=req.laps)
                # task = EnduranceTask(solver, car, track, req.ds, req.laps)
                raise NotImplementedError("endurance task not implemented yet")

            elif isinstance(req, LapMapTestRequest):
                track = Track.load(req.track_name)
                solver = self._create_solver(
                    car, track, laps=req.laps,
                    simulate_full_battery=req.simulate_full_battery,
                    soc_power_mode=req.soc_power_mode,
                    battery_model=req.battery_model
                )
                task = LaptimeMapTask(solver, car, track, req.ds, req.laps)


            elif isinstance(req, TrackGenerationRequest):
                # track generation task doesnt use the car object
                task = TrackGenerationTask(
                    req.data, req.lap_infos, req.is_closed, req.r_penalty, req.w_penalty
                )
            
            elif isinstance(req, OptimizationRequest):
                
                task = OptimizationTask(
                    car_params=req.car_params,
                    ds=req.ds,
                    vehicle_name=req.vehicle_name,
                    tracks=req.tracks,
                    track_weights=req.track_weights,
                    bounds=req.bounds,
                    trials=req.trials
                )

            elif isinstance(req, SensitivityRequest):
                tracks = [Track.load(t_name) for t_name in req.track_names]
                # Pass the solver class, not the instance, because the task needs to instantiate multiple solvers 
                # for different variations of the car
                task = LaptimeSensitivityTask(
                    solver_class=QuasiSteadySolver,
                    car=car,
                    tracks=tracks,
                    ds=req.ds,
                    parameters=req.parameters
                )

            elif isinstance(req, (StrategyOptRequest, DeploymentOptRequest)):
                # update car with strategy specific modifiers
                car.regen_eff = req.regen_val
                car.performance_delta = req.delta_val
                
                laptime_interpolant = req.map_data['interpolant']
                acp_capacity = req.map_data.get('acp_capacity', None)
                solver = StrategyMaker(
                    car, laptime_interpolant, req.laps, 
                    req.min_e_target + 0.01, req.max_e_target - 0.01,
                    acp_capacity=acp_capacity
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