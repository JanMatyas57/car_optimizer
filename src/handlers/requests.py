from dataclasses import dataclass

# base request containing parameters shared across all simulations
@dataclass
class BaseRequest:
    car_params: dict
    ds: float

# specific request types extending the base request
@dataclass
class SingleLapRequest(BaseRequest):
    track_name: str

@dataclass
class EnduranceRequest(BaseRequest):
    track_name: str
    laps: int

@dataclass
class LapMapTestRequest(BaseRequest):
    track_name: str
    laps: int

@dataclass
class StrategyOptRequest(BaseRequest):
    laps: int
    regen_val: float
    delta_val: float
    min_e_target: float
    max_e_target: float
    map_data: dict

@dataclass
class DeploymentOptRequest(StrategyOptRequest):
    # inherits all fields from strategyoptrequest, used just for routing
    pass

@dataclass
class TrackGenerationRequest(BaseRequest):
    # request for track generation and smoothing
    data: dict
    lap_infos: list
    is_closed: bool
    r_penalty: float
    w_penalty: float


@dataclass
class OptimizationRequest(BaseRequest):
    vehicle_name: str
    tracks: list      
    track_weights: dict
    bounds: dict      
    trials: int

@dataclass
class SensitivityRequest(BaseRequest):
    track_names: list
    parameters: dict