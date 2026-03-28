# src/optimization/solvers/base_solver.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SolverResults:
    #mandatory
    success: bool
    lap_time: float
    t: np.ndarray
    v: np.ndarray #total velocity
    s: np.ndarray
    #inputs 
    f_acc: np.ndarray
    f_brake: np.ndarray
    recuperation:np.ndarray
    #model specific
    vx: np.ndarray      = field(default_factory=lambda: np.array([]))
    vy: np.ndarray      = field(default_factory=lambda: np.array([]))
    n: np.ndarray         = field(default_factory=lambda: np.array([]))  # Lateral deviation
    r: np.ndarray         = field(default_factory=lambda: np.array([]))
    xi: np.ndarray       = field(default_factory=lambda: np.array([]))  # Heading error
    kappa_car: np.ndarray = field(default_factory=lambda: np.array([]))  # Actual curvature
    delta: np.ndarray     = field(default_factory=lambda: np.array([]))  # Steering angle
    E: np.ndarray         = field(default_factory=lambda: np.array([]))  
    drs_usage: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_f: np.ndarray = field(default_factory=lambda: np.array([]))  # Slip angle front
    alpha_r: np.ndarray = field(default_factory=lambda: np.array([]))  # Slip angle rear
    SoC: np.ndarray = field(default_factory=lambda: np.array([]))

    # 2RC battery model outputs (None when no battery model is loaded)
    U1: np.ndarray = None          # RC1 voltage profile [V/cell]
    U2: np.ndarray = None          # RC2 voltage profile [V/cell]
    U_terminal: np.ndarray = None  # Pack terminal voltage [V]
    I_cell: np.ndarray = None      # Cell current profile [A/cell]

    # Sensitivities (s/unit)
    sens: dict = field(default_factory=dict)
    #
    mu_x: float = 0.0
    mu_y: float = 0.0
    regen_eff: float = 0.0
    ap_id: int = 0
@dataclass
class InitConditions:
    v: float = 0.0  
    T: float = 0.0   
    E: float = 0.0 
    n: float = 0.0
    xi: float = 0.0
    E_target: float = 0.0
    target_time: float = None

class BaseSolver(ABC):
    
    def __init__(self, car, track, laps=1):
        self.car = car
        self.track = track
        self.laps = laps

    @abstractmethod
    def solve(self, init_cond: InitConditions, init_guess=None, ds=0.5) -> SolverResults:
        """
        Abstract method for all solvers
        """
        pass