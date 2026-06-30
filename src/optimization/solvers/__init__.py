# src/optimization/solvers/__init__.py
from .base_solver import BaseSolver, SolverResults, InitConditions
from .mass_point import MassPointSolver
#from .single_track import SingleTrackSolver
from .quasi_steady import QuasiSteadySolver
from .mpcc import MPCC_Solver
from .strategy_maker import StrategyMaker
from .init_guess import calculate_initial_guess

import inspect
import pkgutil
import importlib
import os

def get_available_solvers():
    """
    Discover all solver classes that inherit from BaseSolver and have a DISPLAY_NAME.
    Returns a dictionary mapping DISPLAY_NAME to the class object.
    """
    solvers_dict = {}
    current_dir = os.path.dirname(__file__)
    
    for _, module_name, _ in pkgutil.iter_modules([current_dir]):
        if module_name in ['base_solver', 'single_track']: #hide single track for now protoze to saje pera
            continue
            
        try:
            module = importlib.import_module(f"src.optimization.solvers.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseSolver) and obj is not BaseSolver:
                    display_name = getattr(obj, "DISPLAY_NAME", None)
                    if display_name and display_name not in solvers_dict:
                        solvers_dict[display_name] = obj
        except Exception:
            pass
            
    return solvers_dict