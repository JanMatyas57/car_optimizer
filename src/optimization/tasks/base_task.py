from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class SolverPayload:
    kind: str
    data: object
    meta: dict = field(default_factory=dict)

class SimulationTask(ABC):
    #abstract base class from which all simulation tasks inherit
    
    @abstractmethod
    def execute(self, progress_callback) -> SolverPayload:
        """
        Method called by the worker, returns a SolverPayload with the results
        """
        pass