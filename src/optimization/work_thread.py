from PyQt6.QtCore import QThread, pyqtSignal
from src.optimization.tasks.base_task import SimulationTask

class SimulationWorker(QThread):
    """Thread that executes simulation tasks"""
    finished = pyqtSignal(object) # returns SolverPayload
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)     

    def __init__(self, task: SimulationTask):
        super().__init__()
        self.task = task
    
    def run(self):
        try:
            payload = self.task.execute(progress_callback=self.progress.emit)
            self.finished.emit(payload)
        except Exception as e:
            self.error.emit(str(e))