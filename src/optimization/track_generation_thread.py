from PyQt6.QtCore import QThread, pyqtSignal
from src.core.trackGenerator import TrackGenerator


class TrackGenerationWorker(QThread):
    finished = pyqtSignal(bool, str) # success, message/path

    def __init__(self, data, lap_info, is_closed, save_path):
        super().__init__()
        self.data = data
        self.lap_info = lap_info
        self.is_closed = is_closed
        self.save_path = save_path

    def run(self):
        try:
        
            success,msg = TrackGenerator.process_and_save_lap(
                self.data, 
                self.lap_info, 
                self.is_closed, 
                self.save_path
            )
            
            if success:
                self.finished.emit(True, self.save_path)
            else:
                self.finished.emit(False, "Saving track failed: " + msg)
                
        except Exception as e:
            self.finished.emit(False, str(e))