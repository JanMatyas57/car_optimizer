import os
import sys
import ctypes
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from src.ui.main_window import MainWindow
from src.ui.results_window import ResultsWindow
from src.ui.endurance_strategy import LaptimeAppMainWindow
from src.ui.endurance_results import EnduranceResultsWindow
from src.ui.style_manager import StyleManager
from src.handlers.controller import SimulationController
from src.optimization.work_thread import SimulationWorker
from src.handlers.requests import StrategyOptRequest, DeploymentOptRequest
from src.ui.optimization_results_dialog import OptimizationResultsDialog
from src.ui.sensitivity_results_dialog import SensitivityResultsDialog

class SimulationApp:
    def __init__(self):
        
        # Create controller
        self.controller = SimulationController()
        
        # Create main window and pass the controller to it
        self.main_window = MainWindow(self.controller)
        
        # here we track references to windows
        self.open_windows = []

        # sginals from controller to update UI
        self.controller.simulationFinished.connect(self.handle_simulation_finished)
        self.controller.simulationError.connect(self.handle_error)
        
        # signals from main window
        self.main_window.openStrategyRequested.connect(self.open_strategy_picker)
        self.main_window.openCompareRequested.connect(self.open_compare_window)

    def start(self):
        self.main_window.show()

   
    # opening different windows based on user actions
  
    def open_strategy_picker(self):
        """Show a dialog that lists all laptime map .npz files, then open the strategy window."""
        from src.ui.laptime_map_picker import LaptimeMapPickerDialog

        lapmaps_dir = "data/tracks/lapmaps"
        dialog = LaptimeMapPickerDialog(lapmaps_dir, parent=self.main_window)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        filepath = dialog.selected_filepath()
        if not filepath:
            return

        self._load_strategy_window(filepath)

    def _load_strategy_window(self, filepath: str):
        """Instantiate LaptimeAppMainWindow for the given laptime map file."""
        try:
            win = LaptimeAppMainWindow(filepath)

            car_params = self.main_window._get_car_params()
            ds = self.main_window.input_ds.value()
            laps = self.main_window.car_params["laps"]

            # connect strategy optimization button
            win.optimization_requested.connect(
                lambda r, d, min_e, max_e, m_data: self.controller.handle_request(
                    StrategyOptRequest(
                        car_params=car_params,
                        ds=ds,
                        laps=laps,
                        regen_val=r,
                        delta_val=d,
                        min_e_target=min_e,
                        max_e_target=max_e,
                        map_data=m_data,
                        battery_model=self.main_window.combo_battery_model.currentData()
                    )
                )
            )

            # connect deployment button
            win.deployment_requested.connect(
                lambda r, d, min_e, max_e, m_data: self.controller.handle_request(
                    DeploymentOptRequest(
                        car_params=car_params,
                        ds=ds,
                        laps=laps,
                        regen_val=r,
                        delta_val=d,
                        min_e_target=min_e,
                        max_e_target=max_e,
                        map_data=m_data,
                        battery_model=self.main_window.combo_battery_model.currentData()
                    )
                )
            )

            # connect progress updates
            self.controller.simulationProgress.connect(win.loading_overlay.update_progress)
            self.controller.simulationFinished.connect(win.stop_loading)
            self.controller.simulationError.connect(win.stop_loading)

            self._register_and_show_window(win)
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Could not load laptime map:\n{e}")

    def open_compare_window(self):
        from src.ui.results_tab import CompareWindow
        win = CompareWindow()
        self._register_and_show_window(win)

    # handling the results

    def handle_simulation_finished(self, payload):
        kind = payload.kind
        data = payload.data
        meta = payload.meta
    
        if kind == "single_lap":
            win = ResultsWindow(data, meta['car'], meta['track'], meta['ds'])
            self._register_and_show_window(win)

        elif kind == "endurance":
            win = EnduranceResultsWindow(data, meta['car'], meta['track'], meta['ds'])
            self._register_and_show_window(win)
        
        elif kind in ("endurance_map", "endurance_deployment"):
    
            for win in self.open_windows:
                if isinstance(win, LaptimeAppMainWindow):
                    win.add_result_tab(payload)
                    break

        elif kind == "optimization":
            win = OptimizationResultsDialog(payload=data, parent=self.main_window)
            win.show()

        elif kind == "sensitivity_analysis":
            win = SensitivityResultsDialog(payload=payload, parent=self.main_window)
            win.show()
                
        elif kind == "track_generation":
            # route the smoothed track data back to main window to finish the DRS / Save process
            self.main_window.finish_track_generation(data)

    def handle_error(self, err_msg):
        
        QMessageBox.critical(self.main_window, "Simulation error", str(err_msg))

    def _register_and_show_window(self, window):
        
        self.open_windows.append(window)
    
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(lambda: self.open_windows.remove(window) if window in self.open_windows else None)
        window.show()

if __name__ == "__main__":
    try:
        myappid = 'optisim' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        pass # Not on windows

    app = QApplication(sys.argv)

    global_style = StyleManager.get_global_stylesheet()
    app.setStyleSheet(global_style)

    icon_path = os.path.join("data", "logo.png")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    
    sim_app = SimulationApp()
    sim_app.start()
    
    sys.exit(app.exec())



#fix the results tab