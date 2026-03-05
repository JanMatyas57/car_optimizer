import os
import glob
import numpy as np 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QGroupBox, QComboBox, 
                             QDoubleSpinBox, QPushButton, QSplitter, QCheckBox, 
                             QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QInputDialog)
from PyQt6.QtCore import Qt, pyqtSignal

# track for rendering
from src.core.track import Track
from src.core.trackGenerator import TrackGenerator
from src.optimization.track_generation_thread import TrackGenerationWorker
from src.handlers.requests import SingleLapRequest, EnduranceRequest, LapMapTestRequest
from src.ui.loader import LoadingOverlay
from src.utils.config_manager import ConfigManager

from PyQt6.QtGui import QTransform, QColor
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.updateGeometry(self)


class MainWindow(QMainWindow):
    openCompareRequested = pyqtSignal()
    openStrategyRequested = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # store defaults to use them when creating a new car or handling missing files
        self.default_params = {
            "acp_capacity": 6.0,
            "weight": 280.0,
            "cd": 1.83,
            "cd_drs": 1.17,
            "cl": 5.64,
            "cl_drs": 4.2,
            "cop": 0.53,
            "power_limit": 80.0,
            "regen_eff": 0.8,
            "laps": 22,
            "ds": 0.5,
            "model_type": "mass_point",
            "energy_limit": None,
            "drs_allowed": True,
            "soc_start": 1.0 # moved to simulation settings
        }
        
        # current active parameters
        self.car_params = self.default_params.copy()
        self.current_vehicle_name = None

        self.setWindowTitle("OptiSim")
        self.resize(1200, 900)

        # main layout setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # left control panel 
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setFixedWidth(320)
        
        self.init_track_section() 
        
        self.btn_config_car = QPushButton("⚙️ Configure car parameters")
        self.btn_config_car.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_config_car.setStyleSheet("background-color: #343a40; color: white; font-weight: bold; padding: 10px; margin-top: 10px;")
        self.btn_config_car.clicked.connect(self.open_car_config)
        self.control_layout.addWidget(self.btn_config_car)
        
        self.init_quick_settings()
        self.init_run_buttons()
        
        self.control_layout.addStretch()

        self.btn_create_track = QPushButton("Create track from data")
        self.btn_create_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create_track.setStyleSheet("""
            QPushButton { background-color: #6f42c1; color: white; font-weight: bold; padding: 10px; border-radius: 4px; margin-top: 10px; }
            QPushButton:hover { background-color: #5a32a3; }
        """)
        self.btn_create_track.clicked.connect(self.on_create_track_clicked)
        self.control_layout.addWidget(self.btn_create_track)

        self.btn_compare = QPushButton("Compare logs")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 10px; border-radius: 4px; margin-top: 10px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.btn_compare.clicked.connect(self.on_compare_clicked)
        self.control_layout.addWidget(self.btn_compare)

        # right panel (graphs)
        self.graph_panel = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_panel)
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.graph_layout.addWidget(self.mpl_canvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.graph_panel)
        splitter.setSizes([320, 880])
        self.main_layout.addWidget(splitter)

        self.loading_overlay = LoadingOverlay(self)
        
        self.controller.simulationStarted.connect(self.loading_overlay.show_loading)
        self.controller.simulationProgress.connect(self.loading_overlay.update_progress)
        self.controller.simulationFinished.connect(lambda _: self.loading_overlay.hide_loading())
        self.controller.simulationError.connect(lambda _: self.loading_overlay.hide_loading())

        # initial loads
        self.load_track_list()
        self.load_vehicle_list()

    def resizeEvent(self, event):
        self.loading_overlay.resize(self.size())
        super().resizeEvent(event)

    # ==========================================
    # vehicle profile management
    # ==========================================
    def load_vehicle_list(self, select_name=None):
        vehicles = ConfigManager.get_available_vehicles()
        
        # create default if directory is totally empty
        if not vehicles:
            ConfigManager.save_vehicle("default_car", self.default_params)
            vehicles = ["default_car"]
            
        self.combo_vehicle.blockSignals(True)
        self.combo_vehicle.clear()
        self.combo_vehicle.addItems(vehicles)
        
        if select_name and select_name in vehicles:
            self.combo_vehicle.setCurrentText(select_name)
        else:
            self.combo_vehicle.setCurrentIndex(0)
            
        self.combo_vehicle.blockSignals(False)
        
        # force load the selected vehicle
        self.current_vehicle_name = self.combo_vehicle.currentText()
        self.car_params = ConfigManager.load_vehicle(self.current_vehicle_name, self.default_params)
        self._update_ui_from_params()

    def on_vehicle_changed(self):
        # 1. save the outgoing vehicle first
        if self.current_vehicle_name:
            self._sync_params_from_ui()
            ConfigManager.save_vehicle(self.current_vehicle_name, self.car_params)
            
        # 2. load the incoming vehicle
        new_name = self.combo_vehicle.currentText()
        if new_name:
            self.current_vehicle_name = new_name
            self.car_params = ConfigManager.load_vehicle(new_name, self.default_params)
            self._update_ui_from_params()

    def on_new_vehicle_clicked(self):
        # ask user for a new name
        name, ok = QInputDialog.getText(self, "New vehicle", "Enter vehicle name:")
        if ok and name:
            name = name.strip()
            if name in ConfigManager.get_available_vehicles():
                QMessageBox.warning(self, "error", "Vehicle with this name already exists!")
                return
            
            # create new vehicle by duplicating current parameters
            self._sync_params_from_ui()
            ConfigManager.save_vehicle(name, self.car_params)
            self.load_vehicle_list(select_name=name)

    # ==========================================
    # ui initialization methods
    # ==========================================
    def open_car_config(self):
        dialog = CarConfigDialog(self.car_params, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.car_params = dialog.get_params()
            self._sync_params_from_ui()
            ConfigManager.save_vehicle(self.current_vehicle_name, self.car_params)
            print(f"vehicle '{self.current_vehicle_name}' updated and saved.")

    def init_track_section(self):
        group = QGroupBox("Track selection")
        layout = QFormLayout()
        self.combo_track = QComboBox()
        self.combo_track.currentIndexChanged.connect(self.on_track_changed) 
        layout.addRow("Track:", self.combo_track)
        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def init_quick_settings(self):
        group = QGroupBox("Simulation settings")
        layout = QFormLayout()

        # 1. vehicle model
        self.combo_model = QComboBox()
        self.combo_model.addItem("Mass point (Rk4)", "mass_point_rk4")
        self.combo_model.addItem("Single track", "single_track")
        layout.addRow("Vehicle model:", self.combo_model)

        # 2. vehicle profile selector
        veh_layout = QHBoxLayout()
        veh_layout.setContentsMargins(0, 0, 0, 0)
        self.combo_vehicle = QComboBox()
        self.combo_vehicle.currentIndexChanged.connect(self.on_vehicle_changed)
        
        self.btn_new_vehicle = QPushButton("+")
        self.btn_new_vehicle.setFixedWidth(30)
        self.btn_new_vehicle.clicked.connect(self.on_new_vehicle_clicked)
        self.btn_new_vehicle.setToolTip("create a new vehicle profile")
        
        veh_layout.addWidget(self.combo_vehicle)
        veh_layout.addWidget(self.btn_new_vehicle)
        layout.addRow("Vehicle:", veh_layout)

        # 3. sim distance step
        self.input_ds = QDoubleSpinBox()
        self.input_ds.setRange(0.1, 2.0)
        self.input_ds.setSingleStep(0.05)
        self.input_ds.setSuffix(" m")
        layout.addRow("Sim distance step:", self.input_ds)

        # 4. energy limit
        self.check_energy_limit = QCheckBox("Enable energy limit")
        self.input_limit_e = QDoubleSpinBox()
        self.input_limit_e.setRange(0.1, 100.0)
        self.input_limit_e.setSingleStep(0.01)
        self.input_limit_e.setSuffix(" kWh")
        self.check_energy_limit.toggled.connect(self.input_limit_e.setEnabled)
        layout.addRow(self.check_energy_limit, self.input_limit_e)
        
        # 5. SoC start
        self.input_soc = QDoubleSpinBox()
        self.input_soc.setRange(0.1, 1.0)
        self.input_soc.setSingleStep(0.05)
        self.input_soc.setDecimals(2)
        layout.addRow("SoC start:", self.input_soc)

        # 6. drs
        self.check_drs = QCheckBox("Enable drs")
        layout.addRow(self.check_drs)

        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def _update_ui_from_params(self):
        """pushes loaded dict values into the ui widgets safely."""
        idx = self.combo_model.findData(self.car_params.get("model_type", "mass_point"))
        if idx >= 0: self.combo_model.setCurrentIndex(idx)
        
        self.input_ds.setValue(self.car_params.get("ds", 0.5))
        
        saved_e_limit = self.car_params.get("energy_limit")
        self.check_energy_limit.setChecked(saved_e_limit is not None)
        self.input_limit_e.setValue(saved_e_limit if saved_e_limit is not None else 0.2)
        self.input_limit_e.setEnabled(self.check_energy_limit.isChecked())
        
        # update soc_start
        self.input_soc.setValue(self.car_params.get("soc_start", 1.0))
        
        self.check_drs.setChecked(self.car_params.get("drs_allowed", True))

    def init_run_buttons(self):
        group = QGroupBox("Run actions")
        layout = QVBoxLayout()

        self.btn_run = QPushButton("Run single lap")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px; margin-bottom: 5px;")
        self.btn_run.clicked.connect(self.on_run_single_clicked)
        layout.addWidget(self.btn_run)

        self.btn_endurance_strategy = QPushButton("Endurance strategy")
        self.btn_endurance_strategy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_endurance_strategy.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; padding: 10px; margin-bottom: 5px; border-radius: 4px;")
        self.btn_endurance_strategy.clicked.connect(self.on_endurance_strategy_clicked)
        layout.addWidget(self.btn_endurance_strategy)

        self.btn_acp_test = QPushButton("Generate laptime map")
        self.btn_acp_test.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_acp_test.setStyleSheet("background-color: #6c757d; color: white; padding: 10px; font-weight: bold; border-radius: 4px;")
        self.btn_acp_test.clicked.connect(self.on_lap_map_clicked)
        layout.addWidget(self.btn_acp_test)

        group.setLayout(layout)
        self.control_layout.addWidget(group)

    # ==========================================
    # core logic / sync
    # ==========================================
    def _sync_params_from_ui(self):
        self.car_params["model_type"] = self.combo_model.currentData()
        self.car_params["drs_allowed"] = self.check_drs.isChecked()
        self.car_params["energy_limit"] = self.input_limit_e.value() if self.check_energy_limit.isChecked() else None
        self.car_params["ds"] = self.input_ds.value()
        self.car_params["soc_start"] = self.input_soc.value() # grab soc from UI

    def _get_car_params(self):
        self._sync_params_from_ui()
        strict_params = self.car_params.copy()
        
        strict_params.pop("laps", None)
        strict_params.pop("ds", None) 
        strict_params.pop("mass", None)
        
        p_kw = strict_params.pop("power_limit", 80.0)
        strict_params["power_max"] = p_kw * 1000.0
        
        return strict_params

    def closeEvent(self, event):
        """ensures data is saved before the window is killed."""
        self._sync_params_from_ui()
        if self.current_vehicle_name:
            ConfigManager.save_vehicle(self.current_vehicle_name, self.car_params)
        super().closeEvent(event)

    # ==========================================
    # track loading
    # ==========================================
    def load_track_list(self):
        track_dir = "data/tracks"
        if not os.path.exists(track_dir):
            os.makedirs(track_dir)
            
        files = glob.glob(os.path.join(track_dir, "*npz"))
        track_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        
        current = self.combo_track.currentText()
        self.combo_track.blockSignals(True) 
        self.combo_track.clear()
        
        if track_names:
            self.combo_track.addItems(sorted(track_names))
            idx = self.combo_track.findText(current)
            if idx >= 0:
                self.combo_track.setCurrentIndex(idx)
        else:
            self.combo_track.addItem("no tracks found")
            self.combo_track.setEnabled(False)
            
        self.combo_track.blockSignals(False)
        self.on_track_changed()

    def on_track_changed(self):
        track_name = self.combo_track.currentText()
        if not track_name or track_name == "no tracks found":
            return
        try:
            self.current_track = Track.load(track_name)
            self.plot_track_map(self.current_track)
            
            dist_m = 22000
            laps = int(round(dist_m / self.current_track.length))
            self.car_params["laps"] = max(1, laps)
        except Exception as e:
            print(f"error loading track '{track_name}': {e}")

    def plot_track_map(self, track: Track):
        self.mpl_canvas.fig.clf()
        ax = self.mpl_canvas.fig.add_subplot(111)
        x, y = track.x, track.y
        alpha_deg = np.degrees(track.alpha) if hasattr(track, 'alpha') and track.alpha is not None else np.zeros_like(x)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=np.min(alpha_deg), vmax=np.max(alpha_deg))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(alpha_deg[:-1])
        lc.set_linewidth(3)
        ax.add_collection(lc)

        ax.set_xlim(x.min() - 20, x.max() + 20)
        ax.set_ylim(y.min() - 20, y.max() + 20)
        ax.set_aspect('equal')
        ax.plot(x[0], y[0], 'o', color='lime', markersize=8, markeredgecolor='black')
        ax.axis('off')
        self.mpl_canvas.draw()
  
    # ==========================================
    # request handlers
    # ==========================================
    def on_run_single_clicked(self):
        req = SingleLapRequest(
            car_params=self._get_car_params(),
            ds=self.input_ds.value(), 
            track_name=self.combo_track.currentText()
        )
        self.controller.handle_request(req)

    def on_lap_map_clicked(self):
        req = LapMapTestRequest(
            car_params=self._get_car_params(),
            ds=self.input_ds.value(), 
            track_name=self.combo_track.currentText(),
            laps=self.car_params["laps"] 
        )
        self.controller.handle_request(req)

    def on_endurance_strategy_clicked(self):
        track_name = self.combo_track.currentText()
        self.openStrategyRequested.emit(track_name)

    def on_compare_clicked(self):
        self.openCompareRequested.emit()

    # ==========================================
    # track generation wizard
    # ==========================================
    def on_create_track_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, "select telemetry log", "", "MAT Files (*.mat)")
        if not fname: return

        self.loading_overlay.show_loading("parsing log data...", indefinite=True)
        QApplication.processEvents()

        try:
            data = TrackGenerator.filter_data(fname)
        except Exception as e:
            self.loading_overlay.hide_loading()
            QMessageBox.critical(self, "error", f"failed to parse file:\n{e}")
            return
            
        self.loading_overlay.hide_loading()
        if data is None: return

        msg = QMessageBox(self)
        msg.setWindowTitle("select track mode")
        msg.setText("How do you want to define the track?")
        btn_closed = msg.addButton("closed track (endurance)", QMessageBox.ButtonRole.AcceptRole)
        btn_open = msg.addButton("open section (autocross)", QMessageBox.ButtonRole.RejectRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        
        if msg.clickedButton() == msg.button(QMessageBox.StandardButton.Cancel): return
        is_closed_mode = (msg.clickedButton() == btn_closed)

        lap_info = None
        if is_closed_mode:
            lap_info = TrackGenerator.run_lap_selector(data)
        else:
            lap_info = TrackGenerator.manual_lap_selector(data)

        if not lap_info: return

        default_name = f"track_{lap_info['time']:.2f}s.npz"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "save baked track", 
            os.path.join("data/tracks", default_name), 
            "NumPy Zip (*.npz)"
        )

        if save_path:
            self.loading_overlay.show_loading("smoothing track...", indefinite=True)
            self.track_worker = TrackGenerationWorker(data, lap_info, is_closed_mode, save_path)
            self.track_worker.finished.connect(self.on_track_generation_finished)
            self.track_worker.start()

    def on_track_generation_finished(self, success, message):
        self.loading_overlay.hide_loading()
        if success:
            QMessageBox.information(self, "success", f"track saved successfully!\n{message}")
            self.load_track_list()
            track_name = os.path.splitext(os.path.basename(message))[0]
            idx = self.combo_track.findText(track_name)
            if idx >= 0:
                self.combo_track.setCurrentIndex(idx)
        else:
            QMessageBox.critical(self, "error", f"track generation failed:\n{message}")


class CarConfigDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.current_params = current_params.copy()
        
        self.setWindowTitle("Configure car")
        self.setModal(True) 
        self.resize(300, 250)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        def create_spin(val, unit, min_val=0.0, max_val=10000.0, step=0.1):
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setSuffix(f" {unit}")
            spin.setValue(val if val is not None else min_val)
            return spin

        self.input_weight = create_spin(current_params.get("weight"), "kg", min_val=150, step=0.5)
        form_layout.addRow("Weight:", self.input_weight)
        
        self.input_cd = create_spin(current_params.get("cd"), "", step=0.01)
        form_layout.addRow("Drag coeff (Cd):", self.input_cd)

        self.input_cd_drs = create_spin(current_params.get("cd_drs"), "", step=0.01)
        form_layout.addRow("Drag coeff DRS (Cd drs):", self.input_cd_drs)
        
        self.input_cl = create_spin(current_params.get("cl"), "", step=0.01)
        form_layout.addRow("Downforce coeff (Cl):", self.input_cl)

        self.input_cl_drs = create_spin(current_params.get("cl_drs"), "", step=0.01)
        form_layout.addRow("Downforce coeff DRS (Cl drs):", self.input_cl_drs)

        self.input_cop = create_spin(current_params.get("cop"), "", step=0.01)
        form_layout.addRow("Center of pressure (CoP):", self.input_cop)
        
        self.input_power = create_spin(current_params.get("power_limit", 80.0), "kW", min_val=10.0, max_val=1000.0, step=1.0)
        form_layout.addRow("Power limit:", self.input_power)
        
        self.input_regen = create_spin(current_params.get("regen_eff", 0.8), "", min_val=0.0, max_val=1.0, step=0.05)
        form_layout.addRow("Regen factor:", self.input_regen)

        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_params(self):
        self.current_params.update({
            "weight": round(self.input_weight.value(),1),
            "cd": round(self.input_cd.value(),2),
            "cl": round(self.input_cl.value(),2),
            "cd_drs": round(self.input_cd_drs.value(),2),
            "cl_drs": round(self.input_cl_drs.value(),2),
            "cop": round(self.input_cop.value(),2),
            "power_limit": round(self.input_power.value(),1),
            "regen_eff": round(self.input_regen.value(),2)
        })
        return self.current_params