import os
import glob
import numpy as np 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QGroupBox, QComboBox, 
                             QDoubleSpinBox, QPushButton, QSplitter, QCheckBox, 
                             QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QInputDialog,QScrollArea,QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal

# track for rendering
from src.core.track import Track
from src.core.trackGenerator import TrackGenerator
from src.handlers.requests import SingleLapRequest, EnduranceRequest, LapMapTestRequest
from src.ui.loader import LoadingOverlay
from src.ui.track_generator_dialog import TrackGeneratorDialog
from src.utils.config_manager import ConfigManager
from src.ui.track_tools import DRSEditorDialog
import os
import numpy as np

from PyQt6.QtGui import QGuiApplication, QTransform, QColor, QFont
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#ffffff')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#ffffff')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.updateGeometry(self)


class MainWindow(QMainWindow):
    openCompareRequested = pyqtSignal()
    openStrategyRequested = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.default_params = {
            "acp_capacity": 6.0,  #
            "weight": 280.0,
            "wb": 1.53,           # wheelbase
            "tw": 1.25,           # track width
            "wd": 0.53,           # weight distribution
            "cogz": 0.3,         # center of gravity 
            "i": 11.46,            # gear ratio
            "M_max": 30.0,        # max torque
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
            "soc_start": 0.67 
        }
        
        # current active parameters
        self.car_params = self.default_params.copy()
        self.current_vehicle_name = None

        self.setWindowTitle("OptiSim")
        
    
        screen = QGuiApplication.primaryScreen()
        available_geometry = screen.availableGeometry() 

        screen_width = available_geometry.width()
        screen_height = available_geometry.height()

     
        target_width = int(screen_width * 0.85)
        target_height = int(screen_height * 0.90)

     
        target_width = min(target_width, 1600)
        target_height = min(target_height, 1200)

        self.resize(target_width, target_height)

        if target_width >= screen_width * 0.95 or target_height >= screen_height * 0.95:
            self.showMaximized()

        # main layout
        self.central_widget = QWidget()
        self.central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # left panel
        self.control_panel = QWidget()
        self.control_panel.setObjectName("ControlPanel")
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(0, 0, 10, 0) 
        self.control_layout.setSpacing(15)
        
    
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setObjectName("LeftScrollArea")
        self.left_scroll_area.setWidgetResizable(True) 
        self.left_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.left_scroll_area.setWidget(self.control_panel)

        self.left_scroll_area.setFixedWidth(360) 
        
        self.init_track_section() 
        
        self.btn_config_car = QPushButton("⚙️ Configure car parameters")
        self.btn_config_car.setObjectName("BtnConfig")
        self.btn_config_car.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_config_car.clicked.connect(self.open_car_config)
        self.control_layout.addWidget(self.btn_config_car)
        
        self.init_quick_settings()
        self.init_run_buttons()
        
        self.control_layout.addStretch()

        self.btn_create_track = QPushButton("Create track from data")
        self.btn_create_track.setObjectName("BtnCreateTrack")
        self.btn_create_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create_track.clicked.connect(self.on_create_track_clicked)
        self.control_layout.addWidget(self.btn_create_track)

        self.btn_compare = QPushButton("Compare logs")
        self.btn_compare.setObjectName("BtnCompare")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.clicked.connect(self.on_compare_clicked)
        self.control_layout.addWidget(self.btn_compare)

        # right panel
        self.graph_panel = QWidget()
        self.graph_panel.setObjectName("GraphPanel")
        self.graph_layout = QVBoxLayout(self.graph_panel)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
     
        self.canvas_container = QWidget()
        self.canvas_container.setObjectName("CanvasContainer")
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.addWidget(self.mpl_canvas)
        
        self.graph_layout.addWidget(self.canvas_container)

        # spllitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setHandleWidth(8)
    
        splitter.addWidget(self.left_scroll_area) 
        splitter.addWidget(self.graph_panel)
        splitter.setSizes([360, 940])
        self.main_layout.addWidget(splitter)

        #loader and styles
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
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(10)
        self.combo_track = QComboBox()
        self.combo_track.currentIndexChanged.connect(self.on_track_changed) 
        layout.addRow("Track:", self.combo_track)
        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def init_quick_settings(self):
        group = QGroupBox("Simulation settings")
        layout = QFormLayout()
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(12)

        # 1. vehicle model
        self.combo_model = QComboBox()
        self.combo_model.addItem("Mass point (Rk4)", "mass_point_rk4")
        self.combo_model.addItem("Quasi Steady", "quasi_steady")
        layout.addRow("Vehicle model:", self.combo_model)

        # 2. vehicle profile selector
        veh_layout = QHBoxLayout()
        veh_layout.setContentsMargins(0, 0, 0, 0)
        veh_layout.setSpacing(8)
        self.combo_vehicle = QComboBox()
        self.combo_vehicle.currentIndexChanged.connect(self.on_vehicle_changed)
        
        self.btn_new_vehicle = QPushButton("+")
        self.btn_new_vehicle.setObjectName("BtnTool")
        self.btn_new_vehicle.setFixedWidth(32)
        self.btn_new_vehicle.setFixedHeight(32)
        self.btn_new_vehicle.clicked.connect(self.on_new_vehicle_clicked)
        self.btn_new_vehicle.setToolTip("Create a new vehicle profile")
        
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
        self.input_soc.setRange(10, 100)
        self.input_soc.setSingleStep(1)
        self.input_soc.setDecimals(2)     
        self.input_soc.setSuffix(" %")
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
        soc_val = self.car_params.get("soc_start", 0.67)
        self.input_soc.setValue(soc_val * 100.0)
        
        self.check_drs.setChecked(self.car_params.get("drs_allowed", True))

    def init_run_buttons(self):
        group = QGroupBox("Run actions")
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(10)

        self.btn_run = QPushButton("Run single lap")
        self.btn_run.setObjectName("BtnRunSingle")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self.on_run_single_clicked)
        layout.addWidget(self.btn_run)

        self.btn_endurance_strategy = QPushButton("Endurance strategy")
        self.btn_endurance_strategy.setObjectName("BtnEndurance")
        self.btn_endurance_strategy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_endurance_strategy.clicked.connect(self.on_endurance_strategy_clicked)
        layout.addWidget(self.btn_endurance_strategy)

        self.btn_acp_test = QPushButton("Generate laptime map")
        self.btn_acp_test.setObjectName("BtnLapMap")
        self.btn_acp_test.setCursor(Qt.CursorShape.PointingHandCursor)
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
        self.car_params["soc_start"] = self.input_soc.value()/100# grab soc from UI
        self.car_params["name"] = self.current_vehicle_name

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
        ax.set_facecolor('#ffffff')
        
        x, y = track.x, track.y
        
        alpha_deg = np.degrees(track.alpha) if hasattr(track, 'alpha') and track.alpha is not None else np.zeros_like(x)

      
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

  
        norm = Normalize(vmin=np.min(alpha_deg), vmax=np.max(alpha_deg))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(alpha_deg[:-1])
        lc.set_linewidth(3.5) 
        
        line = ax.add_collection(lc)

       
        ax.set_xlim(x.min() - 20, x.max() + 20)
        ax.set_ylim(y.min() - 20, y.max() + 20)
        ax.set_aspect('equal')
        
      
        ax.plot(x[0], y[0], 'o', color='#10b981', markersize=8, markeredgecolor='black', zorder=5)
        
       
        self.mpl_canvas.fig.subplots_adjust(right=0.85)
        
        cax = self.mpl_canvas.fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cb = self.mpl_canvas.fig.colorbar(line, cax=cax)
        

        cb.outline.set_visible(False) # Skrytí rámečku legendy
        cb.set_label('Elevation [deg]', color='#64748b', size=10, labelpad=10)
        cb.ax.yaxis.set_tick_params(colors='#64748b', labelsize=9)
        
        # Vypnutí os mapy
        ax.axis('off')
        
        # Překreslení
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

    def on_create_track_clicked(self):
        from src.ui.track_generator_dialog import TrackGeneratorDialog
        dialog = TrackGeneratorDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  

        fname = dialog.selected_file
        is_closed_mode = dialog.radio_closed.isChecked()
        r_penalty = dialog.spin_r.value()
        w_penalty = dialog.spin_w.value()

        if not fname:
            return

        self.loading_overlay.show_loading("parsing log data...", indefinite=True)
        QApplication.processEvents()

        try:
            from src.handlers.telemetry_parser import filter_data
            data = filter_data(fname)
        except Exception as e:
            self.loading_overlay.hide_loading()
            QMessageBox.critical(self, "error", f"failed to parse file:\n{e}")
            return
            
        self.loading_overlay.hide_loading()
        if data is None: 
            return

        from src.ui.track_tools import run_lap_selector, manual_lap_selector
        
        # pass self as parent so dialogs center correctly over main window
        if is_closed_mode:
            lap_infos = run_lap_selector(data, self)
        else:
            lap_infos = manual_lap_selector(data, self)

        if not lap_infos:
            return 

        self.loading_overlay.show_loading("smoothing track (casadi)...", indefinite=True)
        from src.handlers.requests import TrackGenerationRequest
        
        req = TrackGenerationRequest(
            car_params=self._get_car_params(), 
            ds=self.input_ds.value(),          
            data=data,
            lap_infos=lap_infos,
            is_closed=is_closed_mode,
            r_penalty=r_penalty,
            w_penalty=w_penalty
        )
        self.controller.handle_request(req)

    def finish_track_generation(self, result_data):
        
    
        # show native pyqtgraph dialog instead of MapDRSEditor class
        dialog = DRSEditorDialog(result_data['x'], result_data['y'], result_data['s'], self)
        dialog.exec()
        result_data['drs'] = dialog.get_mask()

        avg_time = result_data.get('lap_time', 0.0)
        default_name = f"track_{avg_time:.2f}s.npz"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "save baked track", 
            os.path.join("data/tracks", default_name), 
            "NumPy Zip (*.npz)"
        )

        if save_path:
            
            np.savez(save_path, **result_data)
            QMessageBox.information(self, "success", f"track saved successfully!\n{save_path}")
            
            self.load_track_list()
            track_name = os.path.splitext(os.path.basename(save_path))[0]
            idx = self.combo_track.findText(track_name)
            if idx >= 0:
                self.combo_track.setCurrentIndex(idx)


class CarConfigDialog(QDialog):
    def __init__(self, car_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vehicle configuration")
        self.setMinimumWidth(500)
        self.setMinimumHeight(700)
        
        # Profile parameters from UI 
        self.params = car_params if car_params else {}
        
        self._init_ui()
        self._populate_data()

    def _init_ui(self):
        # Main dialog layout 
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Scroll Area - fix for black background 
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        # Force transparent viewport to see the container background 
        self.scroll_area.viewport().setStyleSheet("background: transparent;")
        
        # 2. Content Container 
        self.container = QWidget()
        # White background matches your QTabWidget > QWidget rule 
        self.container.setStyleSheet("background-color: #ffffff;")
        
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(20, 20, 20, 20)
        self.container_layout.setSpacing(15)

        # Helper to style groupboxes with blue border 
        blue_border_style = "QGroupBox { border: 1px solid #3b82f6; }"

        # --- Section: General ---
        group_general = QGroupBox("General")
        group_general.setStyleSheet(blue_border_style)
        form_general = QFormLayout()
        self.edit_name = QLineEdit()
        form_general.addRow("Profile name:", self.edit_name)
        group_general.setLayout(form_general)
        self.container_layout.addWidget(group_general)

        # --- Section: Chassis & Geometry ---
        group_chassis = QGroupBox("Chassis")
        group_chassis.setStyleSheet(blue_border_style)
        form_chassis = QFormLayout()
        self.spin_weight = self._create_double_spinbox(0, 1000, 280.0, decimals=1)
        self.spin_wb = self._create_double_spinbox(0.1, 5.0, 1.53, decimals=3)
        self.spin_tw = self._create_double_spinbox(0.1, 3.0, 1.20, decimals=3)
        self.spin_wd = self._create_double_spinbox(0.0, 1.0, 0.53, step=0.01)
        self.spin_cogz = self._create_double_spinbox(0.0, 1.5, 0.25, step=0.01)

        form_chassis.addRow("Total weight [kg]:", self.spin_weight)
        form_chassis.addRow("Wheelbase [m]:", self.spin_wb)
        form_chassis.addRow("Track width [m]:", self.spin_tw)
        form_chassis.addRow("Weight distribution (front) [0-1]:", self.spin_wd)
        form_chassis.addRow("CoG height [m]:", self.spin_cogz)
        group_chassis.setLayout(form_chassis)
        self.container_layout.addWidget(group_chassis)

        # --- Section: Aerodynamics ---
        group_aero = QGroupBox("Aerodynamics")
        group_aero.setStyleSheet(blue_border_style)
        form_aero = QFormLayout()
        self.spin_cd = self._create_double_spinbox(0, 10, 1.83, step=0.01)
        self.spin_cl = self._create_double_spinbox(0, 15, 5.64, step=0.05)
        self.spin_cop = self._create_double_spinbox(0, 1.0, 0.53, step=0.01)
        self.spin_cd_drs = self._create_double_spinbox(0, 10, 1.17, step=0.01)
        self.spin_cl_drs = self._create_double_spinbox(0, 15, 4.20, step=0.05)

        form_aero.addRow("Drag coefficient (CdA):", self.spin_cd)
        form_aero.addRow("Lift coefficient (ClA):", self.spin_cl)
        form_aero.addRow("Center of pressure [0-1]:", self.spin_cop)
        form_aero.addRow("CdA (DRS open):", self.spin_cd_drs)
        form_aero.addRow("ClA (DRS open):", self.spin_cl_drs)
        group_aero.setLayout(form_aero)
        self.container_layout.addWidget(group_aero)

        # --- Section: Powertrain ---
        group_power = QGroupBox("Powertrain")
        group_power.setStyleSheet(blue_border_style)
        form_power = QFormLayout()
        self.spin_power_max = self._create_double_spinbox(0, 250, 80.0, step=1.0, decimals=1)
        self.spin_M_max = self._create_double_spinbox(0, 500, 30.0, step=0.5)
        self.spin_i = self._create_double_spinbox(0, 50, 12.0, step=0.1)
        self.spin_regen = self._create_double_spinbox(0.0, 1.0, 0.8, step=0.05)

        form_power.addRow("Max power limit [kW]:", self.spin_power_max)
        form_power.addRow("Max motor torque [Nm]:", self.spin_M_max)
        form_power.addRow("Gear ratio (i):", self.spin_i)
        form_power.addRow("Driver recuperation factor [0-1]:", self.spin_regen)
        group_power.setLayout(form_power)
        self.container_layout.addWidget(group_power)

        self.scroll_area.setWidget(self.container)
        self.main_layout.addWidget(self.scroll_area)

        # --- Bottom Button Area ---

        self.button_container = QWidget()
        self.button_container.setObjectName("HighlightedGroup")
        
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(15, 15, 15, 15)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        self.button_layout.addWidget(self.buttons)
        self.main_layout.addWidget(self.button_container)

    def _create_double_spinbox(self, v_min, v_max, default_val, step=0.1, decimals=2):
        sb = QDoubleSpinBox()
        sb.setRange(v_min, v_max)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        sb.setValue(default_val)
        return sb

    def _populate_data(self):
        # Fill UI from dict 
        p = self.params
        self.edit_name.setText(p.get("name", "New Vehicle"))
        self.spin_weight.setValue(p.get("weight", 280.0))
        self.spin_wb.setValue(p.get("wb", 1.53))
        self.spin_tw.setValue(p.get("tw", 1.20))
        self.spin_wd.setValue(p.get("wd", 0.53))
        self.spin_cogz.setValue(p.get("cogz", 0.25))
        self.spin_cd.setValue(p.get("cd", 1.83))
        self.spin_cl.setValue(p.get("cl", 5.64))
        self.spin_cop.setValue(p.get("cop", 0.53))
        self.spin_cd_drs.setValue(p.get("cd_drs", 1.17))
        self.spin_cl_drs.setValue(p.get("cl_drs", 4.2))
        self.spin_power_max.setValue(p.get("power_limit", 80.0))
        self.spin_M_max.setValue(p.get("M_max", 30.0))
        self.spin_i.setValue(p.get("i", 12.0))
        self.spin_regen.setValue(p.get("regen_eff", 0.8))

    def get_params(self) -> dict:
        # Return updated dict 
        new_params = self.params.copy()
        new_params.update({
            "name": self.edit_name.text(),
            "weight": self.spin_weight.value(),
            "wb": self.spin_wb.value(),
            "tw": self.spin_tw.value(),
            "wd": self.spin_wd.value(),
            "cogz": self.spin_cogz.value(),
            "cd": self.spin_cd.value(),
            "cl": self.spin_cl.value(),
            "cop": self.spin_cop.value(),
            "cd_drs": self.spin_cd_drs.value(),
            "cl_drs": self.spin_cl_drs.value(),
            "power_limit": self.spin_power_max.value(),
            "M_max": self.spin_M_max.value(),
            "i": self.spin_i.value(),
            "regen_eff": self.spin_regen.value()
        })
        return new_params