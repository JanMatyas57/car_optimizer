
import os
import glob
import numpy as np 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QGroupBox, QComboBox, 
                             QDoubleSpinBox, QPushButton, QSplitter, QCheckBox, QSpinBox,QFileDialog,QMessageBox)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from src.core.track import Track
from  src.core.trackGenerator import TrackGenerator
from src.core.car import Car
from src.optimization.solver import MassPointSolver, MPCC_Solver, SingleTrackSolver, StrategyMaker
from src.optimization.work_thread import SolverWorker, MODE_ACP_TEST, MODE_ENDURANCE, MODE_SINGLE, MODE_STRATEGY_OPT
from src.optimization.track_generation_thread import TrackGenerationWorker
from src.ui.results_window import ResultsWindow
from src.ui.endurance_results import EnduranceResultsWindow
from src.ui.results_tab import CompareWindow
from src.ui.loader import LoadingOverlay
from src.ui.drs_plot import DRSResultsWindow
from src.ui.sensitivity_plot import LaptimeSensitivityPlot
from src.ui.endurance_strategy import LaptimeAppMainWindow
from PyQt6.QtGui import QTransform, QColor
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
       
        FigureCanvasQTAgg.updateGeometry(self)

# graph styling
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True) 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OptiSim")
        self.resize(1200, 900)

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # left panel 
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setFixedWidth(320)
        
        self.init_track_section() 
        self.init_car_params_section()
        
        # Single lap button
        self.btn_run = QPushButton("Run single lap")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.on_run_single_clicked)
        self.control_layout.addWidget(self.btn_run)
        
        self.control_layout.addStretch()
        
        # Endurance section
        self.init_endurance_section()
        
        self.control_layout.addStretch()

        self.btn_create_track = QPushButton("Create track from data")
        self.btn_create_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create_track.setStyleSheet("""
            QPushButton { 
                background-color: #6f42c1; /* Fialová */
                color: white;              /* Bílý text */
                font-weight: bold; 
                padding: 10px; 
                border-radius: 4px; 
                margin-top: 10px; 
            }
            QPushButton:hover { 
                background-color: #5a32a3; /* Tmavší fialová pro hover */
            }
        """)
        self.btn_create_track.clicked.connect(self.on_create_track_clicked)
        self.control_layout.addWidget(self.btn_create_track)

        # Compare Button
        self.btn_compare = QPushButton("Compare logs")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 10px; border-radius: 4px; margin-top: 10px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.btn_compare.clicked.connect(self.on_compare_clicked)
        self.control_layout.addWidget(self.btn_compare)

      # --- RIGHT PANEL (Graphs) ---
        self.graph_panel = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_panel)
        
        # Místo PyQtGraph vytvoříme Matplotlib Canvas
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.graph_layout.addWidget(self.mpl_canvas)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.graph_panel)
        splitter.setSizes([320, 880])
        self.main_layout.addWidget(splitter)

        # Loading overlay
        self.loading_overlay = LoadingOverlay(self)
        
        # Initial data load
        self.load_track_list()

    def resizeEvent(self, event):
        self.loading_overlay.resize(self.size())
        super().resizeEvent(event)

    # ==========================================
    # UI INITIALIZATION METHODS
    # ==========================================

    def init_track_section(self):
        group = QGroupBox("Track Selection")
        layout = QFormLayout()

        self.combo_track = QComboBox()
        self.combo_track.currentIndexChanged.connect(self.on_track_changed) 
        
        layout.addRow("Track:", self.combo_track)
        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def init_car_params_section(self):
        group = QGroupBox("Car Parameters")
        layout = QFormLayout()

        self.combo_model = QComboBox()
        self.combo_model.addItem("Mass point (fixed line)", "mass_point")
        self.combo_model.addItem("Mass point (free line)", "mass_point_mpcc")
        self.combo_model.addItem("Single track", "single_track")

    
        layout.addRow("Vehicle model:", self.combo_model)

        def add_param(label, val, unit, min_val=0.0, max_val=10000.0, step=0.1):
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(val)
            spin.setSingleStep(step)
            spin.setSuffix(f" {unit}")
            layout.addRow(label, spin)
            return spin

        self.input_mass = add_param("Mass:", 280, "kg", min_val=150, step=0.5)
        self.input_cd = add_param("Drag Coeff (Cd):", 1.83, "", step=0.01)
        self.input_cl = add_param("Downforce Coeff (Cl):", 5.64, "", step=0.01)
        self.input_mu_x = add_param("Friction coeff x:", 1.25, "", step=0.01)
        self.input_mu_y = add_param("Friction coeff y:", 1.25, "", step=0.01)
        self.input_ds = add_param("Sim distance step:", 0.5, "m", min_val=0.1, max_val=2.0, step=0.05)
        
        # Single Lap Energy Limit
        self.check_energy_limit = QCheckBox("Enable Energy Limit")
        self.check_energy_limit.setChecked(False) 
        self.input_limit_e = add_param("Energy limit", 0.2, "kWh", min_val=0.05, step=0.01)
        self.input_limit_e.setEnabled(False)

        self.check_energy_limit.toggled.connect(self.input_limit_e.setEnabled)
        layout.addRow(self.check_energy_limit, self.input_limit_e)
        self.check_drs = QCheckBox("Enable DRS")
        self.check_drs.setChecked(True) 
        layout.addRow(self.check_drs)

        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def init_endurance_section(self):
        group = QGroupBox("Endurance Parameters")
        layout = QFormLayout()

        def add_dspin(val, unit, min_val, max_val, step):
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(val)
            spin.setSingleStep(step)
            spin.setSuffix(f" {unit}")
            return spin

        self.input_battery_cap = add_dspin(6.0, "kWh", 1.0, 20.0, 0.05)
        layout.addRow("Battery Capacity:", self.input_battery_cap)

        self.input_laps = QSpinBox()
        self.input_laps.setRange(1, 200)
        self.input_laps.setValue(22) 
        self.input_laps.setSuffix(" laps")
        layout.addRow("Number of Laps:", self.input_laps)

        # Endurance Run Button
        self.btn_run_endurance = QPushButton("Run endurance")
        self.btn_run_endurance.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_endurance.setStyleSheet("""
            QPushButton { background-color: #198754; color: white; font-weight: bold; padding: 10px; margin-top: 5px; }
            QPushButton:hover { background-color: #157347; }
        """)
        self.btn_run_endurance.clicked.connect(self.on_run_endurance_clicked)
        layout.addRow(self.btn_run_endurance)

        # Endurance Strategy Button
        self.btn_endurance_strategy = QPushButton("Endurance strategy")
        self.btn_endurance_strategy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_endurance_strategy.setStyleSheet("""
            QPushButton { background-color: #17a2b8; color: white; font-weight: bold; padding: 10px; margin-top: 5px; border-radius: 4px; }
            QPushButton:hover { background-color: #138496; }
        """)
        self.btn_endurance_strategy.clicked.connect(self.on_endurance_strategy_clicked)
        layout.addRow(self.btn_endurance_strategy)

        # ACP Test Button
        self.btn_acp_test = QPushButton("Run laptime sensitivity test")
        self.btn_acp_test.setStyleSheet("background-color: #6c757d; color: white; padding: 5px; margin-top: 5px;")
        self.btn_acp_test.clicked.connect(self.on_acp_test_clicked)
        layout.addRow(self.btn_acp_test)

        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def create_solver(self,car, track, endurance=False, laps=1):
        """
        Creates the correct solver based on car configuration.
        """
        if car.model_type == "single_track":
            return SingleTrackSolver(car, track, laps=laps)
            
        elif car.model_type == "mass_point":
            return MassPointSolver(car, track, laps=laps)
        
        elif car.model_type == "mass_point_mpcc":
            return MPCC_Solver(car,track,laps=laps)
            
        else:
            # Fallback
            print(f"⚠️ Unknown model type '{car.model_type}', defaulting to MassPoint.")
            return MassPointSolver(car, track, endurance=endurance, laps=laps)

    # ==========================================
    # DATA LOADING & TRACK PLOTTING
    # ==========================================

    def load_track_list(self):
        """Searches 'data/tracks_render' for files."""
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
            self.combo_track.addItem("No tracks found")
            self.combo_track.setEnabled(False)
            
        self.combo_track.blockSignals(False)
        self.on_track_changed()

    def on_track_changed(self):
        """Loads track preview using Track.load instead of from_csv."""
        track_name = self.combo_track.currentText()
        if not track_name or track_name == "No tracks found":
            return

        try:
            # Změna: Místo Track.from_csv(filepath) použijeme Track.load(track_name)
            # Track.load si interně sáhne do správné složky (data/tracks)
            self.current_track = Track.load(track_name)
            self.plot_track_map(self.current_track)
            
            # --- FEATURE: Auto-Calculate Laps ---
            dist_m = 22000
            laps = int(round(dist_m / self.current_track.length))
            laps = max(1, laps)
            
            self.input_laps.setValue(laps)
            print(f"Track loaded: {track_name} (L={self.current_track.length:.1f}m) -> Auto-set {laps} laps.")
            
        except Exception as e:
            print(f"Error loading track '{track_name}': {e}")
            self.track_plot_widget.clear()
            self.track_plot_widget.setTitle(f"Error: {e}")

    def plot_track_map(self, track: Track):
        # 1. Vyčištění celé figury (smaže i staré colorbary)
        self.mpl_canvas.fig.clf()
        ax = self.mpl_canvas.fig.add_subplot(111)
        
        # 2. Příprava dat
        x = track.x
        y = track.y
        
        # Získání Alpha (sklonu) ve stupních
        if hasattr(track, 'alpha') and track.alpha is not None:
            alpha_deg = np.degrees(track.alpha)
            legend_flag = True
        else:
            alpha_deg = np.zeros_like(x)
            legend_flag = False

        # 3. Segmenty pro LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 4. Nastavení barev (Sklon)
        # Rozsah -10° (modrá) až +10° (červená)
        norm = Normalize(vmin=np.min(alpha_deg), vmax=np.max(alpha_deg))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(alpha_deg[:-1])
        lc.set_linewidth(3)
        ax.add_collection(lc)

        # 5. Limity a Aspect Ratio
        ax.set_xlim(x.min() - 20, x.max() + 20)
        ax.set_ylim(y.min() - 20, y.max() + 20)
        ax.set_aspect('equal')

        # 6. Start a Šipka
        ax.plot(x[0], y[0], 'o', color='lime', markersize=8, markeredgecolor='black')

        if hasattr(track, 'psi') and track.psi is not None:
            heading = track.psi[0]
            ax.arrow(x[0], y[0], 
                     np.cos(heading) * 40, np.sin(heading) * 40, 
                     head_width=5, head_length=5, fc='black', ec='black')

        # 7. Legenda (Colorbar)
        # fraction=0.046 a pad=0.04 jsou "magická čísla" v Matplotlibu, 
        # která zajistí, že výška legendy přesně odpovídá výšce grafu.
        if legend_flag:
            cb = self.mpl_canvas.fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Incline [deg]", size=10)
            cb.outline.set_visible(False) # Odstraní rámeček kolem legendy pro čistší vzhled

        # 8. Čistý vzhled (bez os)
        ax.axis('off')
        
        self.mpl_canvas.draw()
  
    # ==========================================
    # SIMULATION HANDLERS
    # ==========================================

    def on_run_single_clicked(self):
        print("Preparing SINGLE LAP simulation...")
        try:
            track_name = self.combo_track.currentText()
            sim_track = Track.load(track_name)
        except: return

        car = Car.from_ui_widget(self)
        solver = self.create_solver(car, sim_track)
        
        self.loading_overlay.show_loading(text="Running single lap.", indefinite=True)
        
        self.ds_used = self.input_ds.value()
        self.worker = SolverWorker(solver, car, sim_track, self.ds_used, mode=MODE_SINGLE)
        
        self.connect_worker_signals()
        self.worker.start()

    def on_run_endurance_clicked(self):
        print("Preparing ENDURANCE simulation...")
        try:
            track_name = self.combo_track.currentText()
            sim_track = Track.load(track_name)
        except: return

        car = Car.from_ui_widget(self)
        laps_count = self.input_laps.value()
        
        # Override car limits for endurance
        batt_kwh = self.input_battery_cap.value()
        car.acp_capacity = batt_kwh # Store for solver usage
        
        solver = self.create_solver(car, sim_track, endurance=True, laps=laps_count)
        
        self.loading_overlay.show_loading(
            text="Starting endurance...", 
            indefinite=True,
            progress_text_fmt="Simulating lap {} / {}..."
        )
        
        self.ds_used = self.input_ds.value()
        self.worker = SolverWorker(solver, car, sim_track, self.ds_used, laps=laps_count, mode=MODE_ENDURANCE)
        
        self.connect_worker_signals()
        self.worker.start()

    def on_endurance_strategy_clicked(self):
       
        try:
            track_name = self.combo_track.currentText()
            name = track_name+"_laptime_map"
            tracks_dir="data/tracks/lapmaps"
            filepath = os.path.join(tracks_dir, f"{name}.npz")
            self.laptime_map_plot = LaptimeAppMainWindow(filepath)
            self.laptime_map_plot.optimization_requested.connect(self.start_strategy_optimization)
            self.laptime_map_plot.show()


        except Exception as e: 
            QMessageBox.critical(self, "Error", f"Could not load laptime map:\n{e}")
            return
        

    def start_strategy_optimization(self, regen_val, delta_val,map_data):
       
        car = Car.from_ui_widget(self)
        laps_count = self.input_laps.value()
        
        laptime_interpolant = map_data['interpolant']
        #create solver

        solver = StrategyMaker(car,laptime_interpolant,laps_count)

        self.loading_overlay.show_loading("Optimizing strategy...", indefinite=True)
        
        # Vytvoříme worker s novým módem
        self.ds_used = self.input_ds.value()
        car.regen_eff = regen_val
        car.performance_delta = delta_val
     
        self.worker = SolverWorker(solver, car, None, self.ds_used, mode=MODE_STRATEGY_OPT, laps=laps_count)
        
        self.connect_worker_signals()
        self.worker.start()

        

    def on_acp_test_clicked(self):
        
        try:
             track_name = self.combo_track.currentText()
             sim_track = Track.load(track_name)
        except: return

        car = Car.from_ui_widget(self)
        
        # Ensure we start with correct mass and capacity from UI
        car.acp_capacity = self.input_battery_cap.value()

        laps_count = self.input_laps.value()
        
        solver = MassPointSolver(car, sim_track,laps=laps_count)
        
        self.loading_overlay.show_loading(
            text="Starting sensitivity analysis...", 
            indefinite=True,
            progress_text_fmt="Running lap {} / {}..."
        )
        
        self.ds_used = self.input_ds.value()
        self.worker = SolverWorker(solver, car, sim_track, self.ds_used, mode=MODE_ACP_TEST,laps=laps_count)
        
        self.connect_worker_signals()
        self.worker.start()

    def connect_worker_signals(self):
        """Helper to connect standard worker signals."""
        self.worker.finished.connect(self.on_solver_finished)
        self.worker.error.connect(self.on_solver_error)
        self.worker.progress.connect(self.loading_overlay.update_progress)

    def on_create_track_clicked(self):
        """
        Wizard pro vytvoření tratě z .mat logu.
        """
        fname, _ = QFileDialog.getOpenFileName(self, "Select telemetry log", "", "MAT Files (*.mat)")
        if not fname: return

        # Loading dat (to je rychlé, může být v main threadu, nebo taky ve workeru, ale nevadí to tolik)
        self.loading_overlay.show_loading("Parsing Log Data...", indefinite=True)
        QApplication.processEvents()

        try:
            data = TrackGenerator.filter_data(fname)
        except Exception as e:
            self.loading_overlay.hide_loading()
            QMessageBox.critical(self, "Error", f"Failed to parse file:\n{e}")
            return
            
        self.loading_overlay.hide_loading()
        if data is None: return

        # Výběr módu
        msg = QMessageBox(self)
        msg.setWindowTitle("Select track mode")
        msg.setText("How do you want to define the track?")
        btn_closed = msg.addButton("Closed track (endurance)", QMessageBox.ButtonRole.AcceptRole)
        btn_open = msg.addButton("Open section (autocross)", QMessageBox.ButtonRole.RejectRole)
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

        # 
        default_name = f"track_{lap_info['time']:.2f}s.npz"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Baked Track", 
            os.path.join("data/tracks", default_name), 
            "NumPy Zip (*.npz)"
        )

        if save_path:
        
            self.loading_overlay.show_loading("Smoothing track...", indefinite=True)
            
            self.track_worker = TrackGenerationWorker(data, lap_info, is_closed_mode, save_path)
            self.track_worker.finished.connect(self.on_track_generation_finished)
            self.track_worker.start()

    def on_track_generation_finished(self, success, message):
        self.loading_overlay.hide_loading()
        
        if success:
            QMessageBox.information(self, "Success", f"Track saved successfully!\n{message}")
            self.load_track_list()
            
            # Automaticky vybrat novou trať v Comboboxu
            track_name = os.path.splitext(os.path.basename(message))[0]
            idx = self.combo_track.findText(track_name)
            if idx >= 0:
                self.combo_track.setCurrentIndex(idx)
        else:
            QMessageBox.critical(self, "Error", f"Track generation failed:\n{message}")

    # ==========================================
    # RESULTS HANDLERS
    # ==========================================

    def on_solver_finished(self, res):
        self.loading_overlay.hide_loading()
        
        if hasattr(self, 'laptime_map_plot') and self.laptime_map_plot is not None:
            self.laptime_map_plot.stop_loading()
    
        if res.kind == "single_lap":
            res = res.data
            self.results_window = ResultsWindow(res, self.worker.car, self.worker.track, ds=self.ds_used)
            self.results_window.show()
            return
        
        elif res.kind == "endurance":
            self.results_window = EnduranceResultsWindow(res.data, self.worker.car, self.worker.track, ds=self.ds_used)
            self.results_window.show()
            return
        
        elif res.kind == "drs_test":
            drs_data = res.data['drs']
            nodrs_data = res.data['nodrs']
            self.results_window = DRSResultsWindow(drs_data, nodrs_data)
            self.results_window.show()
            return
        
        elif res.kind == "acp_test":
            self.sens_plotter = LaptimeSensitivityPlot(res.data,laps=self.worker.laps) 
            self.sens_plotter.open_windows()
            return
        
        elif res.kind == "endurance_strategy":
        
            if hasattr(self, 'laptime_map_plot') and self.laptime_map_plot is not None:
                # res je vlastně SolverPayload, takže ho můžeme předat jako slovník s atributy
                payload_dict = {
                    'data': res.data,
                    'meta': res.meta
                }
                self.laptime_map_plot.add_result_tab(payload_dict)
            return
        
        else:
            print("Optimization failed!")
            return
        

    def on_solver_error(self, error_msg):
        self.loading_overlay.hide_loading()
        if hasattr(self, 'laptime_map_plot') and self.laptime_map_plot is not None:
            self.laptime_map_plot.stop_loading()
        print(f"Solver Error: {error_msg}")

    def on_compare_clicked(self):
        self.compare_window = CompareWindow()
        self.compare_window.show()
    '''
    def show_acp_plot(self, results):
        """Creates strategy analysis window."""
        results.sort(key=lambda x: x.E[-1])

        times = np.array([r.lap_time for r in results])
        energies_kwh = np.array([r.E[-1] - r.E[0] for r in results])
        
        # --- POINTS CALCULATION (FSG RULES) ---
        T_min = np.min(times)
        T_max = T_min * 1.333
        denom = (T_max / T_min) - 1
        
        p_endurance = []
        for t in times:
            if t > T_max: p_endurance.append(0)
            else: p_endurance.append(250 * ((T_max / t) - 1) / denom)
        p_endurance = np.array(p_endurance)

        eff_metric = energies_kwh * (times**2)
        metric_best = np.min(eff_metric)
        metric_worst = np.max(eff_metric)
        
        if metric_worst != metric_best:
            p_efficiency = 75 * (1 - (eff_metric - metric_best) / (metric_worst - metric_best))
        else:
            p_efficiency = np.full_like(eff_metric, 75)

        p_total = p_endurance + p_efficiency
        opt_idx = np.argmax(p_total)
        opt_energy = energies_kwh[opt_idx]
        opt_score = p_total[opt_idx]

        # --- PLOTTING ---
        self.acp_window = pg.GraphicsLayoutWidget(show=True, title="ACP Strategy Analysis")
        self.acp_window.resize(1200, 900)
        self.acp_window.setBackground('w')

        # 1. Trade-off Curve
        p1 = self.acp_window.addPlot(title="Lap Time vs Energy Used")
        p1.setLabel('left', 'Lap Time', units='s')
        p1.setLabel('bottom', 'Energy Used', units='kWh')
        p1.plot(energies_kwh, times, pen=pg.mkPen('b', width=3), symbol='o', symbolBrush='b')
        p1.plot([opt_energy], [times[opt_idx]], symbol='x', symbolSize=20, symbolBrush='y')

        self.acp_window.nextRow()

        # 2. Points Curve
        p2 = self.acp_window.addPlot(title="Points estimator")
        p2.setLabel('left', 'Points', units='pts')
        p2.setLabel('bottom', 'Energy Used', units='kWh')
        p2.addLegend()

        p2.plot(energies_kwh, p_endurance, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine), name='Endurance')
        p2.plot(energies_kwh, p_efficiency, pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine), name='Efficiency')
        p2.plot(energies_kwh, p_total, pen=pg.mkPen('k', width=4), name='Total Score')

        v_line = pg.InfiniteLine(pos=opt_energy, angle=90, pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
        p2.addItem(v_line)
        
        info_html = (f"<div style='color:black; background-color: rgba(255,255,0,0.7); padding:3px; font-weight:bold;'>"
                     f"OPTIMUM:<br>E: {opt_energy:.2f} kWh<br>Pts: {opt_score:.1f}</div>")
        text = pg.TextItem(html=info_html, anchor=(0.5, 0))
        text.setPos(opt_energy, opt_score)
        p2.addItem(text)
        
        p2.setXLink(p1)
        '''