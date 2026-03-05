import numpy as np
import casadi as ca
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QMessageBox, QPushButton, 
                             QDoubleSpinBox, QTabWidget, QGroupBox, QApplication, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from src.ui.loader import LoadingOverlay
from scipy.ndimage import median_filter

class MapTab(QWidget):
    def __init__(self, map_data, fixed_axis='soc', parent=None):
        super().__init__(parent)
        self.map_data = map_data
        self.fixed_axis = fixed_axis
        
        self.laptime_map = map_data['interpolant']
        self.z_min = map_data['z_min']
        self.z_max = map_data['z_max']
        
        main_layout = QVBoxLayout(self) 
        
        # Matplotlib Canvas 
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#ffffff') # Bílé pozadí pro moderní look
        self.canvas = FigureCanvasQTAgg(self.fig)
                
        content_layout = QHBoxLayout()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self.canvas, stretch=1)
        
        # --- Slider ---
        slider_layout = QVBoxLayout() 
        self.lbl_slider = QLabel("")
        self.lbl_slider.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.lbl_slider.setStyleSheet("font-weight: bold; color: #3b82f6; font-size: 14px;")
        
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TickPosition.TicksRight) 
        
        if self.fixed_axis == 'soc':
            self.slider.setMinimum(10)
            self.slider.setMaximum(100)
            self.slider.setTickInterval(10)
            self.slider.setValue(50)
            self.X_mesh, self.Y_mesh = np.meshgrid(map_data['e_fine'], map_data['r_fine'])
        else:
            self.slider.setMinimum(int(map_data['r_min'] * 100))
            self.slider.setMaximum(100)
            self.slider.setTickInterval(10)
            self.slider.setValue(int(map_data['r_max'] * 50))
            self.X_mesh, self.Y_mesh = np.meshgrid(map_data['e_fine'], map_data['soc_fine'])

        slider_layout.addWidget(self.lbl_slider)
        slider_layout.addWidget(self.slider)
        
        content_layout.addLayout(slider_layout)
        main_layout.addLayout(content_layout)
        
        # Camera default view
        self.ax3d = None
        self.view_elev = 20
        self.view_azim = 30
        
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.on_slider_changed(self.slider.value()) 

    def on_slider_changed(self, val):
        if self.fixed_axis == 'soc':
            self.lbl_slider.setText(f"SoC:\n{val} %")
        else:
            self.lbl_slider.setText(f"Regen:\n{val/100.0:.2f}")
        self.update_plot()

    def update_plot(self):
        if self.fixed_axis == 'soc':
            soc_val = self.slider.value() / 100.0
            pts_casadi = np.vstack((
                np.full_like(self.X_mesh.ravel(), soc_val),
                self.X_mesh.ravel(),
                self.Y_mesh.ravel()
            ))
            x_label, y_label = 'Energy target [kWh]', 'Regen [%]'
            title = f'Laptime map (SoC = {soc_val*100:.0f} %)'
        else:
            regen_val = self.slider.value() / 100.0
            pts_casadi = np.vstack((
                self.Y_mesh.ravel(),
                self.X_mesh.ravel(),
                np.full_like(self.X_mesh.ravel(), regen_val)
            ))
            x_label, y_label = 'Energy target [kWh]', 'SoC [%]'
            title = f'Laptime map (Regen = {regen_val:.2f})'

        Z_laptime = np.array(self.laptime_map(pts_casadi)).reshape(self.X_mesh.shape)

        if self.ax3d is not None:
            self.view_elev = self.ax3d.elev
            self.view_azim = self.ax3d.azim

        self.fig.clear()
        self.ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
        
        surf = self.ax3d.plot_surface(
            self.X_mesh, self.Y_mesh*100, Z_laptime, 
            cmap='viridis_r', edgecolor='none', alpha=0.9,
            vmin=self.z_min, vmax=self.z_max
        )
        
        self.ax3d.set_zlim(self.z_min, self.z_max)
        
        # Sjednocení stylu fontů - odstranění bold a úprava barev
        self.ax3d.set_xlabel('Energy target [kWh]', color='#475569', fontweight='normal')
        self.ax3d.set_ylabel(y_label, color='#475569', fontweight='normal')
        self.ax3d.set_zlabel('Lap time [s]', color='#475569', fontweight='normal')
        
        # Title bez bold, stejná barva jako u výsledků
        self.ax3d.set_title(title, color='#475569', fontweight='normal', pad=15)
        
        # Stylování čísel na osách (ticks)
        self.ax3d.xaxis.set_tick_params(colors='#64748b')
        self.ax3d.yaxis.set_tick_params(colors='#64748b')
        self.ax3d.zaxis.set_tick_params(colors='#64748b')
        
        self.ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Optimalizace okrajů, aby graf nebyl zbytečně malý
      # self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)
        self.canvas.draw_idle()


class LaptimeAppMainWindow(QWidget):
    optimization_requested = pyqtSignal(float, float, float, float, object)
    deployment_requested = pyqtSignal(float, float, float, float, object)

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Endurance strategy")

        self.setStyleSheet("""
            /* Nastavíme pozadí jen pro hlavní okno a čisté kontejnery */
            LaptimeAppMainWindow, .QWidget {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                color: #334155;
                background-color: #f8f9fa;
            }
            
            /* --- TLAČÍTKA --- */
            QPushButton {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 15px;
                color: white;
                border: none;
            }
            QPushButton#BtnCompare { background-color: #3b82f6; } /* Modrá */
            QPushButton#BtnCompare:hover { background-color: #2563eb; }
            
            QPushButton#BtnRunSingle { background-color: #10b981; } /* Zelená */
            QPushButton#BtnRunSingle:hover { background-color: #059669; }

            /* --- KONTEJNERY A ZÁLOŽKY --- */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0px 5px;
                color: #475569;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e2e8f0;
                color: #64748b;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
                border-bottom: 2px solid #3b82f6;
            }
            QTabBar::tab:hover:!selected {
                background: #cbd5e1;
            }
        """)
    
        self.loading_overlay = LoadingOverlay(self)
        self.map_data = self.load_data(filepath)
        
        # --- DYNAMICKÁ VELIKOST OKNA ---
        screen = QGuiApplication.primaryScreen()
        available_geometry = screen.availableGeometry()

        screen_width = available_geometry.width()
        screen_height = available_geometry.height()

        target_width = int(screen_width * 0.85)
        target_height = int(screen_height * 0.90)

        # Naše nové limity pro 2K monitory
        target_width = min(target_width, 1600)
        target_height = min(target_height, 1200)

        self.resize(target_width, target_height)

        if target_width >= screen_width * 0.95 or target_height >= screen_height * 0.95:
            self.showMaximized()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # --- Ovládací Panel ---
        control_group = QGroupBox("Optimization Settings")
        control_group.setObjectName("HighlightedGroup") # Aplikace šedého podkladu
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(15, 10, 15, 15)
        
        # Napojení tlačítek na globální barvy přes ID
        self.btn_optimize = QPushButton("Calculate optimal energy map")
        self.btn_optimize.setObjectName("BtnCompare") # Modrá z globálního CSS
        self.btn_optimize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_optimize.clicked.connect(self.run_optimization)

        self.btn_deploy = QPushButton("Calculate deployment for given inputs")
        self.btn_deploy.setObjectName("BtnRunSingle") # Zelená z globálního CSS
        self.btn_deploy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_deploy.clicked.connect(self.run_deployment)
        
        # Inputs
        lbl_regen = QLabel("Recuperation [%]:")
        lbl_regen.setStyleSheet("font-weight: bold;")
        self.spin_regen = QDoubleSpinBox()
        self.spin_regen.setRange(0.0, 100)
        self.spin_regen.setSingleStep(1)
        self.spin_regen.setValue(80) 
        
        lbl_delta = QLabel("Time delta [%]:")
        lbl_delta.setStyleSheet("font-weight: bold;")
        self.spin_delta = QDoubleSpinBox()
        self.spin_delta.setRange(0, 10.0)
        self.spin_delta.setSingleStep(0.5)
        self.spin_delta.setValue(0.0)
        
        control_layout.addWidget(self.btn_optimize)
        control_layout.addWidget(self.btn_deploy)
        control_layout.addStretch() 
        control_layout.addWidget(lbl_regen)
        control_layout.addWidget(self.spin_regen)
        control_layout.addSpacing(20)
        control_layout.addWidget(lbl_delta)
        control_layout.addWidget(self.spin_delta)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # --- Záložky ---
        self.tabs = QTabWidget()
        
        if self.map_data:
            self.tab_soc = MapTab(self.map_data, fixed_axis='soc')
            self.tab_regen = MapTab(self.map_data, fixed_axis='regen')
            
            side_by_side_widget = QWidget()
            side_by_side_layout = QHBoxLayout(side_by_side_widget)
            side_by_side_layout.setContentsMargins(10, 10, 10, 10)
            side_by_side_layout.addWidget(self.tab_soc)
            side_by_side_layout.addWidget(self.tab_regen)
            
            self.tabs.addTab(side_by_side_widget, "Laptime maps ")
        else:
            self.tabs.addTab(QLabel("Failed to load map data."), "Error")
            
        main_layout.addWidget(self.tabs)

    def load_data(self, filepath):
        try:
            data = np.load(filepath)
            soc_targets = data['soc_targets']
            e_targets = data['e_targets']
            regen_targets = data['regen_targets']
            flat_lap_times = data['flat_lap_times']
            
            shape_3d = (len(soc_targets), len(e_targets), len(regen_targets))
            lap_times_3d = flat_lap_times.reshape(shape_3d)
            
            # Outlier filter
            smoothed_3d = median_filter(lap_times_3d, size=3)
            diff = np.abs(lap_times_3d - smoothed_3d)
            outliers_mask = diff > 5.0
            num_outliers = np.sum(outliers_mask)
            
            if num_outliers > 0:
                print(f"[Data Load] Repaired {num_outliers} outliers via median filter.")
                lap_times_3d[outliers_mask] = smoothed_3d[outliers_mask]
            
            flat_for_casadi = lap_times_3d.ravel(order='F')
            grid_axes = [soc_targets.tolist(), e_targets.tolist(), regen_targets.tolist()]
            interpolant = ca.interpolant('laptime_map', 'bspline', grid_axes, flat_for_casadi.tolist())

            self.min_e_target = min(e_targets)
            self.max_e_target = max(e_targets)
            
            return {
                'interpolant': interpolant,
                'e_fine': np.linspace(min(e_targets), max(e_targets), 50),
                'r_fine': np.linspace(min(regen_targets), 1.0, 50),
                'soc_fine': np.linspace(min(soc_targets), 1.0, 50),
                'r_min': min(regen_targets),
                'r_max': max(regen_targets),
                'z_min': np.nanmin(lap_times_3d),
                'z_max': np.nanmax(lap_times_3d) 
            }
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load map from {filepath}:\n{e}")
            return None

    def resizeEvent(self, event):
        self.loading_overlay.resize(self.size())
        super().resizeEvent(event)

    def add_result_tab(self, payload):
        if payload.kind == "endurance_map": 
            has_opt_energy_tab = False
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Optimal energy map":
                    has_opt_energy_tab = True
                    self.tabs.setCurrentIndex(i) 
                    
            if not has_opt_energy_tab:
                opt_energy_tab = OptimalEnergyMapTab(payload, self)
                self.tabs.addTab(opt_energy_tab, "Optimal energy map")
                self.tabs.setCurrentWidget(opt_energy_tab)

        elif payload.kind == "endurance_deployment":
            result_count = sum(1 for i in range(self.tabs.count()) if self.tabs.tabText(i).startswith("Deployment #"))
            result_tab = StrategyResultTab(payload, self)
            tab_name = f"Deployment #{result_count + 1}"
            self.tabs.addTab(result_tab, tab_name)
            self.tabs.setCurrentWidget(result_tab)

    def run_optimization(self):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Optimal energy map":
                self.tabs.setCurrentIndex(i)
                return
            
        regen_val = self.spin_regen.value() / 100
        delta_val = self.spin_delta.value() / 100
        self.loading_overlay.show_loading("Calculating optimal strategy...", indefinite=True, progress_text_fmt="Solving grid points ({} / {})")
        self.optimization_requested.emit(regen_val, delta_val, self.min_e_target, self.max_e_target, self.map_data)

    def run_deployment(self):
        regen_val = self.spin_regen.value() / 100
        delta_val = self.spin_delta.value() / 100
        self.loading_overlay.show_loading("Calculating optimum energy deployment...", indefinite=True)
        self.deployment_requested.emit(regen_val, delta_val, self.min_e_target, self.max_e_target, self.map_data)

    def stop_loading(self, *args, **kwargs):
        self.loading_overlay.hide_loading()


class StrategyResultTab(QWidget):
    def __init__(self, payload, parent=None):
        super().__init__(parent)
        
        self.res_opt = payload.data.get("opt_soc", {})
        self.res_full = payload.data.get("full_soc", {})
        self.meta = payload.meta
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10) # Rozestup mezi summary a grafem
        
        if not self.res_opt.get("success", False):
            error_lbl = QLabel("Optimization failed for these specific inputs. Try different values.")
            error_lbl.setStyleSheet("color: #ef4444; font-size: 16px; font-weight: bold;")
            main_layout.addWidget(error_lbl)
            return

        # --- 1. Summary ---
        summary_container = QWidget()
        summary_container.setStyleSheet("""
            background-color: #f1f5f9; 
            border: 1px solid #cbd5e1; 
            border-radius: 8px;
        """)
        summary_layout = QHBoxLayout(summary_container)
        summary_layout.setContentsMargins(20, 20, 20, 20)
        
        # ... (výpočet total_time, points atd. zůstává stejný jako ve tvém kódu) ...
        total_time = self.res_opt['total_time']
        points = self.res_opt['points']
        total_energy = sum(self.res_opt['e_targets'])
        soc_start = self.res_opt['soc_start']
        has_full_soc = self.res_full.get("success", False)
        
        if has_full_soc:
            time_saved = self.res_full['total_time'] - total_time
            points_diff = points - self.res_full['points']
            energy_diff = total_energy - sum(self.res_full['e_targets'])
            c_time = "#10b981" if time_saved >= 0 else "#ef4444"
            c_pts = "#10b981" if points_diff >= 0 else "#ef4444"
            c_eng = "#10b981" if energy_diff <= 0 else "#ef4444"
            
            info_text = (
                f"<span style='color:#3b82f6; font-size: 18px;'><b>Opt. SoC Start:</b> {soc_start * 100:.1f} %</span>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                f"<span style='color:#64748b;'><b>Points:</b></span> <span style='color:#0f172a; font-size: 16px;'>{points:.2f}</span> "
                f"(<span style='color:{c_pts}; font-weight:bold;'>{'+' if points_diff > 0 else ''}{points_diff:.2f}</span>)"
                f"&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;"
                f"<span style='color:#64748b;'><b>Energy:</b></span> <span style='color:#0f172a; font-size: 16px;'>{total_energy:.3f} kWh</span> "
                f"(<span style='color:{c_eng}; font-weight:bold;'>{'+' if energy_diff > 0 else ''}{energy_diff:.3f}</span>)"
                f"&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;"
                f"<span style='color:#64748b;'><b>Total Time:</b></span> <span style='color:#0f172a; font-size: 16px;'>{total_time:.2f} s</span> "
                f"(<span style='color:{c_time}; font-weight:bold;'>{'-' if time_saved > 0 else '+'}{abs(time_saved):.2f} s</span>)"
            )
        else:
            info_text = (
                f"<span style='color:#3b82f6; font-size: 18px;'><b>Opt. SoC Start:</b> {soc_start * 100:.1f} %</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Points:</b> {points:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Energy used:</b> {total_energy:.3f} kWh &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Time:</b> {total_time:.2f} s"
            )
        
        lbl_summary = QLabel(info_text)
        lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_summary.setStyleSheet("background: transparent;")
        summary_layout.addWidget(lbl_summary)
        
        main_layout.addWidget(summary_container)
        
        # --- 2. Matplotlib grafy ---
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#ffffff') 
        self.canvas = FigureCanvasQTAgg(self.fig)
        # Toolbar jsme odstranili
        
        graph_container = QWidget()
        graph_container.setObjectName("CanvasContainer")
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(5, 5, 5, 5)
        graph_layout.addWidget(self.canvas)
        
        main_layout.addWidget(graph_container)
        self.plot_results()

    def plot_results(self):
        self.fig.clear()
        laps = np.arange(1, len(self.res_opt['lap_times']) + 1)
        opt_lap_times = np.array(self.res_opt['lap_times'])
        opt_e_targets = np.array(self.res_opt['e_targets'])
        
        has_full_soc = self.res_full.get("success", False)
        
        # --- LAP TIMES ---
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax1.plot(laps, opt_lap_times, marker='o', color='#3b82f6', linewidth=2.5, markersize=7, label='Optimal SoC Start')
        
        if has_full_soc:
            full_lap_times = np.array(self.res_full['lap_times'])
            ax1.plot(laps, full_lap_times, marker='x', color='#94a3b8', linewidth=2, linestyle='--', label='100% SoC Start')
        
        ax1.set_ylabel('Lap time [s]', color='#475569')
        ax1.tick_params(axis='y', labelcolor='#475569')
        ax1.tick_params(axis='x', labelcolor='#475569')
        ax1.set_xticks(laps)
        ax1.grid(True, linestyle='--', alpha=0.3)
        # Bold odstraněn, barva zjemněna
        ax1.set_title('Lap Times Evolution', color='#475569', fontweight='normal', pad=15)
        ax1.legend(frameon=True, edgecolor='#e2e8f0', loc='best')

        # --- ENERGY ---
        ax2 = self.fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(laps, opt_e_targets, marker='s', color='#10b981', linewidth=2.5, markersize=6, label='Optimal SoC Start')
        
        if has_full_soc:
            full_e_targets = np.array(self.res_full['e_targets'])
            ax2.plot(laps, full_e_targets, marker='^', color='#cbd5e1', linewidth=2, linestyle='--', label='100% SoC Start')
        
        ax2.set_xlabel('Lap number', color='#475569', fontweight='normal')
        ax2.set_ylabel('Energy target [kWh]', color='#475569')
        ax2.tick_params(axis='y', labelcolor='#475569')
        ax2.tick_params(axis='x', labelcolor='#475569')
        ax2.grid(True, linestyle='--', alpha=0.3)
        # Bold odstraněn, barva zjemněna
        ax2.set_title('Energy Deployment per Lap', color='#475569', fontweight='normal', pad=15)
        ax2.legend(frameon=True, edgecolor='#e2e8f0', loc='best')

        # Klíč k úspěchu: tight_layout s h_pad definuje vertikální mezery mezi grafy
        self.fig.tight_layout(pad=3.0, h_pad=4.0) 
        self.canvas.draw()


class OptimalEnergyMapTab(QWidget):
    def __init__(self, solver_res, parent=None):
        super().__init__(parent)
        self.meta = solver_res.meta
        
        self.e_interpolant = self.meta['e_interpolant']
        self.soc_interpolant = self.meta['soc_interpolant']
        self.grid_regen = self.meta['grid_regen']
        self.grid_delta = self.meta['grid_delta']
        self.z_min = self.meta['z_min_e']
        self.z_max = self.meta['z_max_e']
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # --- Matplotlib Canvas ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
       #elf.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # --- Controls (Postranní panel vlevo) ---
        control_panel = QWidget()
        control_panel.setFixedWidth(260) # Pevná šířka, zbytek se dá grafu
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 10, 15, 10)
        
        # Výpis (Moderní karta)
        self.lbl_opt_energy = QLabel("Optimal energy:\n-- kWh\n\nOptimal SoC start:\n-- %")
        self.lbl_opt_energy.setStyleSheet("""
            font-size: 15px; 
            color: #0f172a; 
            background-color: #f1f5f9; 
            padding: 15px; 
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        """)
        self.lbl_opt_energy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Vertikální slidery vedle sebe
        sliders_layout = QHBoxLayout()
        
        # Regen Slider
        regen_vbox = QVBoxLayout()
        self.lbl_regen = QLabel("Regen:\n0 %")
        self.lbl_regen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_regen.setStyleSheet("font-weight: bold; color: #475569;")
        
        self.slider_regen = QSlider(Qt.Orientation.Vertical)
        self.slider_regen.setMinimum(int(min(self.grid_regen) * 100))
        self.slider_regen.setMaximum(100)
        self.slider_regen.setValue(int(max(self.grid_regen) * 50))
        self.slider_regen.setTickPosition(QSlider.TickPosition.TicksLeft)
        
        regen_vbox.addWidget(self.lbl_regen)
        regen_vbox.addWidget(self.slider_regen, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Delta Slider
        delta_vbox = QVBoxLayout()
        self.lbl_delta = QLabel("Delta:\n0 %")
        self.lbl_delta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_delta.setStyleSheet("font-weight: bold; color: #475569;")
        
        self.slider_delta = QSlider(Qt.Orientation.Vertical)
        self.slider_delta.setMinimum(int(min(self.grid_delta) * 100))
        self.slider_delta.setMaximum(int(max(self.grid_delta) * 100))
        self.slider_delta.setValue(0)
        self.slider_delta.setTickPosition(QSlider.TickPosition.TicksRight)
        
        delta_vbox.addWidget(self.lbl_delta)
        delta_vbox.addWidget(self.slider_delta, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        sliders_layout.addLayout(regen_vbox)
        sliders_layout.addLayout(delta_vbox)
        
        control_layout.addWidget(self.lbl_opt_energy)
        control_layout.addSpacing(20)
        control_layout.addLayout(sliders_layout)
        
        # --- Hlavní sestavení kontejneru s grafem ---
        graph_container = QWidget()
        graph_container.setObjectName("CanvasContainer")
        graph_main_layout = QVBoxLayout(graph_container)
        
        #raph_main_layout.addWidget(self.toolbar)
        
        # Rozdělení: Levý panel s ovládáním, pravá část obří graf
        content_layout = QHBoxLayout()
        content_layout.addWidget(control_panel)
        content_layout.addWidget(self.canvas, stretch=1) 
        
        graph_main_layout.addLayout(content_layout)
        main_layout.addWidget(graph_container)
        
        self.ax3d = None
        self.view_elev = 20  
        self.view_azim = 38
        self.point_plot = None 
        
        self.plot_optimal_surface()
        
        self.slider_regen.valueChanged.connect(self.update_point)
        self.slider_delta.valueChanged.connect(self.update_point)
        
        self.update_point()

    def plot_optimal_surface(self):
        r_fine = np.linspace(min(self.grid_regen), max(self.grid_regen), 50)
        d_fine = np.linspace(min(self.grid_delta), max(self.grid_delta), 50)

        R_mesh, D_mesh = np.meshgrid(r_fine, d_fine, indexing='ij')
        pts_scipy = np.column_stack((R_mesh.ravel(), D_mesh.ravel()))
        
        Z_opt_energy = self.e_interpolant(pts_scipy).reshape(R_mesh.shape)
        
        self.fig.clear()
        self.ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
        
        surf = self.ax3d.plot_surface(
            R_mesh * 100, D_mesh * 100, Z_opt_energy, 
            cmap='plasma', edgecolor='none', alpha=0.9,
            vmin=self.z_min, vmax=self.z_max
        )
        
        self.ax3d.set_xlabel('Recuperation [%]', color='#475569')
        self.ax3d.set_ylabel('Time delta [%]', color='#475569')
        self.ax3d.set_zlabel('Energy target [kWh]', color='#475569')

        
        self.ax3d.xaxis.set_tick_params(colors='#64748b')
        self.ax3d.yaxis.set_tick_params(colors='#64748b')
        self.ax3d.zaxis.set_tick_params(colors='#64748b')

        self.ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Agresivnější zmenšení okrajů Matplotlibu pro větší využití plochy
        self.fig.subplots_adjust(left=0.0, right=0.95, bottom=0.05, top=0.95)

    def update_point(self):
        regen_val = self.slider_regen.value() / 100.0
        delta_val = self.slider_delta.value() / 100.0
        
        # Přidány nové řádky (\n) aby text nad vertikálním sliderem vypadal dobře
        self.lbl_regen.setText(f"Regen:\n{regen_val*100:.0f} %")
        self.lbl_delta.setText(f"Delta:\n{delta_val*100:.0f} %")

        opt_e = float(self.e_interpolant([regen_val, delta_val])[0])
        opt_soc = float(self.soc_interpolant([regen_val, delta_val])[0])
        
        self.lbl_opt_energy.setText(
            f"<div style='font-size: 13px; color: #64748b; text-transform: uppercase;'>Optimal Energy</div>"
            f"<div style='font-size: 26px; color: #10b981; font-weight: bold;'>{opt_e:.3f} kWh</div>"
            f"<hr style='border: 1px solid #cbd5e1; margin: 10px 0;'>"
            f"<div style='font-size: 13px; color: #64748b; text-transform: uppercase;'>Optimal SoC Start</div>"
            f"<div style='font-size: 26px; color: #3b82f6; font-weight: bold;'>{opt_soc*100:.1f} %</div>"
        )

        if self.point_plot is not None:
            self.point_plot.remove()
            self.point_plot = None
            
        if self.ax3d is not None and not np.isnan(opt_e):
            self.point_plot, = self.ax3d.plot(
                [regen_val * 100], [delta_val * 100], [opt_e],
                marker='o', color='#0f172a', markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=10
            )
            
            self.canvas.draw_idle()