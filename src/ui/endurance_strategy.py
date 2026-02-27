import numpy as np
import casadi as ca
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QMessageBox, QPushButton, 
                             QDoubleSpinBox, QTabWidget, QGroupBox, QApplication,QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from src.ui.loader import LoadingOverlay


class MapTab(QWidget):
 
    def __init__(self, map_data, fixed_axis='soc', parent=None):
        super().__init__(parent)
        #self.setStyleSheet("background-color: white; color: black;")
        self.map_data = map_data
        self.fixed_axis = fixed_axis
        
        
        self.laptime_map = map_data['interpolant']
        self.z_min = map_data['z_min']
        self.z_max = map_data['z_max']
        
        # --- UI Layout ---
        main_layout = QVBoxLayout(self) 
        
        # Matplotlib Canvas 
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
                
        # Horizontal layout
        content_layout = QHBoxLayout()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self.canvas, stretch=1)
        
        # --- Slider 
        slider_layout = QVBoxLayout() 
        self.lbl_slider = QLabel("")
        self.lbl_slider.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        # Vertical slider
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
            self.slider.setMaximum(int(map_data['r_max'] * 100))
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
            self.lbl_slider.setText(f"SoC: {val} %")
        else:
            self.lbl_slider.setText(f"Regen: {val/100.0:.2f}")
        self.update_plot()

    def update_plot(self):
        if self.fixed_axis == 'soc':
            soc_val = self.slider.value() / 100.0
            # [SoC, Energy, Regen]
            pts_casadi = np.vstack((
                np.full_like(self.X_mesh.ravel(), soc_val),
                self.X_mesh.ravel(),
                self.Y_mesh.ravel()
            ))
            x_label, y_label = 'Energy target [kWh]', 'Regen [%]'
            title = f'Laptime map (SoC = {soc_val*100:.0f} %)'
        else:
            regen_val = self.slider.value() / 100.0
            # [SoC, Energy, Regen]
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
        self.ax3d.set_xlabel(x_label)
        self.ax3d.set_ylabel(y_label)
        self.ax3d.set_zlabel('Lap time [s]')
        self.ax3d.set_title(title)
        
        #self.fig.colorbar(surf, ax=self.ax3d, shrink=0.5, aspect=10, label='Lap time [s]')
        self.ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)
        self.canvas.draw_idle()


class LaptimeAppMainWindow(QWidget):
    optimization_requested = pyqtSignal(float, float, object)

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Endurance strategy")
        self.showMaximized()

        self.loading_overlay = LoadingOverlay(self)

                
        self.map_data = self.load_data(filepath)
        
        
        main_layout = QVBoxLayout(self)
        
        
        control_group = QGroupBox("Optimization settings")
        control_layout = QHBoxLayout()
        
        # Button
        self.btn_optimize = QPushButton("Optimize strategy")
        self.btn_optimize.setStyleSheet("font-weight: bold; background-color: #0d16d1; color: white; padding: 5px;")
        self.btn_optimize.clicked.connect(self.run_optimization)
        
        # Inputs
        lbl_regen = QLabel("Recuperation [%]:")
        self.spin_regen = QDoubleSpinBox()
        self.spin_regen.setRange(0.0, 94)
        self.spin_regen.setSingleStep(1)
        self.spin_regen.setValue(85) 
        
        lbl_delta = QLabel("Time delta [%]:")
        self.spin_delta = QDoubleSpinBox()
        self.spin_delta.setRange(0, 20.0)
        self.spin_delta.setSingleStep(0.5)
        self.spin_delta.setValue(0.0)

        
        control_layout.addWidget(self.btn_optimize)
        control_layout.addStretch() 
        control_layout.addWidget(lbl_regen)
        control_layout.addWidget(self.spin_regen)
        control_layout.addSpacing(20)
        control_layout.addWidget(lbl_delta)
        control_layout.addWidget(self.spin_delta)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
     
        self.tabs = QTabWidget()
        
        if self.map_data:
            #creting tabs
            self.tab_soc = MapTab(self.map_data, fixed_axis='soc')
            self.tab_regen = MapTab(self.map_data, fixed_axis='regen')
            
            side_by_side_widget = QWidget()
            side_by_side_widget.setStyleSheet("background-color: white; color: black;")
            side_by_side_layout = QHBoxLayout(side_by_side_widget)
            side_by_side_layout.addWidget(self.tab_soc)
            side_by_side_layout.addWidget(self.tab_regen)
            
            
            self.tabs.addTab(side_by_side_widget, "Laptime maps")

        else:
            self.tabs.addTab(QLabel("Data se nepodařilo načíst."), "Error")
            
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
            flat_for_casadi = lap_times_3d.ravel(order='F')
            
            grid_axes = [soc_targets.tolist(), e_targets.tolist(), regen_targets.tolist()]
            interpolant = ca.interpolant('laptime_map', 'bspline', grid_axes, flat_for_casadi.tolist())
            
            
            e_fine = np.linspace(min(e_targets), 0.3, 50) 
            r_fine = np.linspace(min(regen_targets), max(regen_targets), 50)
            soc_fine = np.linspace(min(soc_targets), max(soc_targets), 50)
            
            return {
                'interpolant': interpolant,
                'e_fine': e_fine,
                'r_fine': r_fine,
                'soc_fine': soc_fine,
                'r_min': min(regen_targets),
                'r_max': max(regen_targets),
                'z_min': np.nanmin(flat_lap_times),
                'z_max': np.nanmax(flat_lap_times)
            }
        except Exception as e:
            QMessageBox.critical(self, "Chyba načítání", f"Nepodařilo se načíst mapu z {filepath}:\n{e}")
            return None
        
 
    def resizeEvent(self, event):
        self.loading_overlay.resize(self.size())
        super().resizeEvent(event)

    def add_result_tab(self, payload):
        """
        Tato metoda vezme payload (výsledky solveru) a vytvoří pro ně nový tab.
        """

        has_opt_energy_tab = False
        result_count = 0
        
        #get existing tabs
        for i in range(self.tabs.count()):
            text = self.tabs.tabText(i)
            if text == "Optimal energy":
                has_opt_energy_tab = True
            elif text.startswith("Result #"):
                result_count += 1

        # add optimal energy tab only if it doesnt exist yet
        if not has_opt_energy_tab:
            opt_energy_tab = OptimalEnergyMapTab(payload, self)
            self.tabs.addTab(opt_energy_tab, "Optimal energy")

        # add results tab
        result_tab = StrategyResultTab(payload, self)
        tab_name = f"Result #{result_count + 1}"
        self.tabs.addTab(result_tab, tab_name)
        
       
        self.tabs.setCurrentWidget(opt_energy_tab)

    def run_optimization(self):
      
        regen_val = self.spin_regen.value()/100
        delta_val = self.spin_delta.value()/100
        self.loading_overlay.show_loading("Calculating optimal strategy...", indefinite=True)
        self.optimization_requested.emit(regen_val, delta_val,self.map_data)
        
        
    def stop_loading(self):
        self.loading_overlay.hide_loading()

class StrategyResultTab(QWidget):
    def __init__(self, solver_res, parent=None):
        super().__init__(parent)
        self.res = solver_res['data'] 
        self.meta = solver_res['meta']
        
        main_layout = QVBoxLayout(self)
        
        if not self.res.get("success", False):
            error_lbl = QLabel("Optimalization failed, try for different values")
            error_lbl.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            main_layout.addWidget(error_lbl)
            return

        # summary layout
        summary_layout = QHBoxLayout()
        
        total_time = self.res['total_time']
        points = self.res['points']
        total_energy = sum(self.res['e_targets'])
        time_saved = self.res['time_saved']
        

        info_text = (
            f"<b>Total points:</b> {points:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Energy used:</b> {total_energy:.3f} kWh &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Opt. time:</b> {total_time:.2f} s &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<span style='color:#f8f7fa;'><b>Time saved vs flat:</b> {time_saved:.2f} s</span>"
        )
        
        lbl_summary = QLabel(info_text)
        lbl_summary.setStyleSheet("font-size: 14px; padding: 10px; background-color: #05001f; border: 1px solid #dee2e6;")
        lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        summary_layout.addWidget(lbl_summary)
        main_layout.addLayout(summary_layout)
        
        # --- matplotlib graphs
        self.fig = Figure(figsize=(10, 9), dpi=100) 
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.plot_results()

    def plot_results(self):
        self.fig.clear()
        
        laps = range(1, len(self.res['lap_times']) + 1)
        
        # Lap times 
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax1.plot(laps, self.res['lap_times'], marker='o', color='#d62728', linewidth=2, label='Optimal strategy')
        ax1.plot(laps, self.res['flat_lap_times'], marker='x', color='gray', linewidth=2, linestyle='--', label='Flat distribution')
        ax1.set_ylabel('Lap time [s]')
        ax1.set_title('Lap time comparison')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xticks(laps)
        ax1.legend()

        # Energy target & SoC
        ax2 = self.fig.add_subplot(3, 1, 2, sharex=ax1)
        
        line1 = ax2.plot(laps, self.res['e_targets'], marker='s', color='#1f77b4', linewidth=2, label='Opt energy target')
        line_flat_e = ax2.plot(laps, self.res['flat_e_targets'], color='lightblue', linewidth=2, linestyle='--', label='Flat target')
        
        ax2.set_ylabel('Energy target [kWh]', color='#1f77b4', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        ax3 = ax2.twinx()
        soc_after_lap = self.res['soc_profile'][1:] 
        flat_soc_after_lap = self.res['flat_soc_profile'][1:]
        
        line2 = ax3.plot(laps, soc_after_lap, marker='^', color='#2ca02c', linewidth=2, label='Opt. SoC')
        line_flat_soc = ax3.plot(laps, flat_soc_after_lap, color='lightgreen', linewidth=2, linestyle=':', label='Flat SoC')
        
        ax3.set_ylabel('SoC [-]', color='#2ca02c')
        ax3.tick_params(axis='y', labelcolor='#2ca02c')
        
        lines = line1 + line_flat_e + line2 + line_flat_soc
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper center', ncol=2)
        ax2.set_title('Energy & SoC profile')

        #  Time Delta
        ax4 = self.fig.add_subplot(3, 1, 3, sharex=ax1)
        
        # Delta: heuristic 
        deltas = np.array(self.res['flat_lap_times']) - np.array(self.res['lap_times'])
      
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in deltas]
        
        ax4.bar(laps, deltas, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(0, color='black', linewidth=1)
        ax4.set_xlabel('Lap Number')
        ax4.set_ylabel('Time delta [s]')
        ax4.set_title('Advantage per lap (positive = opt. is faster)')
        ax4.grid(True, axis='y', linestyle='--', alpha=0.5)
        self.fig.subplots_adjust(hspace=0.4)

       # self.fig.tight_layout()
        self.canvas.draw()



class OptimalEnergyMapTab(QWidget):
    def __init__(self, solver_res, parent=None):
        super().__init__(parent)
        self.meta = solver_res['meta']
        
        # Get interpolant
        self.points_interpolant = self.meta['points_interpolant']
        self.grid_e = self.meta['grid_e']
        self.grid_regen = self.meta['grid_regen']
        self.grid_delta = self.meta['grid_delta']
        
        # UI layout 
        main_layout = QVBoxLayout(self)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        
        control_layout = QHBoxLayout()
        sliders_vbox = QVBoxLayout()
        
        # Slider Regen
        regen_layout = QHBoxLayout()
        self.lbl_regen = QLabel("Regen: 0.00")
        self.lbl_regen.setFixedWidth(100)
        self.slider_regen = QSlider(Qt.Orientation.Horizontal)
        self.slider_regen.setMinimum(int(min(self.grid_regen) * 100))
        self.slider_regen.setMaximum(int(max(self.grid_regen) * 100))
        self.slider_regen.setValue(int(max(self.grid_regen) * 50))
        regen_layout.addWidget(self.lbl_regen)
        regen_layout.addWidget(self.slider_regen)
        
        # Slider Delta
        delta_layout = QHBoxLayout()
        self.lbl_delta = QLabel("Delta: 0.00")
        self.lbl_delta.setFixedWidth(100)
        self.slider_delta = QSlider(Qt.Orientation.Horizontal)
        self.slider_delta.setMinimum(int(min(self.grid_delta) * 100))
        self.slider_delta.setMaximum(int(max(self.grid_delta) * 100))
        self.slider_delta.setValue(0)
        delta_layout.addWidget(self.lbl_delta)
        delta_layout.addWidget(self.slider_delta)
        
        sliders_vbox.addLayout(regen_layout)
        sliders_vbox.addLayout(delta_layout)
        
        # Z value
        self.lbl_opt_energy = QLabel("Optimal energy:\n-- kWh")
        self.lbl_opt_energy.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #1f77b4; 
            background-color: #f8f9fa; 
            padding: 10px; 
            border: 2px solid #1f77b4;
            border-radius: 5px;
        """)
        self.lbl_opt_energy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        control_layout.addLayout(sliders_vbox)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.lbl_opt_energy)
        
        # build main layout
        main_layout.addWidget(self.toolbar)
        main_layout.addLayout(control_layout) 
        main_layout.addWidget(self.canvas, stretch=1) # stretch my booy
        
      
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
        e_fine = np.linspace(min(self.grid_e), max(self.grid_e), 100)

        E_mesh, R_mesh, D_mesh = np.meshgrid(e_fine, r_fine, d_fine, indexing='ij')
        pts_scipy = np.column_stack((E_mesh.ravel(), R_mesh.ravel(), D_mesh.ravel()))
        
        # evaluate points for the grid
        points_eval = self.points_interpolant(pts_scipy).reshape(E_mesh.shape)
        best_e_indices = np.argmax(points_eval, axis=0) 
        Z_opt_energy = e_fine[best_e_indices] 
        
        X_plot, Y_plot = np.meshgrid(r_fine, d_fine, indexing='ij')
        
        self.fig.clear()
        self.ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
        
   
        surf = self.ax3d.plot_surface(
            X_plot*100, Y_plot*100, Z_opt_energy, 
            cmap='plasma', edgecolor='none', alpha=0.9
        )
        
        self.ax3d.set_xlabel('Recuperation percentage [%]')
        self.ax3d.set_ylabel('Time delta [%]')
        self.ax3d.set_zlabel('Opt energy target [kWh]')
        self.ax3d.set_title('Optimal energy target')
        
        self.fig.colorbar(surf, ax=self.ax3d, shrink=0.5, aspect=10, label='Energy target [kWh]')
        self.ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

    def update_point(self):
        # get input
        regen_val = self.slider_regen.value() / 100.0
        delta_val = self.slider_delta.value() / 100.0
        
       
        self.lbl_regen.setText(f"Regen: {regen_val*100:.2f}")
        self.lbl_delta.setText(f"Delta: {delta_val*100:.2f}")

      
        e_fine = np.linspace(min(self.grid_e), max(self.grid_e), 100)
        
       
        pts_eval = np.column_stack((
            e_fine,
            np.full_like(e_fine, regen_val),
            np.full_like(e_fine, delta_val)
        ))
        
        points_eval = self.points_interpolant(pts_eval)
        best_idx = np.argmax(points_eval)
        opt_e = e_fine[best_idx]
        
        self.lbl_opt_energy.setText(f"Optimal energy:\n{opt_e:.3f} kWh")


        if self.point_plot is not None:
            self.point_plot.remove()
            self.point_plot = None
            
        
        if self.ax3d is not None:
            
            self.point_plot, = self.ax3d.plot(
                [regen_val], [delta_val], [opt_e],
                marker='o', color='cyan', markersize=12, markeredgecolor='black', zorder=10
            )
            
           
            self.canvas.draw_idle()