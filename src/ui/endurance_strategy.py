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
from scipy.ndimage import median_filter

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
    deployment_requested = pyqtSignal(float, float, object)

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Endurance strategy")
    
        self.loading_overlay = LoadingOverlay(self)
        self.map_data = self.load_data(filepath)
        main_layout = QVBoxLayout(self)

        self.showMaximized()
        
        control_group = QGroupBox("Optimization settings")
        control_layout = QHBoxLayout()
        
        # Button
        self.btn_optimize = QPushButton("Calculate opt energy map")
        self.btn_optimize.setStyleSheet("font-weight: bold; background-color: #0d16d1; color: white; padding: 5px;")
        self.btn_optimize.clicked.connect(self.run_optimization)

        self.btn_deploy = QPushButton("Calculate opt deployment for given input")
        self.btn_deploy.setStyleSheet("font-weight: bold; background-color: #2ca02c; color: white; padding: 5px;")
        self.btn_deploy.clicked.connect(self.run_deployment)
        
        # Inputs
        lbl_regen = QLabel("Recuperation [%]:")
        self.spin_regen = QDoubleSpinBox()
        self.spin_regen.setRange(0.0, 94)
        self.spin_regen.setSingleStep(1)
        self.spin_regen.setValue(85) 
        
        lbl_delta = QLabel("Time delta [%]:")
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
            
            # --- ZDE ZAČÍNÁ FILTRACE OUTLIERŮ ---
            
            # Vytvoříme vyhlazenou mapu pomocí okna 3x3x3
            # Parametr 'size=3' znamená, že se průměruje buňka a její bezprostřední sousedé
            smoothed_3d = median_filter(lap_times_3d, size=3)
            
            # Vypočítáme absolutní rozdíl mezi původními daty a mediánem okolí
            diff = np.abs(lap_times_3d - smoothed_3d)
            
            # Nastavíme práh (threshold). 
            # Pokud se bod liší od svého okolí o více než 5 sekund, považujeme ho za selhání solveru.
            outlier_threshold = 5.0 
            
            outliers_mask = diff > outlier_threshold
            num_outliers = np.sum(outliers_mask)
            
            if num_outliers > 0:
                print(f"[Zpracování dat] Nalezeno a opraveno {num_outliers} outlierů pomocí mediánového filtru.")
                # Přepíšeme nesmyslné hodnoty hodnotami z vyhlazené matice
                lap_times_3d[outliers_mask] = smoothed_3d[outliers_mask]
            # --- KONEC FILTRACE ---
            
            # Pokračujeme v původním kódu - připravíme pole pro CasADi
            flat_for_casadi = lap_times_3d.ravel(order='F')
            
            grid_axes = [soc_targets.tolist(), e_targets.tolist(), regen_targets.tolist()]
            interpolant = ca.interpolant('laptime_map', 'bspline', grid_axes, flat_for_casadi.tolist())
            
            e_fine = np.linspace(0.17, max(e_targets), 50) 
            r_fine = np.linspace(min(regen_targets), max(regen_targets), 50)
            soc_fine = np.linspace(min(soc_targets), max(soc_targets), 50)
            
            return {
                'interpolant': interpolant,
                'e_fine': e_fine,
                'r_fine': r_fine,
                'soc_fine': soc_fine,
                'r_min': min(regen_targets),
                'r_max': max(regen_targets),
                'z_min': np.nanmin(lap_times_3d), # Použijeme už vyfiltrovanou matici pro min/max
                'z_max': np.nanmax(lap_times_3d)  # Jinak by se ti zde uložilo těch chybných 120+ s
            }
        except Exception as e:
            QMessageBox.critical(self, "Chyba načítání", f"Nepodařilo se načíst mapu z {filepath}:\n{e}")
            return None
        
 
    def resizeEvent(self, event):
        self.loading_overlay.resize(self.size())
        super().resizeEvent(event)


    def add_result_tab(self, payload):
        """
        Zpracovává payloady z workeru. Rozlišuje, zda jde o celou 3D mapu, nebo jen deployment.
        """
        # --- Zpracování 3D Mapy (Tlačítko 1) ---
        if payload.kind == "endurance_map": 
            has_opt_energy_tab = False
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Optimal energy map":
                    has_opt_energy_tab = True
                    self.tabs.setCurrentIndex(i) # Pokud existuje, jen se na ni přepneme
                    
            if not has_opt_energy_tab:
                opt_energy_tab = OptimalEnergyMapTab(payload, self)
                self.tabs.addTab(opt_energy_tab, "Optimal energy map")
                self.tabs.setCurrentWidget(opt_energy_tab)

        # --- Zpracování konkrétního deploymentu (Tlačítko 2) ---
        elif payload.kind == "endurance_deployment":
            result_count = sum(1 for i in range(self.tabs.count()) if self.tabs.tabText(i).startswith("Deployment #"))
            
            result_tab = StrategyResultTab(payload, self)
            tab_name = f"Deployment #{result_count + 1}"
            self.tabs.addTab(result_tab, tab_name)
            self.tabs.setCurrentWidget(result_tab)

    def run_optimization(self):
      
        # 1. Nejdříve zkontrolujeme, zda už tab "Optimal energy map" existuje
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Optimal energy map":
                # Tab existuje! Jen se na něj přepneme a funkci rovnou ukončíme (zabráníme spuštění simulace)
                self.tabs.setCurrentIndex(i)
                return
            
        regen_val = self.spin_regen.value()/100
        delta_val = self.spin_delta.value()/100
        self.loading_overlay.show_loading("Calculating optimal strategy...", indefinite=True,progress_text_fmt="Solving grid points ({} / {})")
        self.optimization_requested.emit(regen_val, delta_val,self.map_data)

    def run_deployment(self):
        regen_val = self.spin_regen.value()/100
        delta_val = self.spin_delta.value()/100
        # Zde použijeme točící se GIF, protože jde jen o jeden konkrétní bod, takže to bude hned
        self.loading_overlay.show_loading("Calculating optimum energy deployment for given input", indefinite=True)
        self.deployment_requested.emit(regen_val, delta_val, self.map_data)
        
        

    def stop_loading(self):
        self.loading_overlay.hide_loading()

class StrategyResultTab(QWidget):
    def __init__(self, payload, parent=None):
        super().__init__(parent)
        
        # Rozbalení obou výsledků ze solveru
        self.res_opt = payload.data.get("opt_soc", {})
        self.res_full = payload.data.get("full_soc", {})
        self.meta = payload.meta
        
        main_layout = QVBoxLayout(self)
        
        # Kontrola, zda prošla alespoň ta hlavní optimalizace
        if not self.res_opt.get("success", False):
            error_lbl = QLabel("Optimization failed for these specific inputs. Try different values.")
            error_lbl.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            main_layout.addWidget(error_lbl)
            return

        # --- 1. Výpočty pro textové shrnutí ---
        summary_layout = QHBoxLayout()
        
        total_time = self.res_opt['total_time']
        points = self.res_opt['points']
        total_energy = sum(self.res_opt['e_targets'])
        soc_start = self.res_opt['soc_start']
        points = self.res_opt['points']
        total_energy = sum(self.res_opt['e_targets'])

        has_full_soc = self.res_full.get("success", False)
        
        if has_full_soc:
            # Rozdíly: Opt - Full
            time_saved = self.res_full['total_time'] - total_time   # Kladné = ušetřili jsme čas
            points_diff = points - self.res_full['points']          # Kladné = získali jsme body
            energy_diff = total_energy - sum(self.res_full['e_targets']) # Záporné = spálili jsme méně energie
            
            # Nastavení barev (Zelená pro zlepšení, Červená pro zhoršení)
            c_time = "#2ca02c" if time_saved >= 0 else "#d62728"
            c_pts = "#2ca02c" if points_diff >= 0 else "#d62728"
            c_eng = "#2ca02c" if energy_diff <= 0 else "#d62728" # U energie je menší = lepší
            
            info_text = (
                f"<span style='color:#1f77b4;'><b>Opt. SoC Start:</b> {soc_start * 100:.1f} %</span> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; "
                f"<b>Points:</b> {points:.2f} (<span style='color:{c_pts};'>{'+' if points_diff > 0 else ''}{points_diff:.2f}</span>) &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Energy:</b> {total_energy:.3f} kWh (<span style='color:{c_eng};'>{'+' if energy_diff > 0 else ''}{energy_diff:.3f}</span>) &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Time:</b> {total_time:.2f} s (<span style='color:{c_time};'>{'-' if time_saved > 0 else '+'}{abs(time_saved):.2f} s</span>)"
            )
        else:
            # Fallback text, kdyby srovnávací "Full SoC" solver z nějakého důvodu selhal
            info_text = (
                f"<span style='color:#1f77b4;'><b>Opt. SoC Start:</b> {soc_start * 100:.1f} %</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Total points:</b> {points:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Total Energy used:</b> {total_energy:.3f} kWh &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Opt. total time:</b> {total_time:.2f} s"
            )
        
        lbl_summary = QLabel(info_text)
        lbl_summary.setStyleSheet("""
            font-size: 15px; 
            padding: 12px; 
            background-color: #05001f; 
            color: white;
            border: 1px solid #444;
            border-radius: 6px;
        """)
        lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        summary_layout.addWidget(lbl_summary)
        main_layout.addLayout(summary_layout)
        
        # --- 2. Matplotlib grafy ---
        self.fig = Figure(figsize=(10, 8), dpi=100) 
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.plot_results()

    def plot_results(self):
        self.fig.clear()
        
        laps = np.arange(1, len(self.res_opt['lap_times']) + 1)
        
        # Data z optimální strategie
        opt_lap_times = np.array(self.res_opt['lap_times'])
        opt_e_targets = np.array(self.res_opt['e_targets'])
        
        # Ověříme, jestli máme data i z porovnávací "Full SoC" simulace
        has_full_soc = self.res_full.get("success", False)
        if has_full_soc:
            full_lap_times = np.array(self.res_full['lap_times'])
            full_e_targets = np.array(self.res_full['e_targets'])

        # ==========================================
        # GRAF 1: Lap Times (Nahoře)
        # ==========================================
        ax1 = self.fig.add_subplot(2, 1, 1)
        
        ax1.plot(laps, opt_lap_times, marker='o', color='#d62728', linewidth=2.5, label='Opt. SoC Start')
        
        if has_full_soc:
            ax1.plot(laps, full_lap_times, marker='x', color='gray', linewidth=2, linestyle='--', label='100% SoC Start')
        
        ax1.set_ylabel('Lap time [s]', color='#d62728')
        ax1.tick_params(axis='y', labelcolor='#d62728')
        ax1.set_xticks(laps)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title('Laptimes')
        ax1.legend()

        # ==========================================
        # GRAF 2: Energy Target (Dole)
        # ==========================================
        ax2 = self.fig.add_subplot(2, 1, 2, sharex=ax1)
        
        ax2.plot(laps, opt_e_targets, marker='s', color='#1f77b4', linewidth=2.5, markersize=6, label='Opt. SoC Start')
        
        if has_full_soc:
            ax2.plot(laps, full_e_targets, marker='^', color='lightblue', linewidth=2, linestyle='--', label='100% SoC Start')
        
        ax2.set_xlabel('Lap number', fontweight='bold')
        ax2.set_ylabel('Energy target [kWh]', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_title('Energy deployment strategy')
        ax2.legend()

        self.fig.tight_layout()
        self.canvas.draw()


class OptimalEnergyMapTab(QWidget):
    def __init__(self, solver_res, parent=None):
        super().__init__(parent)
        self.meta = solver_res.meta
        
        # Načteme NAŠE NOVÉ interpolanty a limity z payloadu
        self.e_interpolant = self.meta['e_interpolant']
        self.soc_interpolant = self.meta['soc_interpolant']
        self.grid_regen = self.meta['grid_regen']
        self.grid_delta = self.meta['grid_delta']
        self.z_min = self.meta['z_min_e']
        self.z_max = self.meta['z_max_e']
        
        # --- UI Layout (Stejný jako minule) ---
        main_layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        control_layout = QHBoxLayout()
        sliders_vbox = QVBoxLayout()
        
        # Slidery
        regen_layout = QHBoxLayout()
        self.lbl_regen = QLabel("Regen: 0 %")
        self.lbl_regen.setFixedWidth(100)
        self.slider_regen = QSlider(Qt.Orientation.Horizontal)
        self.slider_regen.setMinimum(int(min(self.grid_regen) * 100))
        self.slider_regen.setMaximum(int(max(self.grid_regen) * 100))
        self.slider_regen.setValue(int(max(self.grid_regen) * 50))
        regen_layout.addWidget(self.lbl_regen)
        regen_layout.addWidget(self.slider_regen)
        
        delta_layout = QHBoxLayout()
        self.lbl_delta = QLabel("Delta: 0 %")
        self.lbl_delta.setFixedWidth(100)
        self.slider_delta = QSlider(Qt.Orientation.Horizontal)
        self.slider_delta.setMinimum(int(min(self.grid_delta) * 100))
        self.slider_delta.setMaximum(int(max(self.grid_delta) * 100))
        self.slider_delta.setValue(0)
        delta_layout.addWidget(self.lbl_delta)
        delta_layout.addWidget(self.slider_delta)
        
        sliders_vbox.addLayout(regen_layout)
        sliders_vbox.addLayout(delta_layout)
        
        # ZMĚNA: Přidán prostor pro výpis Optimálního SoC!
        self.lbl_opt_energy = QLabel("Optimal Energy:\n-- kWh\n\nOptimal SoC Start:\n-- %")
        self.lbl_opt_energy.setStyleSheet("""
            font-size: 15px; 
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
        
        main_layout.addWidget(self.toolbar)
        main_layout.addLayout(control_layout) 
        main_layout.addWidget(self.canvas, stretch=1) 
        
        self.ax3d = None
        self.view_elev = 20  
        self.view_azim = 38
        self.point_plot = None 
        
        self.plot_optimal_surface()
        
        self.slider_regen.valueChanged.connect(self.update_point)
        self.slider_delta.valueChanged.connect(self.update_point)
        
        self.update_point()

    def plot_optimal_surface(self):
        # Nyní tvoříme jen 2D MeshGrid (Regen a Delta)
        r_fine = np.linspace(min(self.grid_regen), max(self.grid_regen), 50)
        d_fine = np.linspace(min(self.grid_delta), max(self.grid_delta), 50)

        R_mesh, D_mesh = np.meshgrid(r_fine, d_fine, indexing='ij')
        pts_scipy = np.column_stack((R_mesh.ravel(), D_mesh.ravel()))
        
        # Místo argmax rovnou vyhodnotíme e_interpolant
        Z_opt_energy = self.e_interpolant(pts_scipy).reshape(R_mesh.shape)
        
        self.fig.clear()
        self.ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
        
        surf = self.ax3d.plot_surface(
            R_mesh * 100, D_mesh * 100, Z_opt_energy, 
            cmap='plasma', edgecolor='none', alpha=0.9,
            vmin=self.z_min, vmax=self.z_max
        )
        
        self.ax3d.set_xlabel('Recuperation percentage [%]')
        self.ax3d.set_ylabel('Time delta [%]')
        self.ax3d.set_zlabel('Opt energy target [kWh]')
        self.ax3d.set_title('Optimal energy target')
        
        self.fig.colorbar(surf, ax=self.ax3d, shrink=0.5, aspect=10, label='Energy target [kWh]')
        self.ax3d.view_init(elev=self.view_elev, azim=self.view_azim)
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

    def update_point(self):
        regen_val = self.slider_regen.value() / 100.0
        delta_val = self.slider_delta.value() / 100.0
        
        self.lbl_regen.setText(f"Regen: {regen_val*100:.0f} %")
        self.lbl_delta.setText(f"Delta: {delta_val*100:.0f} %")

        # ROVNOU přečteme hodnoty pro zadaný regen a deltu ze 2D interpolantů
        # ([0] je nutné, protože interpolant vrací pole výsledků o délce 1)
        opt_e = float(self.e_interpolant([regen_val, delta_val])[0])
        opt_soc = float(self.soc_interpolant([regen_val, delta_val])[0])
        
        # Aktualizace textu s OBĚMA proměnnými
        self.lbl_opt_energy.setText(f"Optimal Energy:\n{opt_e:.3f} kWh\n\nOptimal SoC Start:\n{opt_soc*100:.1f} %")

        if self.point_plot is not None:
            self.point_plot.remove()
            self.point_plot = None
            
        if self.ax3d is not None and not np.isnan(opt_e):
            self.point_plot, = self.ax3d.plot(
                [regen_val * 100], [delta_val * 100], [opt_e],
                marker='o', color='cyan', markersize=12, markeredgecolor='black', zorder=10
            )
            
            self.canvas.draw_idle()