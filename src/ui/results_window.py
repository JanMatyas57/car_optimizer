import datetime
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTabWidget, QGroupBox, QGridLayout, QFileDialog, QDoubleSpinBox,QApplication,QPushButton)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from src.utils.saver import save_results_to_npz
import os

class ResultsWindow(QWidget):
    def __init__(self, res, car, track,ds):
        super().__init__()
        self.res = res
        self.car = car
        self.track = track
        self.ds = ds

        self.resize(1100, 800)
        
        self.center_window()
        self.setWindowTitle(f"Simulation Results - {res.lap_time:.3f} s")
     
        
        #main layout
        self.layout = QVBoxLayout(self)
        
        # 1. Summary panel 
        self.init_summary_panel()
        
        # 2. Tabs with graphs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_dynamics_tab()
        self.init_powertrain_tab()
        self.init_forces_tab()
        self.init_sensitivity_tab()
        self.init_map_tab()
        self.init_drs_map_tab()
        self.init_trajectory_tab()
        self.init_soc_tab()

    def center_window(self):
        """centers where is mice"""
        # get frame geometry
        qr = self.frameGeometry()
        # get the mice position
        cp = QApplication.primaryScreen().availableGeometry().center()

        qr.moveCenter(cp)
        
        self.move(qr.topLeft())
    
        self.showNormal()

    def init_summary_panel(self):
        group = QGroupBox("Lap Summary")
        main_h_layout = QHBoxLayout() 
        
        # left part
        stats_widget = QWidget()
        grid = QGridLayout(stats_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        
        def add_stat(row, col, label, value, unit):
            lbl_name = QLabel(f"<b>{label}:</b>")
            lbl_val = QLabel(f"{value}")
            lbl_val.setStyleSheet("color: #007bff; font-size: 14px; font-weight: bold;")
            grid.addWidget(lbl_name, row, col)
            grid.addWidget(lbl_val, row, col+1)
            grid.addWidget(QLabel(unit), row, col+2)

        max_v = np.max(self.res.v) * 3.6
        avg_v = np.mean(self.res.v) * 3.6
        total_energy = self.res.E[-1]
        
        add_stat(0, 0, "Lap Time", f"{self.res.lap_time:.3f}", "s")
        add_stat(0, 3, "Energy Used", f"{total_energy:.3f}", "kWh")
        add_stat(1, 0, "Max Speed", f"{max_v:.1f}", "km/h")
        add_stat(1, 3, "Avg Speed", f"{avg_v:.1f}", "km/h")
        
        main_h_layout.addWidget(stats_widget)
        
        # Save Button and spacer
        main_h_layout.addStretch() # puts the button to right
        
        self.btn_save = QPushButton("Save Results (.npz)")
        self.btn_save.setFixedWidth(150)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8; 
                color: white; 
                font-weight: bold; 
                padding: 8px; 
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.btn_save.clicked.connect(self.on_save_clicked)
        
        main_h_layout.addWidget(self.btn_save)
        
        group.setLayout(main_h_layout)
        self.layout.addWidget(group)

    def calculate_validation_metrics(self,sim_res, ref_data):
        """
        Porovná výsledky simulace s referenčními daty.
        
        Args:
            sim_res: SolverResults (obsahuje .s, .t, .v ze solveru)
            ref_data: Dict {'s': ..., 'v': ..., 'p': ...} (Reference z Tracku)
                    Předpoklad: 'v' je v m/s, 'p' je v kW
        """
        if ref_data is None or ref_data['v'] is None:
            return None

        # 1. Získání os (Solver vs Reference)
        s_sim = sim_res.s          # Mřížka solveru (např. po 0.5m)
        s_ref = ref_data['s']      # Mřížka trati (např. z logu)
        v_ref_raw = ref_data['v']  # Rychlost na mřížce trati [m/s]
        
        # 2. DOPOČET REFERENČNÍHO ČASU (Integrace: dt = ds / v)
        # Musíme to udělat na mřížce reference, než budeme interpolovat
        ds_ref = np.diff(s_ref, prepend=0)
        
        # Ošetření dělení nulou (pokud auto v logu stálo nebo jelo super pomalu)
        # Cokoliv pod 1.0 m/s nahradíme 1.0 m/s pro výpočet času, jinak čas vyletí do nekonečna
        v_safe = np.maximum(v_ref_raw, 1.0) 
        
        dt_ref = ds_ref / v_safe
        t_ref_calculated = np.cumsum(dt_ref) # Kumulativní součet -> Časová osa reference

        # 3. INTERPOLACE NA MŘÍŽKU SOLVERU
        # Abychom mohli odečítat (Sim - Real), musí mít vektory stejnou délku
        v_ref_interp = np.interp(s_sim, s_ref, v_ref_raw)
        t_ref_interp = np.interp(s_sim, s_ref, t_ref_calculated)
        
        # Power (volitelně) - Log je v kW, Solver je ve Wattech -> Převedeme log na W nebo Sim na kW
        # Zde převedeme interp log na Watty pro konzistenci výpočtů, v grafu to pak dáme na kW
        p_ref_interp = None
        if ref_data.get('p') is not None:
            p_ref_interp = np.interp(s_sim, s_ref, ref_data['p']) * 1000.0 # kW -> W

        # 4. VÝPOČET DELT
        # Delta V: Kladné = Sim je rychlejší
        delta_v = sim_res.v - v_ref_interp 
        
        # Delta T: Kladné = Sim je pomalejší (ztrácí čas)
        # Resetujeme start na 0, abychom porovnávali čistý průjezd kola
        t_sim_zeroed = sim_res.t - sim_res.t[0]
        t_ref_zeroed = t_ref_interp - t_ref_interp[0]
        
        delta_t = t_sim_zeroed - t_ref_zeroed

        # 5. METRIKY
        rmse_v = np.sqrt(np.mean(delta_v**2))
        total_time_diff = t_sim_zeroed[-1] - t_ref_zeroed[-1]
        
        return {
            's': s_sim,
            'v_ref': v_ref_interp,   # [m/s] aligned
            't_ref': t_ref_zeroed,   # [s] aligned
            'p_ref': p_ref_interp,   # [W] aligned
            'delta_v': delta_v,      # [m/s]
            'delta_t': delta_t,      # [s]
            'rmse_v': rmse_v,
            'diff_total': total_time_diff
        }

    def on_save_clicked(self):
        """opens dialog and saves the results."""
       
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        default_name = f"{self.track.name}_{now_str}.npz"
        
        # results folder
        if not os.path.exists("results"):
            os.makedirs("results")
            
        default_path = os.path.join("results", default_name)

        # QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Results",
            default_path,
            "NumPy Zip (*.npz)"
        )

        if filepath:
            try:
                save_results_to_npz(
                    filepath, 
                    self.res, 
                    self.track, 
                    self.ds
                )
               
                self.btn_save.setText("Saved!")
                self.btn_save.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
            except Exception as e:
                print(f"Error saving file: {e}")
                self.btn_save.setText("Error!")
                self.btn_save.setStyleSheet("background-color: #dc3545; color: white;")

    def init_dynamics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # --- 1. SPEED GRAPH ---
        plot_v = pg.PlotWidget(title="Velocity Comparison")
        plot_v.setLabel('left', 'Speed', units='km/h')
        plot_v.setLabel('bottom', 'Distance', units='m')
        #plot_v.showGrid(x=True, y=True)
        plot_v.addLegend()
      
        # Simulation (Blue)
        plot_v.plot(self.res.s, self.res.v * 3.6, pen=pg.mkPen('b', width=2), name="Simulation")
        
        # Validation Logic
        val_data = None
        if self.track.ref_data is not None:
           
            val_data = self.calculate_validation_metrics(self.res, self.track.ref_data)
            
            if val_data:
                # Reference (Black Dashed)
                v_ref_kmh = val_data['v_ref'] * 3.6
                plot_v.plot(val_data['s'], v_ref_kmh, 
                           pen=pg.mkPen('k', width=2, style=Qt.PenStyle.DashLine), 
                           name=f"Real Log (RMSE: {val_data['rmse_v']*3.6:.1f} km/h)")

        layout.addWidget(plot_v)

        # --- 2. DELTA TIME GRAPH ---
        plot_dt = pg.PlotWidget(title="Delta Time (Accumulated Gap)")
        plot_dt.setLabel('left', 'Delta Time', units='s')
        plot_dt.setLabel('bottom', 'Distance', units='m')
        #plot_dt.showGrid(x=True, y=True)
        plot_dt.setXLink(plot_v)
        
        # Zero Line (Vizuální linka)
        plot_dt.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1)))

        if val_data:
            # 1. Vykreslíme křivku delty a uložíme si ji do proměnné
            dt_curve = plot_dt.plot(val_data['s'], val_data['delta_t'], pen=pg.mkPen('r', width=2))
            
            # 2. Vytvoříme pomocnou "neviditelnou" nulovou křivku pro výplň
            # (FillBetweenItem funguje nejlépe mezi dvěma PlotCurveItems)
            dummy_zero = pg.PlotCurveItem(val_data['s'], np.zeros_like(val_data['s']), pen=None)
            plot_dt.addItem(dummy_zero)
            
            # 3. Vytvoříme výplň mezi Delta křivkou a nulou
            # Používáme přímé reference 'dt_curve' a 'dummy_zero' - ŽÁDNÉ INDEXY!
            fill = pg.FillBetweenItem(dt_curve, dummy_zero, brush=(255, 0, 0, 50))
            plot_dt.addItem(fill)
            
            # Title update
            diff = val_data['diff_total']
            sign = "+" if diff > 0 else ""
            plot_dt.setTitle(f"Delta Time (Final Gap: {sign}{diff:.3f} s)")
        else:
            plot_dt.setTitle("No reference data available")

        layout.addWidget(plot_dt)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dynamics & Validation")

    def init_trajectory_tab(self):
        """
        Porovná ideální centerline (Track) a optimalizovanou stopu (Car).
        Rekonstruuje XY pozici auta pomocí referenční čáry a odchylky 'n'.
        Přidává graf steering angle.
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # --- 1. Výpočet souřadnic auta ---
        if not hasattr(self.res, 'n') or len(self.res.n) == 0:
            layout.addWidget(QLabel("Error: 'n' (lateral deviation) missing in SolverResults."))
            widget.setLayout(layout)
            self.tabs.addTab(widget, "Trajectory")
            return

        s_sim = self.res.s
        n_sim = self.res.n 
        
        # Interpolace X, Y reference na mřížku solveru
        x_ref = np.interp(s_sim, self.track.s, self.track.x)
        y_ref = np.interp(s_sim, self.track.s, self.track.y)
        
        # Výpočet heading reference
        dx = np.gradient(x_ref)
        dy = np.gradient(y_ref)
        psi_ref = np.arctan2(dy, dx)
        
        # Rekonstrukce pozice auta
        x_car = x_ref - n_sim * np.sin(psi_ref)
        y_car = y_ref + n_sim * np.cos(psi_ref)

        # --- 2. Grafy ---
        
        # A) Lateral Deviation Plot (n)
        plot_dev = pg.PlotWidget(title="Lateral deviation (n)")
        plot_dev.setLabel('left', 'Deviation', units='m')
        plot_dev.setLabel('bottom', 'Distance', units='m')
        plot_dev.showGrid(x=True, y=True)
        
        # Centerline (0) a Tunel
        plot_dev.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine)))    
        plot_dev.plot(s_sim, n_sim, pen=pg.mkPen('b', width=2), name="Deviation")

        # B) Steering Angle Plot (Delta)
        plot_steer = pg.PlotWidget(title="Wheel angle")
        plot_steer.setLabel('left', 'Steer', units='deg') # Lepší čitelnost ve stupních
        plot_steer.setLabel('bottom', 'Distance', units='m')
        plot_steer.showGrid(x=True, y=True)
        
        # Příprava dat pro Delta (Input vs State length fix)
        delta_data = self.res.delta
        s_steer = s_sim
        
        # Delta má obvykle délku N, s_sim má N+1. Ořízneme s_sim.
        if len(delta_data) < len(s_steer):
            s_steer = s_sim[:len(delta_data)]
            
        # Převod na stupně
        delta_deg = np.degrees(delta_data)
        
        # Vykreslení
        plot_steer.plot(s_steer, delta_deg, pen=pg.mkPen('r', width=2), name="Steering")
        plot_steer.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine)))

        # C) XY Map Comparison
        plot_map = pg.PlotWidget(title="Trajectory comparison")
        plot_map.setAspectLocked(True)
        plot_map.setLabel('bottom', 'X', units='m')
        plot_map.setLabel('left', 'Y', units='m')
        plot_map.addLegend()

        plot_map.plot(x_ref, y_ref, pen=pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine), name="Track Center")
        plot_map.plot(x_car, y_car, pen=pg.mkPen('b', width=2), name="Car Path")
        plot_map.plot([x_car[0]], [y_car[0]], symbol='t1', symbolBrush='g', symbolSize=15, name="Start")

        # --- Layout Assembly ---
        # Stretch faktory určují poměr výšky (1:1:2)
        layout.addWidget(plot_dev, stretch=1)
        layout.addWidget(plot_steer, stretch=1)
        layout.addWidget(plot_map, stretch=2)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Trajectory")

    def init_powertrain_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # --- 1. POWER GRAPH (Zůstává stejný) ---
        v_interp = self.res.v[:-1]
        power_kw = (self.res.f_acc * v_interp) / 1000.0 
        brake_kw = (self.res.f_brake * v_interp) / 1000.0 
        
        plot_p = pg.PlotWidget(title="Power usage")
        plot_p.setLabel('left', 'Power', units='kW')
        plot_p.setLabel('bottom', 'Distance', units='m')
        plot_p.addLegend() 
        
        plot_p.plot(self.res.s[:-1], power_kw, pen=pg.mkPen('g', width=2), name="Sim")
        plot_p.plot(self.res.s[:-1], brake_kw, pen=pg.mkPen('r', width=2), name="Sim")
        plot_p.plot(self.res.s[:-1],-self.res.recuperation/1000,pen=pg.mkPen('b', width=2), name="Sim")
        
        # --- PŘÍPRAVA NA ENERGY GRAPH ---
        # Spočítáme čistě trakční energii (bez rekuperace)
        # E = integral(P * dt) -> sum(P * dt)
        # dt = ds / v
        
        # Získáme časové kroky pro simulaci
        dt_sim = np.diff(self.res.t)
        
        # Filtrujeme jen kladný výkon (Trakce)
        # power_kw je výkon na kolech, musíme ho vydělit účinností motoru, abychom dostali spotřebu z baterie
        # self.car.power_eff je např 0.9. Pokud motor dává 90kW, z baterky jde 100kW.
        pos_power_battery = np.maximum(power_kw, 0) / self.car.power_eff 
        
        # Energie v kWh = (kW * s) / 3600
        e_step_sim = (pos_power_battery * dt_sim) / 3600.0
        e_cum_sim = np.cumsum(e_step_sim)
        # Přidáme nulu na začátek, aby délka odpovídala s[:-1] nebo s
        e_cum_sim = np.insert(e_cum_sim, 0, 0.0)

        # --- VALIDATION DATA PROCESSING ---
        val_data = None
        p_ref_kw = None
        e_cum_real = None
        
        if self.track.ref_data is not None:
            val_data = self.calculate_validation_metrics(self.res, self.track.ref_data)
            
            if val_data and val_data.get('p_ref') is not None:
                # Real Power
                p_ref_kw = val_data['p_ref'] / 1000.0 # W -> kW
                plot_p.plot(val_data['s'], p_ref_kw, pen=pg.mkPen('k', width=2, style=Qt.PenStyle.DotLine), name="Real Power")
                
                # Real Energy (Traction Only)
                # Musíme odhadnout dt pro realitu
                # Validation metrics nám vrací 'v_ref' interpolované na s_sim
                # Takže můžeme použít stejné dt_sim (protože jsme na stejné dráze)
                # NEBO přesněji: dt = ds / v_ref
                
                ds = np.diff(val_data['s'])
                v_ref_safe = np.maximum(val_data['v_ref'][:-1], 1.0)
                dt_ref = ds / v_ref_safe
               
                pos_power_real = np.maximum(p_ref_kw[:-1], 0)
                
                e_step_real = (pos_power_real * dt_ref) / 3600.0
                e_cum_real = np.cumsum(e_step_real)
                e_cum_real = np.insert(e_cum_real, 0, 0.0)

       
        plot_e = pg.PlotWidget(title="Cumulative energy (discharge only)")
        plot_e.setLabel('left', 'Energy Out', units='kWh')
        plot_e.setLabel('bottom', 'Distance', units='m')
        plot_e.addLegend()
        
        # Sim Energy
        # Používáme self.res.s, což má délku N+1, stejně jako e_cum_sim
        plot_e.plot(self.res.s, e_cum_sim, pen=pg.mkPen('orange', width=2), name="Sim")
        
        # Real Energy
        if e_cum_real is not None:
            plot_e.plot(val_data['s'], e_cum_real, pen=pg.mkPen('k', width=2, style=Qt.PenStyle.DotLine), name="Real discharge")
        
        plot_e.setXLink(plot_p)
        
        layout.addWidget(plot_p)
        layout.addWidget(plot_e)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Powertrain")

    def init_forces_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        plot_f = pg.PlotWidget(title="Longitudinal Forces")
        plot_f.setLabel('left', 'Force', units='N')
        plot_f.setLabel('bottom', 'Distance', units='m')
        plot_f.showGrid(x=True, y=True)
        
        plot_f.plot(self.res.s[:-1], self.res.f_acc, pen='g', name="Traction")
        plot_f.plot(self.res.s[:-1], self.res.f_brake, pen='r', name="Braking")
        
        plot_f.addLegend()
        
        layout.addWidget(plot_f)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Forces")

    def init_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Vytvoříme PlotWidget
        self.map_plot = pg.PlotWidget(title="Power map")
        self.map_plot.setAspectLocked(True)
        self.map_plot.setLabel('bottom', 'X Position', units='m')
        self.map_plot.setLabel('left', 'Y Position', units='m')


        # --- 1. DATA PREP ---
        s_sim = self.res.s
        # Interpolace souřadnic
        x_plot = np.interp(s_sim, self.track.s, self.track.x)
        y_plot = np.interp(s_sim, self.track.s, self.track.y)   

        # Výpočet výkonu
        # Zkrátíme o 1 index, aby seděly délky (v_avg je pro intervaly)
        v_avg = self.res.v[:-1]
        f_acc = self.res.f_acc
        f_brake = self.res.f_brake
        
        # Power [kW]
        power_kw = ((f_acc + f_brake) * v_avg) / 1000.0
        
        # X a Y zkrátíme, aby seděly k Power vektoru
        x_plot = x_plot[:-1]
        y_plot = y_plot[:-1]
        
        # Uložíme data do self, abychom k nim mohli v 'mouse_moved'
        self.map_data = {
            'x': x_plot,
            'y': y_plot,
            'p': power_kw,
            'v': v_avg * 3.6, # km/h
            's': s_sim[:-1]
        }

        
        p_min = np.min(power_kw)
        p_max = np.max(power_kw)
        span = p_max - p_min + 1e-9 
        
       
        zero_pos = (0 - p_min) / span
        zero_pos = np.clip(zero_pos, 0.0, 1.0) # Omezit na 0-1

     
        pos = np.array([0.0, zero_pos, 1.0])
        color_def = np.array([
            [255, 0, 0, 255],     # Red (Max Regen)
            [0, 150, 255, 255], # Grey/Yellow (0 kW)
            [0, 255, 0, 255]      # Green (Max Power)
        ], dtype=np.ubyte)
        
        # Vytvoříme ColorMap objekt
        cmap = pg.ColorMap(pos, color_def)
        
        # Získáme barvy pro všechny body najednou (LUT lookup)
        # Nejprve normalizujeme power na 0..1
        norm_power = (power_kw - p_min) / span
        brushes = cmap.map(norm_power, mode='qcolor') # Vrátí pole QColor

     
        self.scatter = pg.ScatterPlotItem(
            x=x_plot, 
            y=y_plot, 
            size=6,           # Velikost bodu 
            pen=None,         # Bez obrysu
            brush=brushes,    # Naše barvy
            hoverable=True,
            hoverSize=10      # Zvětší se při najetí myší
        )
        self.map_plot.addItem(self.scatter)
        
        # Start marker
        start_marker = pg.ScatterPlotItem(x=[x_plot[0]], y=[y_plot[0]], size=15, symbol='o', brush='w')
        self.map_plot.addItem(start_marker)

 
        self.legend = pg.GradientLegend((20, 150), (10, 10)) # Velikost, Offset
        self.legend.setGradient(cmap.getGradient())
        # Nastavíme popisky (Min, 0, Max)
        labels = {
            f"{p_min:.0f} kW": 0,
            "0 kW": zero_pos,
            f"{p_max:.0f} kW": 1
        }
        self.legend.setLabels(labels)
        self.map_plot.addItem(self.legend)

        # --- 5. INTERAKCE (CROSSHAIR & LABEL) ---
        # Přidáme label, který bude ukazovat hodnoty
        self.cursor_label = pg.TextItem(anchor=(0, 1), color="w") # Anchor vlevo dole
        self.map_plot.addItem(self.cursor_label)
        
        # Crosshair (zaměřovač)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
        self.map_plot.addItem(self.v_line)
        self.map_plot.addItem(self.h_line)
        
        # Propojíme pohyb myši s funkcí
        self.map_plot.scene().sigMouseMoved.connect(self.on_map_mouse_move)

        layout.addWidget(self.map_plot)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Power Map")

    def on_map_mouse_move(self, pos):
        """Obsluha pohybu myši nad grafem - hledá nejbližší bod na trati."""
        if self.map_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.map_plot.plotItem.vb.mapSceneToView(pos)
            mx, my = mouse_point.x(), mouse_point.y()
            
            # Najdeme nejbližší bod v datech
            # Vypočteme vzdálenost myši od všech bodů tratě (vektorově - velmi rychlé)
            dists = np.sqrt((self.map_data['x'] - mx)**2 + (self.map_data['y'] - my)**2)
            idx = np.argmin(dists) # Index nejbližšího bodu
            
            # Pokud jsme blízko trati (např. do 20 metrů), zobrazíme info
            if dists[idx] < 20.0:
                x_track = self.map_data['x'][idx]
                y_track = self.map_data['y'][idx]
                p_val = self.map_data['p'][idx]
                v_val = self.map_data['v'][idx]
                s_val = self.map_data['s'][idx]
                
                # Posuneme Crosshair na bod tratě
                self.v_line.setPos(x_track)
                self.h_line.setPos(y_track)
                
                # Barva textu podle výkonu
                color_hex = "#00ff00" if p_val > 0 else "#ff5555" if p_val < 0 else "#ffffff"
                
                # Text Labelu
                html_text = (
                    f"<div style='background-color: rgba(0,0,0,150); padding: 5px; font-size: 14px;'>"
                    f"<b style='color: white;'>Dist: {s_val:.1f} m</b><br>"
                    f"Speed: {v_val:.1f} km/h<br>"
                    f"Power: <b style='color: {color_hex};'>{p_val:.1f} kW</b>"
                    f"</div>"
                )
                
                self.cursor_label.setHtml(html_text)
                self.cursor_label.setPos(x_track, y_track)
                self.cursor_label.setVisible(True)
                self.v_line.setVisible(True)
                self.h_line.setVisible(True)
            else:
                # Jsme moc daleko od tratě
                self.cursor_label.setVisible(False)
                self.v_line.setVisible(False)
                self.h_line.setVisible(False)

    def init_sensitivity_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        
       
        if not self.res.sens:
            main_layout.addWidget(QLabel("No sensitivity data available."))
            widget.setLayout(main_layout)
            self.tabs.addTab(widget, "Sensitivity")
            return

        raw = self.res.sens
        
        self.scenario_data = {
            'Mass (m)': {
                'val': self.car.mass, 
                'sens': raw.get('dTime_dm', 0.0), 
                'unit': 'kg'
            },
            'Drag (Cd)': {
                'val': self.car.cd,
                'sens': raw.get('dTime_dCdA', 0.0), 
                'unit': '-'
            },
            'Downforce (Cl)': {
                'val': self.car.cl,
                'sens': raw.get('dTime_dCL', 0.0), 
                'unit': '-'
            }
        }

        # 2. control panel
        control_group = QGroupBox("Scenario Configuration (Multi-Parametric Analysis)")
        grid = QGridLayout()
        grid.setSpacing(15) 
        
        # Hlavičky tabulky
        grid.addWidget(QLabel("<b>Parameter</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Base Value</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Change (%)</b>"), 0, 2)
        grid.addWidget(QLabel("<b>Delta Time</b>"), 0, 3) 

        self.inputs = {}        # reference na spinboxy
        self.deltas_labels = {} # reference na labely s výsledky

        row = 1
        for name, data in self.scenario_data.items():
            # A) name of parameter
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("font-size: 14px;")
            
            # B) value
            lbl_val = QLabel(f"{data['val']:.3f} {data['unit']}")
            lbl_val.setStyleSheet("color: #666; font-family: Consolas;")
            
            # C) Input SpinBox (%)
            spin = QDoubleSpinBox()
            spin.setRange(-50.0, 50.0)
            spin.setValue(0.0)
            spin.setSuffix("%")
            spin.setSingleStep(0.5)
            spin.setFixedWidth(100)
            spin.valueChanged.connect(self.update_scenario) # Live update
            
            self.inputs[name] = spin
            
            # D) Delta Label 
            lbl_delta = QLabel("0.000 s")
            lbl_delta.setAlignment(Qt.AlignmentFlag.AlignRight)
            lbl_delta.setStyleSheet("font-family: Consolas; font-weight: bold;")
            self.deltas_labels[name] = lbl_delta

            # grid import
            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(lbl_val, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(lbl_delta, row, 3)
            row += 1

        # Reset button
        btn_reset = QPushButton("Reset All to 0%")
        btn_reset.clicked.connect(self.reset_scenario)
        grid.addWidget(btn_reset, row, 2, 1, 2)

        control_group.setLayout(grid)
        main_layout.addWidget(control_group)

        # 3. results panel
        self.lbl_scenario_results = QLabel()
        self.lbl_scenario_results.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_scenario_results.setStyleSheet("""
            background-color: #f8f9fa; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            padding: 20px; 
        """)
        self.lbl_scenario_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(self.lbl_scenario_results)
        main_layout.addStretch() 
        
        widget.setLayout(main_layout)
        self.tabs.addTab(widget, "Sensitivity")
        
  
        self.update_scenario()


    def reset_scenario(self):
        """resets all spinboxs to 0"""
        for spin in self.inputs.values():
            spin.blockSignals(True) 
            spin.setValue(0.0)
            spin.blockSignals(False)
        self.update_scenario()

    def update_scenario(self):
        
        total_dt = 0.0
        
        # iteration over parameters
        for name, spin in self.inputs.items():
            pct = spin.value()
            data = self.scenario_data[name]
            
            # change
            # Delta Param = Base * (Pct / 100)
            d_param = data['val'] * (pct / 100.0)
            
            # Delta Time = Sens * Delta Param
            d_time = data['sens'] * d_param
            
            total_dt += d_time
            
            # color gang
            color = "#28a745" if d_time < 0 else "#dc3545" if d_time > 0 else "#888"
            sign = "+" if d_time > 0 else ""
            self.deltas_labels[name].setText(f"<span style='color:{color}'>{sign}{d_time:.4f} s</span>")

        # overal sum
        base_time = self.res.lap_time
        new_time = base_time + total_dt
        
        # color
        diff_color = "#28a745" if total_dt < 0 else "#dc3545" if total_dt > 0 else "#333"
        diff_sign = "+" if total_dt > 0 else ""

        # 3. HTML 
        html = f"""
        <div style='text-align: center; font-family: sans-serif;'>
            <h3 style='margin-bottom: 5px; color: #666; text-transform: uppercase; letter-spacing: 1px;'>Predicted Lap Time</h3>
            
            <div style='font-size: 48px; font-weight: bold; color: {diff_color}; margin: 10px 0;'>
                {new_time:.3f} s
            </div>
            
            <div style='font-size: 20px; color: {diff_color}; font-weight: bold;'>
                ({diff_sign}{total_dt:.3f} s)
            </div>
            
            <div style='margin-top: 15px; font-size: 14px; color: #999; border-top: 1px solid #eee; padding-top: 10px; display: inline-block;'>
                Baseline: {base_time:.3f} s
            </div>
        </div>
        """
        
        self.lbl_scenario_results.setText(html)

    def init_drs_map_tab(self):
        
        widget = QWidget()
        layout = QVBoxLayout()

        self.drs_plot = pg.PlotWidget(title="DRS Usage Map")
        self.drs_plot.setAspectLocked(True)
        self.drs_plot.setLabel('bottom', 'X Position', units='m')
        self.drs_plot.setLabel('left', 'Y Position', units='m')

        # --- DATA PREP ---
        # DRS usage je pole 0..1
        drs_data = self.res.drs_usage
        
        # Interpolace souřadnic na délku DRS pole (které má délku N)
        # res.s má délku N+1, takže použijeme s[:-1]
        s_sim = self.res.s[:-1]
        
        x_plot = np.interp(s_sim, self.track.s, self.track.x)
        y_plot = np.interp(s_sim, self.track.s, self.track.y) 

        # --- COLOR MAPPING ---
        # 0.0 (Closed) -> Red
        # 1.0 (Open)   -> Green
        # Mezi tím -> Interpolace
        
        pos = np.array([0.0, 1.0])
        color_def = np.array([
            [255, 0, 0, 255],   # Red (Closed)
            [0, 255, 0, 255]    # Green (Open)
        ], dtype=np.ubyte)
        
        cmap = pg.ColorMap(pos, color_def)
        
        # Obarvení bodů
        brushes = cmap.map(drs_data, mode='qcolor')

        # --- SCATTER PLOT ---
        self.drs_scatter = pg.ScatterPlotItem(
            x=x_plot, 
            y=y_plot, 
            size=6, 
            pen=None, 
            brush=brushes,
            hoverable=True,
            hoverSize=10
        )
        self.drs_plot.addItem(self.drs_scatter)
        
        # Start/Finish line marker
        self.drs_plot.plot([x_plot[0]], [y_plot[0]], symbol='o', symbolBrush='w', symbolSize=10)

        # --- LEGEND ---
        legend = pg.GradientLegend((20, 100), (10, 10))
        legend.setGradient(cmap.getGradient())
        legend.setLabels({"Closed": 0, "Open": 1})
        self.drs_plot.addItem(legend)

        layout.addWidget(self.drs_plot)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "DRS Map")

    def init_soc_tab(self):
        """Initializes the State of Charge (SoC) tab if data is available."""
        # Check if soc exists and is not None
        if self.res.SoC is None:
            return

        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create PlotWidget
        plot_soc = pg.PlotWidget(title="State of charge (SoC)")
        plot_soc.setLabel('left', 'SoC', units='%')
        plot_soc.setLabel('bottom', 'Distance', units='m')
        plot_soc.showGrid(x=True, y=True)
        
        # Ensure lengths match (s is usually N+1, soc might be N or N+1)
        s_plot = self.res.s[:len(self.res.SoC)]
        
        # Plot the data
        plot_soc.plot(s_plot, self.res.SoC, pen=pg.mkPen('b', width=2), name="SoC")
        
        layout.addWidget(plot_soc)
        widget.setLayout(layout)
        
        self.tabs.addTab(widget, "Battery")
