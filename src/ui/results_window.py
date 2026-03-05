import datetime
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTabWidget, QGroupBox, QGridLayout, QFileDialog, QDoubleSpinBox, QApplication, QPushButton)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from src.utils.saver import save_results_to_npz
import os

class ResultsWindow(QWidget):
    # Moderní barevná paleta
    C_SIM = '#3b82f6'     # Modrá
    C_REF = '#334155'     # Tmavě šedá
    C_DELTA = '#ef4444'   # Červená
    C_ACC = '#10b981'     # Zelená
    C_BRK = '#ef4444'     # Červená
    C_REGEN = '#8b5cf6'   # Fialová
    C_ENERGY = '#f59e0b'  # Žlutá/Oranžová

    def __init__(self, res, car, track, ds):
        super().__init__()

        # Globální vyhlazování a barvy pro hezčí grafy
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', '#334155')
        pg.setConfigOptions(antialias=True)
        
        self.res = res
        self.car = car
        self.track = track
        self.ds = ds

        
        self.center_window()
        self.setWindowTitle(f"Simulation results - {res.lap_time:.3f} s")
        
        # --- APLIKACE MODERNÍHO VZHLEDU (QSS) ---
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                color: #334155;
                background-color: #f8f9fa;
            }
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
                font-size: 11pt;
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
            QPushButton {
                background-color: #3b82f6; 
                color: white; 
                font-weight: bold; 
                padding: 8px 15px; 
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        
        # main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)
        
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
        self.init_soc_tab()

        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.show()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.showNormal()

    def _style_plot(self, plot, title, y_label, units):
        """Pomocná metoda pro krásný čistý vzhled pyqtgraph plotů"""
        plot.setBackground("#FFFFFF")
        plot.setTitle(title, color='#0f172a', size='12pt', bold=True)
        plot.setLabel('left', y_label, units=units, color='#64748b')
        plot.setLabel('bottom', 'Distance', units='m', color='#64748b')
        
        
        # Obarvení os a textu na jemnou šedou
        ax_pen = pg.mkPen(color='#94a3b8', width=1)
        plot.getAxis('left').setPen(ax_pen)
        plot.getAxis('left').setTextPen('#475569')
        plot.getAxis('bottom').setPen(ax_pen)
        plot.getAxis('bottom').setTextPen('#475569')
        
        # Jemná mřížka
        plot.showGrid(x=True, y=True, alpha=0.15)
        return plot

    def init_summary_panel(self):
        group = QGroupBox("Lap Summary")
        group.setObjectName("HighlightedGroup")
        main_h_layout = QHBoxLayout() 
        main_h_layout.setContentsMargins(15, 15, 15, 15)
        
        stats_widget = QWidget()
        grid = QGridLayout(stats_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(40) # Mírně zvětšen rozestup pro více dat
        stats_widget.setStyleSheet("background-color: transparent;")
        
        def add_stat(row, col, label, value, unit):
            lbl_name = QLabel(f"<b>{label}:</b>")
            lbl_name.setStyleSheet("color: #64748b; font-size: 13px; background-color: transparent;")
            lbl_val = QLabel(f"{value}")
            lbl_val.setStyleSheet("color: #3b82f6; font-size: 16px; font-weight: bold; background-color: transparent;")
            lbl_unit = QLabel(unit)
            lbl_unit.setStyleSheet("color: #94a3b8; font-size: 13px; background-color: transparent;")
            
            grid.addWidget(lbl_name, row, col)
            grid.addWidget(lbl_val, row, col+1)
            grid.addWidget(lbl_unit, row, col+2)

        # Výpočty dat
        max_v = np.max(self.res.v) * 3.6
        avg_v = np.mean(self.res.v) * 3.6
        total_energy = self.res.E[-1] - self.res.E[0]
        
        # Získání SoC hodnot (ošetření pokud res.SoC neexistuje)
        soc_start = self.res.SoC[0] if self.res.SoC is not None else 0.0
        soc_end = self.res.SoC[-1] if self.res.SoC is not None else 0.0

        # První sloupec statistik
        add_stat(0, 0, "Lap Time", f"{self.res.lap_time:.3f}", "s")
        add_stat(1, 0, "Max Speed", f"{max_v:.1f}", "km/h")
        
        # Druhý sloupec statistik
        add_stat(0, 3, "Energy Used", f"{total_energy:.3f}", "kWh")
        add_stat(1, 3, "Avg Speed", f"{avg_v:.1f}", "km/h")

        # Třetí sloupec statistik (SoC data)
        add_stat(0, 6, "SoC Start", f"{soc_start:.1f}", "%")
        add_stat(1, 6, "SoC End", f"{soc_end:.1f}", "%")
        
        main_h_layout.addWidget(stats_widget)
        main_h_layout.addStretch()
        
        self.btn_save = QPushButton("Save Results (.npz)")
        self.btn_save.setFixedWidth(160)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.on_save_clicked)
        
        main_h_layout.addWidget(self.btn_save)
        group.setLayout(main_h_layout)
        self.layout.addWidget(group)

    def calculate_validation_metrics(self, sim_res, ref_data):
        if ref_data is None or ref_data['v'] is None:
            return None

        s_sim = sim_res.s
        s_ref = ref_data['s']
        v_ref_raw = ref_data['v']
        
        ds_ref = np.diff(s_ref, prepend=0)
        v_safe = np.maximum(v_ref_raw, 1.0) 
        dt_ref = ds_ref / v_safe
        t_ref_calculated = np.cumsum(dt_ref)

        v_ref_interp = np.interp(s_sim, s_ref, v_ref_raw)
        t_ref_interp = np.interp(s_sim, s_ref, t_ref_calculated)
        
        p_ref_interp = None
        if ref_data.get('p') is not None:
            p_ref_interp = np.interp(s_sim, s_ref, ref_data['p']) * 1000.0

        delta_v = sim_res.v - v_ref_interp 
        
        t_sim_zeroed = sim_res.t - sim_res.t[0]
        t_ref_zeroed = t_ref_interp - t_ref_interp[0]
        
        delta_t = t_sim_zeroed - t_ref_zeroed
        rmse_v = np.sqrt(np.mean(delta_v**2))
        total_time_diff = t_sim_zeroed[-1] - t_ref_zeroed[-1]
        
        return {
            's': s_sim,
            'v_ref': v_ref_interp,
            't_ref': t_ref_zeroed,
            'p_ref': p_ref_interp,
            'delta_v': delta_v,
            'delta_t': delta_t,
            'rmse_v': rmse_v,
            'diff_total': total_time_diff
        }

    def on_save_clicked(self):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        default_name = f"{self.track.name}_{now_str}.npz"
        
        if not os.path.exists("results"):
            os.makedirs("results")
            
        default_path = os.path.join("results", default_name)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Results",
            default_path,
            "NumPy Zip (*.npz)"
        )

        if filepath:
            try:
                save_results_to_npz(filepath, self.res, self.track, self.ds)
                self.btn_save.setText("Saved!")
                self.btn_save.setStyleSheet("background-color: #10b981; color: white;")
            except Exception as e:
                print(f"Error saving file: {e}")
                self.btn_save.setText("Error!")
                self.btn_save.setStyleSheet("background-color: #ef4444; color: white;")

    def init_dynamics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # --- SPEED ---
        plot_v = pg.PlotWidget()
        self._style_plot(plot_v, "Velocity Comparison", "Speed", "km/h")
        
        legend_v = plot_v.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend_v.setLabelTextColor('#0f172a')
      
        plot_v.plot(self.res.s, self.res.v * 3.6, pen=pg.mkPen(self.C_SIM, width=2.5), name="Simulation")
        
        val_data = None
        if self.track.ref_data is not None:
            val_data = self.calculate_validation_metrics(self.res, self.track.ref_data)
            if val_data:
                plot_v.plot(val_data['s'], val_data['v_ref'] * 3.6, 
                           pen=pg.mkPen(self.C_REF, width=2, style=Qt.PenStyle.DashLine), 
                           name=f"Real Log (RMSE: {val_data['rmse_v']*3.6:.1f} km/h)")

        layout.addWidget(plot_v)

        # --- DELTA TIME ---
        plot_dt = pg.PlotWidget()
        self._style_plot(plot_dt, "Delta Time (Accumulated Gap)", "Delta Time", "s")
        plot_dt.setXLink(plot_v)
        
        plot_dt.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#334155', width=1.5)))

        if val_data:
            dt_curve = plot_dt.plot(val_data['s'], val_data['delta_t'], pen=pg.mkPen(self.C_DELTA, width=2.5))
            dummy_zero = pg.PlotCurveItem(val_data['s'], np.zeros_like(val_data['s']), pen=None)
            plot_dt.addItem(dummy_zero)
            
            fill_color = pg.mkColor(self.C_DELTA)
            fill_color.setAlpha(40)
            fill = pg.FillBetweenItem(dt_curve, dummy_zero, brush=fill_color)
            plot_dt.addItem(fill)
            
            diff = val_data['diff_total']
            sign = "+" if diff > 0 else ""
            plot_dt.setTitle(f"Delta Time (Final Gap: {sign}{diff:.3f} s)", color='#0f172a', size='12pt', bold=True)
        else:
            plot_dt.setTitle("No reference data available", color='#64748b')

        layout.addWidget(plot_dt)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dynamics")

    def init_powertrain_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # --- POWER ---
        v_interp = self.res.v[:-1]
        power_kw = (self.res.f_acc * v_interp) / 1000.0 
        brake_kw = (self.res.f_brake * v_interp) / 1000.0 
        
        plot_p = pg.PlotWidget()
        self._style_plot(plot_p, "Power Usage", "Power", "kW")
        legend_p = plot_p.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend_p.setLabelTextColor('#0f172a')
        
        plot_p.plot(self.res.s[:-1], power_kw, pen=pg.mkPen(self.C_ACC, width=2.5), name="Traction")
        plot_p.plot(self.res.s[:-1], brake_kw, pen=pg.mkPen(self.C_BRK, width=2.5), name="Braking")
        plot_p.plot(self.res.s[:-1], -self.res.recuperation/1000, pen=pg.mkPen(self.C_REGEN, width=2.5), name="Recuperation")
        
        # --- ENERGY ---
        dt_sim = np.diff(self.res.t)
        pos_power_battery = np.maximum(power_kw, 0) / self.car.power_eff 
        e_step_sim = (pos_power_battery * dt_sim) / 3600.0
        e_cum_sim = np.insert(np.cumsum(e_step_sim), 0, 0.0)

        val_data = None
        e_cum_real = None
        
        if self.track.ref_data is not None:
            val_data = self.calculate_validation_metrics(self.res, self.track.ref_data)
            if val_data and val_data.get('p_ref') is not None:
                p_ref_kw = val_data['p_ref'] / 1000.0 
                plot_p.plot(val_data['s'], p_ref_kw, pen=pg.mkPen(self.C_REF, width=2, style=Qt.PenStyle.DotLine), name="Real Power")
                
                ds = np.diff(val_data['s'])
                v_ref_safe = np.maximum(val_data['v_ref'][:-1], 1.0)
                dt_ref = ds / v_ref_safe
                pos_power_real = np.maximum(p_ref_kw[:-1], 0)
                
                e_step_real = (pos_power_real * dt_ref) / 3600.0
                e_cum_real = np.insert(np.cumsum(e_step_real), 0, 0.0)

        plot_e = pg.PlotWidget()
        self._style_plot(plot_e, "Cumulative Energy (Discharge only)", "Energy Out", "kWh")
        legend_e = plot_e.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend_e.setLabelTextColor('#0f172a')
        
        e_curve = plot_e.plot(self.res.s, e_cum_sim, pen=pg.mkPen(self.C_ENERGY, width=3), name="Sim")
        
        # Výplň energie
        dummy_zero = pg.PlotCurveItem(self.res.s, np.zeros_like(self.res.s), pen=None)
        fill_color = pg.mkColor(self.C_ENERGY)
        fill_color.setAlpha(30)
        plot_e.addItem(pg.FillBetweenItem(e_curve, dummy_zero, brush=fill_color))
        
        if e_cum_real is not None:
            plot_e.plot(val_data['s'], e_cum_real, pen=pg.mkPen(self.C_REF, width=2, style=Qt.PenStyle.DashLine), name="Real discharge")
        
        plot_e.setXLink(plot_p)
        
        layout.addWidget(plot_p)
        layout.addWidget(plot_e)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Powertrain")

    def init_forces_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        plot_f = pg.PlotWidget()
        self._style_plot(plot_f, "Longitudinal Forces", "Force", "N")
        legend_f = plot_f.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend_f.setLabelTextColor('#0f172a')
        
        plot_f.plot(self.res.s[:-1], self.res.f_acc, pen=pg.mkPen(self.C_ACC, width=2.5), name="Traction")
        plot_f.plot(self.res.s[:-1], self.res.f_brake, pen=pg.mkPen(self.C_BRK, width=2.5), name="Braking")
        
        layout.addWidget(plot_f)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Forces")

    def init_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Kontejner s bílým pozadím a zaoblením
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.map_plot = pg.PlotWidget(title="Power Map")
        self.map_plot.setBackground('#ffffff')
        self.map_plot.setTitle("Power Map", color='#0f172a', size='12pt', bold=True)
        self.map_plot.setAspectLocked(True)
        
        # Čistá mapa - skrytí os
        self.map_plot.hideAxis('bottom')
        self.map_plot.hideAxis('left')

        s_sim = self.res.s
        x_plot = np.interp(s_sim, self.track.s, self.track.x)[:-1]
        y_plot = np.interp(s_sim, self.track.s, self.track.y)[:-1]

        v_avg = self.res.v[:-1]
        power_kw = ((self.res.f_acc + self.res.f_brake) * v_avg) / 1000.0
        
        self.map_data = {'x': x_plot, 'y': y_plot, 'p': power_kw, 'v': v_avg * 3.6, 's': s_sim[:-1]}

        p_min = np.min(power_kw)
        p_max = np.max(power_kw)
        span = p_max - p_min + 1e-9 
        zero_pos = np.clip((0 - p_min) / span, 0.0, 1.0) 

        pos = np.array([0.0, zero_pos, 1.0])
        color_def = np.array([
            [239, 68, 68, 255],    # Red (Braking/Regen)
            [226, 232, 240, 255],  # Light Gray (0 kW)
            [16, 185, 129, 255]    # Green (Power)
        ], dtype=np.ubyte)
        
        cmap = pg.ColorMap(pos, color_def)
        brushes = cmap.map((power_kw - p_min) / span, mode='qcolor') 

        self.scatter = pg.ScatterPlotItem(x=x_plot, y=y_plot, size=8, pen=None, brush=brushes, hoverable=True, hoverSize=12)
        self.map_plot.addItem(self.scatter)
        
        start_marker = pg.ScatterPlotItem(x=[x_plot[0]], y=[y_plot[0]], size=15, symbol='o', brush='#0f172a')
        self.map_plot.addItem(start_marker)

        self.legend = pg.GradientLegend((20, 150), (10, 10)) 
        self.legend.setGradient(cmap.getGradient())
        self.legend.setLabels({f"{p_min:.0f} kW": 0, "0 kW": zero_pos, f"{p_max:.0f} kW": 1})
        self.legend.labelStyle = {'color': '#334155', 'font-size': '10pt'} # Tmavý text
        self.map_plot.addItem(self.legend)

        self.cursor_label = pg.TextItem(anchor=(0, 1), color="#334155") 
        self.map_plot.addItem(self.cursor_label)
        
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#94a3b8', style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#94a3b8', style=Qt.PenStyle.DashLine))
        self.map_plot.addItem(self.v_line)
        self.map_plot.addItem(self.h_line)
        
        self.map_plot.scene().sigMouseMoved.connect(self.on_map_mouse_move)

        canvas_layout.addWidget(self.map_plot)
        layout.addWidget(canvas_container)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Power Map")

    def on_map_mouse_move(self, pos):
        if self.map_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.map_plot.plotItem.vb.mapSceneToView(pos)
            mx, my = mouse_point.x(), mouse_point.y()
            
            dists = np.sqrt((self.map_data['x'] - mx)**2 + (self.map_data['y'] - my)**2)
            idx = np.argmin(dists) 
            
            if dists[idx] < 20.0:
                p_val = self.map_data['p'][idx]
                html_text = (
                    f"<div style='background-color: rgba(255,255,255,220); border: 1px solid #cbd5e1; padding: 8px; border-radius: 4px; font-family: Segoe UI;'>"
                    f"<b>Dist:</b> {self.map_data['s'][idx]:.1f} m<br>"
                    f"<b>Speed:</b> {self.map_data['v'][idx]:.1f} km/h<br>"
                    f"<b>Power:</b> <span style='color: {'#10b981' if p_val > 0 else '#ef4444'};'>{p_val:.1f} kW</span>"
                    f"</div>"
                )
                self.cursor_label.setHtml(html_text)
                self.cursor_label.setPos(self.map_data['x'][idx], self.map_data['y'][idx])
                self.cursor_label.setVisible(True)
                self.v_line.setPos(self.map_data['x'][idx])
                self.h_line.setPos(self.map_data['y'][idx])
                self.v_line.setVisible(True)
                self.h_line.setVisible(True)
            else:
                self.cursor_label.setVisible(False)
                self.v_line.setVisible(False)
                self.h_line.setVisible(False)

    def init_sensitivity_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        if not self.res.sens:
            main_layout.addWidget(QLabel("No sensitivity data available."))
            widget.setLayout(main_layout)
            self.tabs.addTab(widget, "Sensitivity")
            return

        raw = self.res.sens
        self.scenario_data = {
            'Weight (m)': {'val': self.car.weight, 'sens': raw.get('dTime_dm', 0.0), 'unit': 'kg'},
            'Drag (Cd)': {'val': self.car.cd, 'sens': raw.get('dTime_dCdA', 0.0), 'unit': '-'},
            'Downforce (Cl)': {'val': self.car.cl, 'sens': raw.get('dTime_dCL', 0.0), 'unit': '-'}
        }

        control_group = QGroupBox("Scenario Configuration)")
        
        control_group.setObjectName("HighlightedGroup")
        

        grid = QGridLayout()
        grid.setSpacing(15) 
        
        # Hlavičky s průhledným pozadím
        def create_header(text):
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setStyleSheet("background-color: transparent; color: #334155;")
            return lbl

        grid.addWidget(create_header("Parameter"), 0, 0)
        grid.addWidget(create_header("Base Value"), 0, 1)
        grid.addWidget(create_header("Change (%)"), 0, 2)
        grid.addWidget(create_header("Delta Time"), 0, 3) 

        self.inputs = {} 
        self.deltas_labels = {}

        row = 1
        for name, data in self.scenario_data.items():
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("background-color: transparent; color: #334155; font-weight: bold;")
            
            lbl_val = QLabel(f"{data['val']:.3f} {data['unit']}")
            lbl_val.setStyleSheet("background-color: transparent; color: #64748b; font-family: Consolas;")
            
            spin = QDoubleSpinBox()
            spin.setRange(-50.0, 50.0)
            spin.setValue(0.0)
            spin.setSuffix(" %")
            spin.setSingleStep(0.5)
            spin.setFixedWidth(120)
            spin.valueChanged.connect(self.update_scenario) 
            self.inputs[name] = spin
            
            lbl_delta = QLabel("0.000 s")
            lbl_delta.setAlignment(Qt.AlignmentFlag.AlignRight)
            lbl_delta.setStyleSheet("background-color: transparent; font-family: Consolas; font-weight: bold; color: #475569;")
            self.deltas_labels[name] = lbl_delta

            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(lbl_val, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(lbl_delta, row, 3)
            row += 1

        btn_reset = QPushButton("Reset All")
        btn_reset.setStyleSheet("background-color: #64748b; color: white;")
        btn_reset.clicked.connect(self.reset_scenario)
        grid.addWidget(btn_reset, row, 2, 1, 2)

        control_group.setLayout(grid)
        main_layout.addWidget(control_group)

        self.lbl_scenario_results = QLabel()
        self.lbl_scenario_results.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_scenario_results.setStyleSheet("""
            background-color: #f1f5f9; 
            border: 1px solid #cbd5e1; 
            border-radius: 8px;
            padding: 30px; 
            margin-top: 15px;
        """)
        self.lbl_scenario_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(self.lbl_scenario_results)
        main_layout.addStretch() 
        
        widget.setLayout(main_layout)
        self.tabs.addTab(widget, "Sensitivity")
        self.update_scenario()

    def reset_scenario(self):
        for spin in self.inputs.values():
            spin.blockSignals(True) 
            spin.setValue(0.0)
            spin.blockSignals(False)
        self.update_scenario()

    def update_scenario(self):
        total_dt = 0.0
        
        for name, spin in self.inputs.items():
            pct = spin.value()
            data = self.scenario_data[name]
            d_param = data['val'] * (pct / 100.0)
            d_time = data['sens'] * d_param
            total_dt += d_time
            
            color = "#10b981" if d_time < 0 else "#ef4444" if d_time > 0 else "#94a3b8"
            sign = "+" if d_time > 0 else ""
            self.deltas_labels[name].setText(f"<span style='color:{color}; background-color: transparent;'>{sign}{d_time:.4f} s</span>")

        base_time = self.res.lap_time
        new_time = base_time + total_dt
        
        diff_color = "#10b981" if total_dt < 0 else "#ef4444" if total_dt > 0 else "#475569"
        diff_sign = "+" if total_dt > 0 else ""

        html = f"""
        <div style='text-align: center;'>
            <div style='color: #64748b; font-size: 14px; text-transform: uppercase; font-weight: bold;'>Predicted Lap Time</div>
            <div style='font-size: 56px; font-weight: bold; color: {diff_color}; margin: 15px 0;'>
                {new_time:.3f} s
            </div>
            <div style='font-size: 24px; color: {diff_color}; font-weight: bold;'>
                ({diff_sign}{total_dt:.3f} s)
            </div>
            <div style='margin-top: 25px; font-size: 15px; color: #64748b; border-top: 1px solid #cbd5e1; padding-top: 15px;'>
                Baseline: {base_time:.3f} s
            </div>
        </div>
        """
        self.lbl_scenario_results.setText(html)

    def init_drs_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.drs_plot = pg.PlotWidget(title="DRS Usage Map")
        self.drs_plot.setBackground('#ffffff')
        self.drs_plot.setTitle("DRS Usage Map", color='#0f172a', size='12pt', bold=True)
        self.drs_plot.setAspectLocked(True)
        self.drs_plot.hideAxis('bottom')
        self.drs_plot.hideAxis('left')

        s_sim = self.res.s[:-1]
        x_plot = np.interp(s_sim, self.track.s, self.track.x)
        y_plot = np.interp(s_sim, self.track.s, self.track.y) 

        pos = np.array([0.0, 1.0])
        color_def = np.array([
            [239, 68, 68, 255],   # Red (Closed)
            [16, 185, 129, 255]   # Green (Open)
        ], dtype=np.ubyte)
        
        cmap = pg.ColorMap(pos, color_def)
        brushes = cmap.map(self.res.drs_usage, mode='qcolor')

        self.drs_scatter = pg.ScatterPlotItem(x=x_plot, y=y_plot, size=8, pen=None, brush=brushes, hoverable=True, hoverSize=12)
        self.drs_plot.addItem(self.drs_scatter)
        self.drs_plot.plot([x_plot[0]], [y_plot[0]], symbol='o', symbolBrush='#0f172a', symbolSize=15)

        legend = pg.GradientLegend((20, 100), (10, 10))
        legend.setGradient(cmap.getGradient())
        legend.setLabels({"Closed": 0, "Open": 1})
        legend.labelStyle = {'color': '#334155', 'font-size': '10pt'}
        self.drs_plot.addItem(legend)

        canvas_layout.addWidget(self.drs_plot)
        layout.addWidget(canvas_container)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "DRS Map")

    def init_soc_tab(self):
        if self.res.SoC is None: return

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        plot_soc = pg.PlotWidget()
        self._style_plot(plot_soc, "Battery State of Charge", "SoC", "%")
        
        s_plot = self.res.s[:len(self.res.SoC)]
        soc_curve = plot_soc.plot(s_plot, self.res.SoC, pen=pg.mkPen(self.C_SIM, width=3), name="SoC")
        
        dummy_zero = pg.PlotCurveItem(s_plot, np.zeros_like(s_plot), pen=None)
        fill_color = pg.mkColor(self.C_SIM)
        fill_color.setAlpha(30)
        plot_soc.addItem(pg.FillBetweenItem(soc_curve, dummy_zero, brush=fill_color))
        
        layout.addWidget(plot_soc)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Battery")