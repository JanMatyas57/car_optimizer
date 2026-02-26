import datetime
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTabWidget, QGroupBox, QGridLayout, QFileDialog, 
                             QComboBox, QApplication, QPushButton)
from PyQt6.QtCore import Qt
from src.utils.saver import save_results_to_npz

class EnduranceResultsWindow(QWidget):
    def __init__(self, results_list, car, track, ds):
        super().__init__()
        self.results_list = results_list
        self.car = car
        self.track = track
        self.ds = ds
        
        # default to first lap
        self.current_idx = 0
        self.current_res = self.results_list[0]

        self.resize(1100, 800)
        self.center_window()
        self.setWindowTitle("Endurance simulation results")
        
        # layout setup
        self.layout = QVBoxLayout(self)
        self.init_header()
        self.init_summary_panel()
        
        # tabs setup
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_dynamics_tab()
        self.init_powertrain_tab()
        self.init_forces_tab()
        self.init_map_tab()
        self.init_evolution_tab()
        
        # initial render
        self.update_view()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_header(self):
        """top bar with lap selection and save button."""
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Select Lap:</b>"))
        
        # lap selection combo
        self.combo_laps = QComboBox()
        self.combo_laps.setFixedWidth(150)
        items = [f"Lap {i+1}" for i in range(len(self.results_list))]
        self.combo_laps.addItems(items)
        self.combo_laps.currentIndexChanged.connect(self.on_lap_changed)
        header_layout.addWidget(self.combo_laps)
        
        header_layout.addStretch()
        
        # save button
        self.btn_save = QPushButton("Save All (.npz)")
        self.btn_save.setFixedWidth(150)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8; color: white; font-weight: bold; 
                padding: 8px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #138496; }
        """)
        self.btn_save.clicked.connect(self.on_save_clicked)
        header_layout.addWidget(self.btn_save)
        
        self.layout.addLayout(header_layout)

    def on_lap_changed(self, index):
        self.current_idx = index
        self.current_res = self.results_list[index]
        self.update_view()

    def init_summary_panel(self):
        group = QGroupBox("Lap Summary")
        layout = QVBoxLayout()
        
        stats_widget = QWidget()
        grid = QGridLayout(stats_widget)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(10)
        
        # labels
        self.lbl_lap_time = QLabel("0.000")
        self.lbl_energy = QLabel("0.000")
        self.lbl_max_v = QLabel("0.0")
        self.lbl_avg_v = QLabel("0.0")
        self.lbl_soc = QLabel("0.0")

        style = "color: #007bff; font-size: 16px; font-weight: bold; font-family: Consolas;"
        for lbl in [self.lbl_lap_time, self.lbl_energy, self.lbl_max_v, self.lbl_avg_v, self.lbl_soc]:
            lbl.setStyleSheet(style)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)

        def add_stat_to_grid(row, col_offset, label_text, value_widget, unit_text):
            lbl_name = QLabel(f"{label_text}:")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 14px;")
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            lbl_unit = QLabel(unit_text)
            lbl_unit.setStyleSheet("color: #888;")
            
            grid.addWidget(lbl_name, row, 0 + col_offset)
            grid.addWidget(value_widget, row, 1 + col_offset)
            grid.addWidget(lbl_unit, row, 2 + col_offset)

        # left column
        add_stat_to_grid(0, 0, "Lap Time", self.lbl_lap_time, "s")
        add_stat_to_grid(1, 0, "Max Speed", self.lbl_max_v, "km/h")
        add_stat_to_grid(2, 0, "Cumulative Energy", self.lbl_soc, "kWh")

        # right column
        add_stat_to_grid(0, 4, "Energy Used (Lap)", self.lbl_energy, "kWh")
        add_stat_to_grid(1, 4, "Avg Speed", self.lbl_avg_v, "km/h")
        
        grid.setColumnMinimumWidth(3, 40) 

        layout.addWidget(stats_widget)
        group.setLayout(layout)
        self.layout.addWidget(group)

    def init_dynamics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # speed plot
        self.plot_v = pg.PlotWidget(title="Speed profile")
        self.plot_v.setLabel('left', 'Speed', units='km/h')
        self.plot_v.showGrid(x=True, y=True)
        self.curve_v = self.plot_v.plot([], [], pen=pg.mkPen('b', width=2))
        
        # time plot
        self.plot_t = pg.PlotWidget(title="Time Accumulation (Relative to Lap Start)")
        self.plot_t.setLabel('left', 'Time', units='s')
        self.plot_t.setLabel('bottom', 'Distance', units='m')
        self.plot_t.showGrid(x=True, y=True)
        self.plot_t.setXLink(self.plot_v)
        self.curve_t = self.plot_t.plot([], [], pen=pg.mkPen('r', width=2))
        
        layout.addWidget(self.plot_v)
        layout.addWidget(self.plot_t)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dynamics")

    def init_powertrain_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # power plot
        self.plot_p = pg.PlotWidget(title="Power Usage")
        self.plot_p.setLabel('left', 'Power', units='kW')
        self.plot_p.showGrid(x=True, y=False)
        self.curve_acc = self.plot_p.plot([], [], pen='g', name="Acceleration", width=3)
        self.curve_reg = self.plot_p.plot([], [], pen='r', name="Regen", width=3)
        
        # energy plot
        self.plot_e = pg.PlotWidget(title="Energy Consumption (Cumulative)")
        self.plot_e.setLabel('left', 'Energy', units='kWh')
        self.plot_e.setLabel('bottom', 'Distance', units='m')
        self.plot_e.showGrid(x=True, y=True)
        self.plot_e.setXLink(self.plot_p)
        self.curve_e = self.plot_e.plot([], [], pen=pg.mkPen('orange', width=2))
        
        layout.addWidget(self.plot_p)
        layout.addWidget(self.plot_e)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Powertrain")

    def init_forces_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.plot_f = pg.PlotWidget(title="Longitudinal Forces")
        self.plot_f.setLabel('left', 'Force', units='N')
        self.plot_f.setLabel('bottom', 'Distance', units='m')
        self.plot_f.showGrid(x=True, y=True)
        self.plot_f.addLegend()
        
        self.curve_f_acc = self.plot_f.plot([], [], pen='g', name="Traction")
        self.curve_f_brake = self.plot_f.plot([], [], pen='r', name="Braking")
        
        layout.addWidget(self.plot_f)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Forces")

    def init_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.map_plot = pg.PlotWidget(title="Power Map")
        self.map_plot.setAspectLocked(True)
        self.map_plot.setLabel('bottom', 'X Position', units='m')
        self.map_plot.setLabel('left', 'Y Position', units='m')

        # scatter item
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, hoverable=True, hoverSize=10)
        self.map_plot.addItem(self.scatter)
        
        # markers
        self.start_marker = pg.ScatterPlotItem(size=15, symbol='o', brush='w')
        self.map_plot.addItem(self.start_marker)
        
        self.cursor_label = pg.TextItem(anchor=(0, 1), color="w")
        self.map_plot.addItem(self.cursor_label)
        
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine))
        self.map_plot.addItem(self.v_line)
        self.map_plot.addItem(self.h_line)
        
        self.map_plot.scene().sigMouseMoved.connect(self.on_map_mouse_move)

        layout.addWidget(self.map_plot)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Power Map")

    def init_evolution_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # data prep
        laps = np.arange(2, len(self.results_list) + 1)
        subset = self.results_list[1:]
        mus_x = [r.mu_x for r in subset]
        mus_y = [r.mu_y for r in subset]
        regens = [r.regen_eff for r in subset]
        lap_times = [r.t[-1] - r.t[0] for r in subset]
        
        # 1. grip evolution
        self.plot_grip = pg.PlotWidget(title="Grip evolution (Mu)")
        self.plot_grip.setLabel('left', 'Friction coeff [-]')
        self.plot_grip.setLabel('bottom', 'Lap Number')
        self.plot_grip.addLegend()

        self.plot_grip.plot(laps, mus_x, pen='r', symbol='o', name="Mu X (Long)") 
        self.plot_grip.plot(laps, mus_y, pen='b', symbol='t', name="Mu Y (Lat)")
        
        self.grip_marker = pg.InfiniteLine(pos=1, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine))
        self.plot_grip.addItem(self.grip_marker)

        # 2. regen evolution
        self.plot_regen = pg.PlotWidget(title="Regen Efficiency")
        self.plot_regen.setLabel('left', 'Efficiency [-]')
        self.plot_regen.setLabel('bottom', 'Lap Number')
        
        self.plot_regen.plot(laps, regens, pen='g', symbol='s', name="Regen Eff")
        
        self.regen_marker = pg.InfiniteLine(pos=1, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine))
        self.plot_regen.addItem(self.regen_marker)

        # 3. lap time evolution
        self.plot_times = pg.PlotWidget(title="Lap Time Evolution")
        self.plot_times.setLabel('left', 'Time [s]')
        self.plot_times.setLabel('bottom', 'Lap Number')
        self.plot_times.showGrid(x=False, y=False)
        
        self.plot_times.plot(laps, lap_times, pen='c', symbol='d', name="Lap Time")
        
        self.time_marker = pg.InfiniteLine(pos=1, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine))
        self.plot_times.addItem(self.time_marker)

        # link axes
        self.plot_grip.setXLink(self.plot_times)
        self.plot_regen.setXLink(self.plot_times)

        layout.addWidget(self.plot_grip)
        layout.addWidget(self.plot_regen)
        layout.addWidget(self.plot_times)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Evolution")

    def update_view(self):
        res = self.current_res
        
        # update summary labels
        lap_duration = res.t[-1] - res.t[0]
        energy_used = res.E[-1] - res.E[0]
        max_v = np.max(res.v) * 3.6
        avg_v = np.mean(res.v) * 3.6
        total_energy = res.E[-1]

        self.lbl_lap_time.setText(f"{lap_duration:.3f}")
        self.lbl_energy.setText(f"{energy_used:.3f}")
        self.lbl_max_v.setText(f"{max_v:.1f}")
        self.lbl_avg_v.setText(f"{avg_v:.1f}")
        self.lbl_soc.setText(f"{total_energy:.3f}")
        self.setWindowTitle(f"Endurance results - Lap {self.current_idx + 1}")

        # data prep for plots (relative to lap start)
        s_rel = res.s - res.s[0]
        t_rel = res.t - res.t[0]
        min_len = min(len(res.f_acc), len(res.v))
        s_plot = s_rel[:min_len]
        v_plot = res.v[:min_len]
        
        # update dynamic plots
        self.curve_v.setData(s_rel, res.v * 3.6)
        self.curve_t.setData(s_rel, t_rel)
        
        power_kw = ((res.f_acc[:min_len] + res.f_brake[:min_len]) * v_plot) / 1000.0
        acc_mask = power_kw > 0
        reg_mask = power_kw <= 0
        
        p_acc = np.copy(power_kw)
        p_acc[~acc_mask] = 0
        p_reg = np.copy(power_kw)
        p_reg[~reg_mask] = 0
        
        self.curve_acc.setData(s_plot, p_acc)
        self.curve_reg.setData(s_plot, p_reg)
        self.curve_e.setData(s_rel, res.E) 
        self.curve_f_acc.setData(s_plot, res.f_acc[:min_len])
        self.curve_f_brake.setData(s_plot, res.f_brake[:min_len])

        # update evolution markers
        current_lap_num = self.current_idx + 1
        if hasattr(self, 'grip_marker'):
            self.grip_marker.setPos(current_lap_num)
        if hasattr(self, 'regen_marker'):
            self.regen_marker.setPos(current_lap_num)
        if hasattr(self, 'time_marker'):
            self.time_marker.setPos(current_lap_num)

        # update map
        self.update_map_data(s_rel, v_plot, power_kw)

    def update_map_data(self, s_rel, v_plot, power_kw):
        x_plot = np.interp(s_rel, self.track.s, self.track.x)
        y_plot = np.interp(s_rel, self.track.s, self.track.y)
        
        n = len(power_kw)
        x_plot = x_plot[:n]
        y_plot = y_plot[:n]
        s_plot = s_rel[:n]

        self.map_data = {
            'x': x_plot, 'y': y_plot, 'p': power_kw, 'v': v_plot * 3.6, 's': s_plot
        }

        # coloring logic
        p_min, p_max = np.min(power_kw), np.max(power_kw)
        span = p_max - p_min + 1e-9
        zero_pos = np.clip((0 - p_min) / span, 0.0, 1.0)

        pos = np.array([0.0, zero_pos, 1.0])
        color_def = np.array([
            [255, 0, 0, 255], [0, 150, 255, 255], [0, 255, 0, 255]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color_def)
        
        norm_power = (power_kw - p_min) / span
        brushes = cmap.map(norm_power, mode='qcolor')

        self.scatter.setData(x=x_plot, y=y_plot, brush=brushes)
        self.start_marker.setData(x=[x_plot[0]], y=[y_plot[0]])

    def on_map_mouse_move(self, pos):
        if self.map_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.map_plot.plotItem.vb.mapSceneToView(pos)
            mx, my = mouse_point.x(), mouse_point.y()
            
            # find closest point
            dists = np.sqrt((self.map_data['x'] - mx)**2 + (self.map_data['y'] - my)**2)
            idx = np.argmin(dists)
            
            if dists[idx] < 20.0:
                x, y = self.map_data['x'][idx], self.map_data['y'][idx]
                p, v, s = self.map_data['p'][idx], self.map_data['v'][idx], self.map_data['s'][idx]
                
                self.v_line.setPos(x)
                self.h_line.setPos(y)
                
                color_hex = "#00ff00" if p > 0 else "#ff5555" if p < 0 else "#ffffff"
                html = (f"<div style='background-color: rgba(0,0,0,150); padding: 5px; font-size: 14px;'>"
                        f"<b>Dist: {s:.1f} m</b><br>Speed: {v:.1f} km/h<br>"
                        f"Power: <b style='color: {color_hex};'>{p:.1f} kW</b></div>")
                
                self.cursor_label.setHtml(html)
                self.cursor_label.setPos(x, y)
                self.cursor_label.show()
                self.v_line.show()
                self.h_line.show()
            else:
                self.cursor_label.hide()
                self.v_line.hide()
                self.h_line.hide()

    def on_save_clicked(self):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        default_name = f"{self.track.name}_Endurance_{now_str}.npz"
        if not os.path.exists("results"): os.makedirs("results")
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Endurance Results", 
                                                  os.path.join("results", default_name), 
                                                  "NumPy Zip (*.npz)")
        
        if filepath:
            try:
                # currently saving only the selected lap
                save_results_to_npz(filepath, self.current_res, self.track.name, self.ds)
                self.btn_save.setText("Saved Current Lap!")
            except Exception as e:
                print(f"Error: {e}")