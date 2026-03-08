import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTabWidget, QComboBox, QHBoxLayout)
import json
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#ffffff') # pure white background
        self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout() # use maximum space
        FigureCanvasQTAgg.updateGeometry(self)


class ComparisonPlotWindow(QWidget):
    # modern color palette (tailwind css colors) for graphs
    PALETTE = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

    def __init__(self, selected_data):
        super().__init__()
        
        # global antialiasing for nicer graphs
        pg.setConfigOptions(antialias=True)
        
        self.selected_data = selected_data
        self.runs = []
        
        self.setWindowTitle("Telemetry Comparison")
        self.resize(1100, 850)
        
        # --- application of modern look (qss) ---
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                color: #334155;
                background-color: #f8f9fa;
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
                border-bottom: 2px solid #3b82f6; /* highlighting active tab */
            }
            QTabBar::tab:hover:!selected {
                background: #cbd5e1;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 15px;
                min-width: 180px;
                font-weight: bold;
                color: #0f172a;
            }
            QComboBox:hover {
                border: 1px solid #94a3b8;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)

        # 1. layout initialization
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)
        
        # 2. data loading
        self.load_data()
        
        # 3. data check
        if not self.runs:
            error_lbl = QLabel(
                "Could not load the selected logs.\n"
                "Please generate new simulations and try again."
            )
            error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #ef4444; background: transparent;")
            self.main_layout.addWidget(error_lbl)
            return

        # 4. tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        self.init_dynamics_tab()
        self.init_energy_tab()
        self.init_map_tab()
        self.init_parameters_tab()
        
    def load_data(self):
        self.runs = []
        for item in self.selected_data:
            try:
                data = np.load(item['path'], allow_pickle=True)
                
                # --- safe loading of json car parameters ---
                car_dict = {}
                if 'car_config' in data:
                    raw_config = data['car_config']
                    
                    # 1. output of raw data and type
                    print(f"--- DEBUG {item['legend']} ---")
                    print(f"1. Raw type: {type(raw_config)}")
                    print(f"2. Raw value: {repr(raw_config)}")
                    
                    try:
                        # extracting clean string
                        if hasattr(raw_config, 'item'):
                            raw_config = raw_config.item()
                             
                        car_dict = json.loads(str(raw_config))
                       
                    except Exception as e:
                        car_dict = {"Data status": f"Parse error: {e}"}
                else:
                    car_dict = {"Data status": "No car parameters saved"}

                run_dict = {
                    's': data['s'],
                    'v': data['v'],
                    't': data['t'],
                    'E': data['E'],
                    'x': data['x'], 
                    'y': data['y'], 
                    'f_acc': data['f_acc'],
                    'f_brake': data['f_brake'],
                    'time': float(data['lap_time']),
                    'name': item['legend'],
                    'car_dict': car_dict  # <-- now contains either json or error message
                }
                self.runs.append(run_dict)
                data.close()
            except Exception as e:
                print(f"Error loading {item['path']}: {e}")

        if not self.runs: return

        # sort: fastest first
        self.runs.sort(key=lambda x: x['time'])
        self.ref_run = self.runs[0]

    def _style_plot(self, plot, title, y_label, units):
        """helper method for clean pyqtgraph plot appearance"""
        plot.setBackground('#ffffff')
        plot.setTitle(title, color='#0f172a', size='12pt', bold=True)
        plot.setLabel('left', y_label, units=units, color='#64748b')
        plot.setLabel('bottom', 'Distance', units='m', color='#64748b')
        
        # coloring axes and text with soft grey
        ax_pen = pg.mkPen(color='#94a3b8', width=1)
        plot.getAxis('left').setPen(ax_pen)
        plot.getAxis('left').setTextPen('#475569')
        plot.getAxis('bottom').setPen(ax_pen)
        plot.getAxis('bottom').setTextPen('#475569')
        
        # soft grid
        plot.showGrid(x=True, y=True, alpha=0.15)
        return plot

    def init_dynamics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # --- 1. speed graph ---
        plot_v = pg.PlotWidget()
        self._style_plot(plot_v, "Speed Telemetry", "Speed", "km/h")
        
        # modern legend placement
        legend = plot_v.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend.setLabelTextColor('#0f172a')
        
        # --- 2. delta time graph ---
        plot_dt = pg.PlotWidget()
        self._style_plot(plot_dt, "Delta Time (Accumulated Gap)", "Time", "s")
        plot_dt.setXLink(plot_v)
        
        # zero line (thin dark line)
        plot_dt.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#334155', width=1.5)))

        # rendering
        for i, run in enumerate(self.runs):
            # index 0 is always ref_run -> use blue
            color_hex = self.PALETTE[i % len(self.PALETTE)]
            color_val = pg.mkColor(color_hex)
            width = 2.5 if run is self.ref_run else 2.0
            
            # a) speed
            plot_v.plot(run['s'], run['v'] * 3.6, pen=pg.mkPen(color_val, width=width), name=f"({run['time']:.3f}s)")
            
            # b) delta time (fixed with interpolation)
            # take distance and time axes from reference lap
            s_ref = self.ref_run['s']
            t_ref = self.ref_run['t']
            
            # interpolate comparison lap time to reference lap distance points
            t_run_interp = np.interp(s_ref, run['s'], run['t'])
            
            # safely subtract as both arrays have the same length and correspond to same positions
            delta = t_run_interp - t_ref
            
            dt_curve = plot_dt.plot(s_ref, delta, pen=pg.mkPen(color_val, width=width), name=f"({run['time']:.3f}s)")
            
            # delta fill (with soft transparency)
            if run is not self.ref_run:
                # use s_ref instead of s_axis
                dummy_zero = pg.PlotCurveItem(s_ref, np.zeros_like(s_ref), pen=None)
                plot_dt.addItem(dummy_zero)
                
                fill_color = pg.mkColor(color_hex)
                fill_color.setAlpha(40) # soft 15% transparency (0-255)
                
                fill = pg.FillBetweenItem(dt_curve, dummy_zero, brush=fill_color)
                plot_dt.addItem(fill)

        layout.addWidget(plot_v)
        layout.addWidget(plot_dt)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dynamics")

    def init_energy_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        plot_p = pg.PlotWidget()
        self._style_plot(plot_p, "Powertrain Power", "Power", "kW")
        legend = plot_p.addLegend(offset=(20, 20), pen=pg.mkPen('#e2e8f0'), brush=pg.mkBrush('#ffffffEE'))
        legend.setLabelTextColor('#0f172a')
        
        plot_e = pg.PlotWidget()
        self._style_plot(plot_e, "Energy consumption", "Energy", "kWh")
        plot_e.setXLink(plot_p)

        for i, run in enumerate(self.runs):
            color_hex = self.PALETTE[i % len(self.PALETTE)]
            color_val = pg.mkColor(color_hex)
            width = 2.5 if run is self.ref_run else 2.0
            
            # power calc
            min_len = min(len(run['f_acc']), len(run['v']))
            f_tot = run['f_acc'][:min_len] + run['f_brake'][:min_len]
            v = run['v'][:min_len]
            pwr = (f_tot * v) / 1000.0
            s_axis = run['s'][:min_len]
            
            plot_p.plot(s_axis, pwr, pen=pg.mkPen(color_val, width=width), name=f"({run['time']:.3f}s)")
            
            # energy - fill under curve for better readability
            e_curve = plot_e.plot(run['s'], run['E']-run['E'][0], pen=pg.mkPen(color_val, width=width), name=f"({run['time']:.3f}s)")
            
            if run is self.ref_run:
                dummy_zero = pg.PlotCurveItem(run['s'], np.zeros_like(run['s']), pen=None)
                fill_color = pg.mkColor(color_hex)
                fill_color.setAlpha(20)
                plot_e.addItem(pg.FillBetweenItem(e_curve, dummy_zero, brush=fill_color))

        layout.addWidget(plot_p)
        layout.addWidget(plot_e)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Energy")

    def init_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # --- controls (lap selection for comparison) ---
        controls_layout = QHBoxLayout()
        
        lbl_ref = QLabel(f"Reference Lap: <b>{self.ref_run['name']}</b>")
        lbl_ref.setStyleSheet("font-size: 12pt; color: #0f172a; background: transparent;")
        controls_layout.addWidget(lbl_ref)
        
        lbl_vs = QLabel("  vs  ")
        lbl_vs.setStyleSheet("font-size: 11pt; color: #64748b; background: transparent;")
        controls_layout.addWidget(lbl_vs)
        
        self.combo_compare = QComboBox()
        self.compare_candidates = [r for r in self.runs if r is not self.ref_run]
        
        if not self.compare_candidates:
            layout.addWidget(QLabel("Need at least 2 logs to show comparison map."))
            widget.setLayout(layout)
            self.tabs.addTab(widget, "Delta V Map")
            return

        for run in self.compare_candidates:
            diff = run['time'] - self.ref_run['time']
            sign = "+" if diff > 0 else ""
            self.combo_compare.addItem(f"{run['name']} ({sign}{diff:.3f}s)")
            
        self.combo_compare.currentIndexChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.combo_compare)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # --- matplotlib canvas ---
        # wrap in qwidget with white background and border
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)
        
        self.map_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        canvas_layout.addWidget(self.map_canvas)
        
        layout.addWidget(canvas_container)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Delta V Map")
        
        # first render
        self.update_heatmap()

    def update_heatmap(self):
        if not self.compare_candidates: return

        idx = self.combo_compare.currentIndex()
        comp_run = self.compare_candidates[idx]
        ref_run = self.ref_run

        s_ref = ref_run['s']
        v_ref = ref_run['v'] * 3.6 
        v_comp_interp = np.interp(s_ref, comp_run['s'], comp_run['v']) * 3.6
        
        delta_v = v_comp_interp - v_ref

        self.map_canvas.axes.clear()
        ax = self.map_canvas.axes
        
        max_diff = max(abs(np.min(delta_v)), abs(np.max(delta_v)))
        if max_diff == 0: max_diff = 1 
        
        norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff) 
        
        sc = ax.scatter(ref_run['x'], ref_run['y'], c=delta_v, cmap='coolwarm', s=12, norm=norm)
        
        ax.set_title(f"Speed Delta [km/h]", pad=15, fontweight='bold', color='#0f172a')
        ax.set_aspect('equal')
        
        
        ax.axis('off')
        
        if not hasattr(self, 'cax'):
            # first time reduce main graph and create fixed axis on the right
            self.map_canvas.fig.subplots_adjust(right=0.85)
            self.cax = self.map_canvas.fig.add_axes([0.88, 0.15, 0.02, 0.7])
        else:
            # next redraw just clear legend axis, graph remains untouched
            self.cax.clear()
            
        # render legend into our fixed axis (cax instead of ax)
        self.cbar = self.map_canvas.fig.colorbar(sc, cax=self.cax)
        self.cbar.outline.set_visible(False)
        self.cbar.ax.tick_params(colors='#475569')
        
        self.map_canvas.draw()

    def init_parameters_tab(self):
        widget = QWidget()
        # horizontal layout stacks tables side by side
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        for run in self.runs:
            run_container = QWidget()
            run_layout = QVBoxLayout(run_container)
            run_layout.setContentsMargins(0, 0, 0, 0)
            
            # --- obtaining car name ---
            car_dict = run.get('car_dict', {})
            # create copy so we can safely pop 'name'
            display_dict = dict(car_dict) if isinstance(car_dict, dict) else {}
            car_name = display_dict.pop('name', 'Unknown Car')
            
            # table header with car name and simulation name
            lbl = QLabel(f"Vehicle: <b>{car_name}</b><br><span style='font-size: 10pt; color: #64748b;'>Log: {run['name']}</span>")
            lbl.setStyleSheet("font-size: 12pt; color: #0f172a; margin-bottom: 5px;")
            run_layout.addWidget(lbl)
            
            # the table itself
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Parameter", "Value"])
            
            # styling table for modern design
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            table.setStyleSheet("""
                QTableWidget {
                    background-color: #ffffff;
                    alternate-background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 6px;
                    font-size: 10pt;
                    color: #334155;
                }
                QHeaderView::section {
                    background-color: #f1f5f9;
                    padding: 8px;
                    border: none;
                    border-bottom: 1px solid #cbd5e1;
                    font-weight: bold;
                    color: #475569;
                }
                QTableWidget::item {
                    padding: 5px 10px;
                    border-bottom: 1px solid #f1f5f9;
                }
            """)

            table.setRowCount(len(display_dict))
            
            # data output (without 'name' key)
            for row, (key, value) in enumerate(display_dict.items()):
                item_key = QTableWidgetItem(str(key).replace('_', ' ').capitalize())
                item_val = QTableWidgetItem(str(value))
                
                # highlighting parameter text
                item_key.setForeground(pg.mkColor('#64748b'))
                item_val.setForeground(pg.mkColor('#0f172a'))
                
                table.setItem(row, 0, item_key)
                table.setItem(row, 1, item_val)
                
            run_layout.addWidget(table)
            layout.addWidget(run_container)
            
        self.tabs.addTab(widget, "Car Configs")