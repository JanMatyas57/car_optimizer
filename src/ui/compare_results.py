import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTabWidget)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from PyQt6.QtWidgets import QComboBox, QHBoxLayout
import seaborn as sns


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
       
        FigureCanvasQTAgg.updateGeometry(self)


class ComparisonPlotWindow(QWidget):
    def __init__(self, selected_data):
        super().__init__()
        self.selected_data = selected_data
        self.runs = []
        
        self.setWindowTitle("Log comparison")
        self.resize(1000, 800)
        
        # 1. Načtení dat
        self.load_data()
        
        # 2. Layout
        self.layout = QVBoxLayout(self)
        
        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_dynamics_tab()
        self.init_energy_tab()
        self.init_map_tab()
        
    def load_data(self):
        self.runs = []
        for item in self.selected_data:
            try:
                data = np.load(item['path'], allow_pickle=True)
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
                    'name': item['legend']
                }
                self.runs.append(run_dict)
                data.close()
            except Exception as e:
                print(f"Error: {e}")

        if not self.runs: return

        # Sort: nejrychlejší první
        self.runs.sort(key=lambda x: x['time'])
        self.ref_run = self.runs[0]

    def init_dynamics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Grafy
        plot_v = pg.PlotWidget(title="Speed (km/h)")
        plot_v.showGrid(x=True, y=True, alpha=0.1)
        plot_v.setLabel('left', 'Speed', units='km/h')
        plot_v.setLabel('bottom', 'Distance', units='m')
        
        plot_dt = pg.PlotWidget(title="Delta Time (s)")
        plot_dt.showGrid(x=True, y=True, alpha=0.1)
        plot_dt.setXLink(plot_v) # Jediné propojení X osy
        plot_dt.setLabel('left', 'Delta Time', units='s')
        plot_dt.setLabel('bottom', 'Distance', units='m')
        

        # Vykreslení
        for i, run in enumerate(self.runs):
            # Barva: Reference zelená, ostatní červená/modrá/žlutá
            color = 'b' if run is self.ref_run else (i, len(self.runs)) 
            width = 3 if run is self.ref_run else 2
            
            # Speed
            plot_v.plot(run['s'], run['v'] * 3.6, pen=pg.mkPen(color, width=width))
            
            # Delta Time
            # Musíme ošetřit délky polí, aby šlo odečítat
            min_len = min(len(run['t']), len(self.ref_run['t']))
            
            t_run = run['t'][:min_len]
            t_ref = self.ref_run['t'][:min_len]
            s_axis = run['s'][:min_len]
            
            delta = t_run - t_ref
            plot_dt.plot(s_axis, delta, pen=pg.mkPen(color, width=width))

        layout.addWidget(plot_v)
        layout.addWidget(plot_dt)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dynamics")

    def init_energy_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        plot_p = pg.PlotWidget(title="Power (kW)")
        plot_p.showGrid(x=True, y=True)
        plot_p.setLabel('left', 'Power', units='kW')
        plot_p.setLabel('bottom', 'Distance', units='m')
        
        plot_e = pg.PlotWidget(title="Energy (kWh)")
        plot_e.showGrid(x=True, y=True)
        plot_e.setXLink(plot_p)
        plot_e.setLabel('left', 'Energy', units='kWh')
        plot_e.setLabel('bottom', 'Distance', units='m')

        for i, run in enumerate(self.runs):
            color = 'b' if run is self.ref_run else (i, len(self.runs))
            width = 3 if run is self.ref_run else 2
            
            # Power Calc
            min_len = min(len(run['f_acc']), len(run['v']))
            f_tot = run['f_acc'][:min_len] + run['f_brake'][:min_len]
            v = run['v'][:min_len]
            pwr = (f_tot * v) / 1000.0
            s_axis = run['s'][:min_len]
            
            plot_p.plot(s_axis, pwr, pen=pg.mkPen(color, width=width))
            plot_e.plot(run['s'], run['E'], pen=pg.mkPen(color, width=width))

        layout.addWidget(plot_p)
        layout.addWidget(plot_e)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Energy")

    def init_map_tab(self):
        """
        Vytvoří záložku s Matplotlib mapou pro porovnání Delta V
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # --- Controls (Výběr kola pro porovnání) ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel(f"Reference Lap: {self.ref_run['name']}"))
        controls_layout.addWidget(QLabel("  vs  "))
        
        self.combo_compare = QComboBox()
        # Přidáme do výběru všechna kola kromě referenčního
        self.compare_candidates = [r for r in self.runs if r is not self.ref_run]
        
        if not self.compare_candidates:
            layout.addWidget(QLabel("Need at least 2 logs to show comparison map."))
            widget.setLayout(layout)
            self.tabs.addTab(widget, "Delta V Map")
            return

        for run in self.compare_candidates:
            self.combo_compare.addItem(f"{run['name']} (+{run['time'] - self.ref_run['time']:.2f}s)")
            
        self.combo_compare.currentIndexChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.combo_compare)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # --- Matplotlib Canvas ---
        self.map_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.map_canvas)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Delta V Map")
        
        # První vykreslení
        self.update_heatmap()

    def update_heatmap(self):
        """
        Vypočítá rozdíl rychlostí a vykreslí mapu.
        """
        if not self.compare_candidates: return

        idx = self.combo_compare.currentIndex()
        comp_run = self.compare_candidates[idx]
        ref_run = self.ref_run

        # 1. Interpolace dat (aby seděla na stejné 's')
        # Použijeme 's' z referenčního kola jako osu
        s_ref = ref_run['s']
        v_ref = ref_run['v'] * 3.6 # km/h
        
        # Interpolujeme rychlost porovnávaného kola na pozice reference
        v_comp_interp = np.interp(s_ref, comp_run['s'], comp_run['v']) * 3.6
        
        # 2. Delta V (Kladná = comp je rychlejší, Záporná = comp je pomalejší)
        delta_v = v_comp_interp - v_ref

        # 3. Vykreslení
        self.map_canvas.axes.clear()
        ax = self.map_canvas.axes
        
        # Vykreslíme trať pomocí Scatter plotu, kde barva = delta_v
        # Používáme colormap 'seismic' (modrá-bílá-červená) nebo 'coolwarm'
        # Normování: 0 musí být bílá/střed. 
        
        # Zjištění limitů pro symetrickou barevnou škálu
        max_diff = max(abs(np.min(delta_v)), abs(np.max(delta_v)))
        if max_diff == 0: max_diff = 1 # Ochrana proti dělení nulou
        
        norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff) 
        #cmap = sns.color_palette("pla", as_cmap=True)
        
        sc = ax.scatter(ref_run['x'], ref_run['y'], c=delta_v, cmap='plasma', s=10, norm=norm)
        
        ax.set_title(f"Speed delta")
        ax.set_aspect('equal')
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.3)
        
        # Colorbar (pokud už existuje, musíme ho odstranit, jinak se budou množit)
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        
        self.cbar = self.map_canvas.fig.colorbar(sc, ax=ax, label='Delta Speed [km/h]')
        
        # Červená = Pomalejší než ref, Modrá = Rychlejší než ref
        self.map_canvas.draw()