import numpy as np
import pyqtgraph as pg
import optuna
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QTabWidget, QWidget, QComboBox, QSlider)
from PyQt6.QtCore import Qt

class OptimizationResultsDialog(QDialog):
    C_SIM = '#3b82f6'
    C_BEST = '#10b981'
    C_REF = '#334155'
    
    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self.payload = payload
        
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', '#334155')
        pg.setConfigOptions(antialias=True)
        
        self.setWindowTitle("Optimization results")
        self.resize(1100, 850)
        
        self.setStyleSheet("""
            QDialog {
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
                background-color: #64748b; 
                color: white; 
                font-weight: bold; 
                padding: 8px 15px; 
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QComboBox {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 5px;
                min-width: 150px;
                background: white;
            }
        """)
        
        self._prepare_data()
        self._init_ui()

    def _prepare_data(self):
        # Načtení dat z Optuny
        df = self.payload.study.trials_dataframe()
        self.df = df[df['state'] == 'COMPLETE'].copy()
        
        self.trials = self.df['number'].values
        self.values = self.df['value'].values
        self.best_values = np.minimum.accumulate(self.values)
        
        # Identifikace sloupců s parametry
        self.param_cols = [c for c in self.df.columns if c.startswith('params_')]
        self.param_names = [c.replace('params_', '') for c in self.param_cols]
        
        # Globální minima a maxima pro osu Y, abychom mohli fixovat grafy
        self.y_min = self.df['value'].min()
        self.y_max = self.df['value'].max()

    def _style_plot(self, plot, title, y_label, x_label, units=None):

        plot.setBackground("#FFFFFF")
        plot.setTitle(title, color='#0f172a', size='12pt', bold=True)
        if units:
            plot.setLabel('left', y_label, units=units, color='#64748b')
        else:
            plot.setLabel('left', y_label, color='#64748b')
        plot.setLabel('bottom', x_label, color='#64748b')
        
        ax_pen = pg.mkPen(color='#94a3b8', width=1)
        plot.getAxis('left').setPen(ax_pen)
        plot.getAxis('left').setTextPen('#475569')
        plot.getAxis('bottom').setPen(ax_pen)
        plot.getAxis('bottom').setTextPen('#475569')
        
        plot.showGrid(x=True, y=True, alpha=0.15)
        return plot

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # --- 1. Detailní Summary Panel ---
        summary = QGroupBox(f"Optimization summary (Base: {self.payload.base_vehicle_name})")
        s_layout = QVBoxLayout(summary)
        
        lbl_time = QLabel(f"<b>Best total time:</b> {self.payload.best_lap_time:.3f} s")
        lbl_time.setStyleSheet("font-size: 14pt; color: #10b981;")
        s_layout.addWidget(lbl_time)
        
        params_list = [f"<b>{k}</b>: {v:.4f}" for k, v in self.payload.best_params.items()]
        params_str = "  |  ".join(params_list)
        lbl_params = QLabel(f"<b>Best parameters:</b><br>{params_str}")
        lbl_params.setWordWrap(True)
        lbl_params.setStyleSheet("font-size: 11pt; color: #334155;")
        s_layout.addWidget(lbl_params)
        
        layout.addWidget(summary)

        # --- 2. Záložky s grafy ---
        self.tabs = QTabWidget()
        
        self.init_history_tab()
        self.init_analysis_tab()
        self.init_importance_tab()
        
        layout.addWidget(self.tabs)
        
        # --- 3. Zavírací tlačítko ---
        btn_close = QPushButton("Close")
        btn_close.setMinimumWidth(120)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

    def init_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        plot = pg.PlotWidget()
        self._style_plot(plot, "Optimization History", "Objective Value (Lap Time)", "Trial Number", "s")
        
        # Bodový graf pro všechny trialy
        scatter = pg.ScatterPlotItem(
            x=self.trials, y=self.values, 
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush(self.C_SIM + '88')
        )
        plot.addItem(scatter)
        
        # Křivka nejlepšího dosaženého výsledku
        plot.plot(self.trials, self.best_values, pen=pg.mkPen(self.C_BEST, width=3), name="Best Value")
        
        layout.addWidget(plot)
        self.tabs.addTab(widget, "History")

    def init_analysis_tab(self):
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        
        # Levý panel - Graf
        plot_layout = QVBoxLayout()
        
        self.combo_x = QComboBox()
        self.combo_x.addItems(self.param_names)
        self.combo_x.currentIndexChanged.connect(self.update_analysis_plot)
        
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>X-Axis parameter:</b>"))
        header.addWidget(self.combo_x)
        header.addStretch()
        plot_layout.addLayout(header)
        
        self.plot_slice = pg.PlotWidget()
        self.slice_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('#ffffff', width=0.5), brush=pg.mkBrush(self.C_SIM + 'AA'))
        self.plot_slice.addItem(self.slice_scatter)
        plot_layout.addWidget(self.plot_slice)
        
        main_layout.addLayout(plot_layout, stretch=3)
        
        # Pravý panel - Filtrační slidery
        self.filter_group = QGroupBox("Slice filters")
        self.filter_layout = QVBoxLayout(self.filter_group)
        
        self.sliders = {}
        self.slider_labels = {}
        
        for p_name in self.param_names:
            p_min = self.df[f"params_{p_name}"].min()
            p_max = self.df[f"params_{p_name}"].max()
            
            p_layout = QVBoxLayout()
            lbl = QLabel(f"<b>{p_name}</b>")
            val_lbl = QLabel(f"Range: {p_min:.3f} - {p_max:.3f}")
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50) 
            slider.valueChanged.connect(self.update_analysis_plot)
            
            p_layout.addWidget(lbl)
            p_layout.addWidget(slider)
            p_layout.addWidget(val_lbl)
            self.filter_layout.addLayout(p_layout)
            
            self.sliders[p_name] = slider
            self.slider_labels[p_name] = val_lbl

        self.filter_layout.addStretch()
        main_layout.addWidget(self.filter_group, stretch=1)
        
        self.tabs.addTab(widget, "Dynamic slice")
        self.update_analysis_plot()

    def update_analysis_plot(self):
        x_name = self.combo_x.currentText()
        if not x_name: return
        
        p_col_x = f"params_{x_name}"
        
        # --- Zafixování měřítka os (SCALE FIX) ---
        x_min_global = self.df[p_col_x].min()
        x_max_global = self.df[p_col_x].max()
        
        # Vytvoření malého okraje (paddingu) v grafu (5 %)
        x_margin = (x_max_global - x_min_global) * 0.05 if x_max_global != x_min_global else 1.0
        y_margin = (self.y_max - self.y_min) * 0.05 if self.y_max != self.y_min else 1.0
        
        # Nastavení fixních rozsahů
        self.plot_slice.setXRange(x_min_global - x_margin, x_max_global + x_margin, padding=0)
        self.plot_slice.setYRange(self.y_min - y_margin, self.y_max + y_margin, padding=0)
        
        # --- Aplikace filtrů ---
        tolerance = 0.1 # Tolerance filtru +/- 5%
        mask = np.ones(len(self.df), dtype=bool)
        
        for p_name, slider in self.sliders.items():
            if p_name == x_name:
                self.slider_labels[p_name].setText("Current X-Axis")
                slider.setEnabled(False)
                continue
            
            slider.setEnabled(True)
            p_col = f"params_{p_name}"
            p_min, p_max = self.df[p_col].min(), self.df[p_col].max()
            
            target_val = p_min + (slider.value() / 100.0) * (p_max - p_min)
            margin = (p_max - p_min) * tolerance
            
            mask &= (self.df[p_col] >= target_val - margin) & (self.df[p_col] <= target_val + margin)
            self.slider_labels[p_name].setText(f"Value: ~{target_val:.3f}")

        filtered_df = self.df[mask]
        
        # Aktualizace bodů v grafu
        self.slice_scatter.setData(
            x=filtered_df[p_col_x].values, 
            y=filtered_df['value'].values
        )
        
        self._style_plot(self.plot_slice, f"Slice: {x_name} (Showing {len(filtered_df)} trials)", "Lap Time", x_name, "s")

    def init_importance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        plot = pg.PlotWidget()
        self._style_plot(plot, "Hyperparameter Importance", "Importance", "Parameters")
        
        try:
            # Optuna interně vypočítá důležitost parametrů (často využívá algoritmus fANOVA)
            importances = optuna.importance.get_param_importances(self.payload.study)
            
            # Očištění jmen a seřazení od nejvyšší po nejnižší důležitost
            clean_importances = {k.replace('params_', ''): v for k, v in importances.items()}
            sorted_items = sorted(clean_importances.items(), key=lambda x: x[1], reverse=True)
            
            x_pos = np.arange(len(sorted_items))
            y_vals = [val for key, val in sorted_items]
            labels = [key for key, val in sorted_items]
            
            # Vytvoření BarGraph (sloupcového grafu)
            bg = pg.BarGraphItem(x=x_pos, height=y_vals, width=0.6, brush=pg.mkBrush(self.C_SIM))
            plot.addItem(bg)
            
            # Nastavení textových popisků na osu X
            ticks = [list(zip(x_pos, labels))]
            plot.getAxis('bottom').setTicks(ticks)
            
        except Exception as e:
            # Fallback pro případ, že Optuna selže (např. chybí sklearn knihovna)
            err_lbl = QLabel(f"Could not calculate importances.\n(Make sure 'scikit-learn' is installed).\nError: {e}")
            err_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err_lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
            layout.addWidget(err_lbl)

        layout.addWidget(plot)
        self.tabs.addTab(widget, "Parameter importance")