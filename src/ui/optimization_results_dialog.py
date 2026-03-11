import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
try:
    import optuna
except ImportError:
    pass

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QTabWidget, QWidget, QComboBox, 
                             QSlider, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt

class OptimizationResultsDialog(QDialog):
    C_SIM = '#3b82f6'
    C_BEST = '#10b981'
    C_REF = '#334155'
    
    def __init__(self, payload=None, json_data=None, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.json_data = json_data
        
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
        if self.payload:
            # data from live optuna oject
            df = self.payload.study.trials_dataframe()
            self.df = df[df['state'] == 'COMPLETE'].copy()
            
            self.base_vehicle_name = self.payload.base_vehicle_name
            self.best_lap_time = self.payload.best_lap_time
            self.best_params = self.payload.best_params
            self.tracks = self.payload.tracks
            self.base_car_params = self.payload.base_car_params
            
            try:
                importances = optuna.importance.get_param_importances(self.payload.study)
                self.importances = {k.replace('params_', ''): v for k, v in importances.items()}
            except Exception as e:
                print(f"Could not calculate importances: {e}")
                self.importances = {}
                
        elif self.json_data:
            # Data from laoded JSON file
            self.df = pd.DataFrame(self.json_data['trials'])
            
            meta = self.json_data.get('metadata', {})
            self.base_vehicle_name = meta.get('base_vehicle_name', 'Unknown')
            self.best_lap_time = meta.get('best_lap_time', 0.0)
            self.tracks = meta.get('tracks_used', [])
            
            self.best_params = self.json_data.get('best_params', {})
            self.base_car_params = self.json_data.get('base_car_params', {})
            self.importances = self.json_data.get('parameter_importances', {})
        else:
            raise ValueError("Must provide either payload or json_data")

        self.trials = self.df['number'].values
        self.values = self.df['value'].values
        self.best_values = np.minimum.accumulate(self.values)
        
        self.param_cols = [c for c in self.df.columns if c.startswith('params_')]
        self.param_names = [c.replace('params_', '') for c in self.param_cols]
        
        self.y_min = self.df['value'].min()
        self.y_max = self.df['value'].max()

    def _style_plot(self, plot, title, y_label, x_label, units=None):
        plot.setBackground("#FFFFFF")
        plot.setTitle(title, color='#0f172a', size='12pt', bold=True)
        plot.setLabel('left', y_label, units=units, color='#64748b') if units else plot.setLabel('left', y_label, color='#64748b')
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
        
        summary = QGroupBox(f"Optimization summary (Base: {self.base_vehicle_name})")
        s_layout = QVBoxLayout(summary)
        
        lbl_time = QLabel(f"<b>Best total time:</b> {self.best_lap_time:.3f} s")
        lbl_time.setStyleSheet("font-size: 14pt; color: #10b981;")
        s_layout.addWidget(lbl_time)
        
        params_list = [f"<b>{k}</b>: {v:.4f}" for k, v in self.best_params.items()]
        lbl_params = QLabel(f"<b>Best parameters:</b><br>{'  |  '.join(params_list)}")
        lbl_params.setWordWrap(True)
        lbl_params.setStyleSheet("font-size: 11pt; color: #334155;")
        s_layout.addWidget(lbl_params)
        
        layout.addWidget(summary)

        self.tabs = QTabWidget()
        self.init_history_tab()
        self.init_analysis_tab()
        self.init_importance_tab()
        layout.addWidget(self.tabs)
        
        btn_layout = QHBoxLayout()
        
       
        if self.payload:
            btn_save = QPushButton("Save results")
            btn_save.setMinimumWidth(120)
            btn_save.setStyleSheet("background-color: #10b981; color: white;")
            btn_save.clicked.connect(self.save_results)
            btn_layout.addWidget(btn_save)
            
        btn_layout.addStretch()
        
        btn_close = QPushButton("Close")
        btn_close.setMinimumWidth(120)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)

    def save_results(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"{self.base_vehicle_name}_opt_{timestamp}.json"
        
        save_dir = os.path.join("data", "optimizations")
        os.makedirs(save_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save optimization results", os.path.join(save_dir, default_filename), "JSON Files (*.json)")
        
        if not file_path: return
            
        df_export = self.df.copy()
        for col in df_export.select_dtypes(include=['datetime', 'timedelta']).columns:
            df_export[col] = df_export[col].astype(str)
            
        data = {
            "metadata": {
                "timestamp": timestamp,
                "base_vehicle_name": self.base_vehicle_name,
                "tracks_used": self.tracks,
                "best_lap_time": self.best_lap_time
            },
            "base_car_params": self.base_car_params,
            "best_params": self.best_params,
            "parameter_importances": self.importances,
            "trials": df_export.to_dict(orient='records')
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
            QMessageBox.information(self, "Success", f"Results saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results:\n{str(e)}")

    def init_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        plot = pg.PlotWidget()
        self._style_plot(plot, "Optimization history", "Objective value (Lap Time)", "Trial number", "s")
        plot.addItem(pg.ScatterPlotItem(x=self.trials, y=self.values, size=10, pen=pg.mkPen(None), brush=pg.mkBrush(self.C_SIM + '88')))
        plot.plot(self.trials, self.best_values, pen=pg.mkPen(self.C_BEST, width=3), name="Best Value")
        layout.addWidget(plot)
        self.tabs.addTab(widget, "History")

    def init_analysis_tab(self):
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        
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
        
        self.filter_group = QGroupBox("Slice filters")
        self.filter_layout = QVBoxLayout(self.filter_group)
        self.sliders = {}
        self.slider_labels = {}
        
        for p_name in self.param_names:
            p_min, p_max = self.df[f"params_{p_name}"].min(), self.df[f"params_{p_name}"].max()
            p_layout = QVBoxLayout()
            lbl = QLabel(f"<b>{p_name}</b>")
            val_lbl = QLabel(f"Range: {p_min:.3f} - {p_max:.3f}")
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100); slider.setValue(50)
            slider.valueChanged.connect(self.update_analysis_plot)
            
            p_layout.addWidget(lbl); p_layout.addWidget(slider); p_layout.addWidget(val_lbl)
            self.filter_layout.addLayout(p_layout)
            self.sliders[p_name] = slider; self.slider_labels[p_name] = val_lbl

        self.filter_layout.addStretch()
        main_layout.addWidget(self.filter_group, stretch=1)
        self.tabs.addTab(widget, "Dynamic slice")
        self.update_analysis_plot()

    def update_analysis_plot(self):
        x_name = self.combo_x.currentText()
        if not x_name: return
        p_col_x = f"params_{x_name}"
        x_min, x_max = self.df[p_col_x].min(), self.df[p_col_x].max()
        
        x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
        y_margin = (self.y_max - self.y_min) * 0.05 if self.y_max != self.y_min else 1.0
        
        self.plot_slice.setXRange(x_min - x_margin, x_max + x_margin, padding=0)
        self.plot_slice.setYRange(self.y_min - y_margin, self.y_max + y_margin, padding=0)
        
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
            margin = (p_max - p_min) * 0.1
            mask &= (self.df[p_col] >= target_val - margin) & (self.df[p_col] <= target_val + margin)
            self.slider_labels[p_name].setText(f"Value: ~{target_val:.3f}")

        filtered_df = self.df[mask]
        self.slice_scatter.setData(x=filtered_df[p_col_x].values, y=filtered_df['value'].values)
        self._style_plot(self.plot_slice, f"Slice: {x_name} (Showing {len(filtered_df)} trials)", "Lap Time", x_name, "s")

    def init_importance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        plot = pg.PlotWidget()
        self._style_plot(plot, "Hyperparameter importance", "Importance", "Parameters")
        
        if self.importances:
            sorted_items = sorted(self.importances.items(), key=lambda x: x[1], reverse=True)
            x_pos = np.arange(len(sorted_items))
            y_vals = [val for _, val in sorted_items]
            labels = [key for key, _ in sorted_items]
            
            plot.addItem(pg.BarGraphItem(x=x_pos, height=y_vals, width=0.6, brush=pg.mkBrush(self.C_SIM)))
            plot.getAxis('bottom').setTicks([list(zip(x_pos, labels))])
        else:
            err_lbl = QLabel("Importance data is not available for this run.")
            err_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err_lbl.setStyleSheet("color: #64748b; font-weight: bold; font-size: 12pt;")
            layout.addWidget(err_lbl)

        layout.addWidget(plot)
        self.tabs.addTab(widget, "Parameter importance")