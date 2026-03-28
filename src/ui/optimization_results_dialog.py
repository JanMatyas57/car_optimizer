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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QTabWidget, QWidget, QComboBox, 
                             QSlider, QFileDialog, QMessageBox, QScrollArea)
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
        
        # Match height to 95% of screen available height to account for window decorations/taskbar safely
        screen_geo = self.screen().availableGeometry()
        w, h = 1000, max(600, int(screen_geo.height() * 0.95))
        self.setGeometry((screen_geo.width() - w) // 2, screen_geo.y() + (screen_geo.height() - h) // 2, w, h)
       
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
            self.best_lap_time = meta.get('best_objective', meta.get('best_lap_time', 0.0))
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

        # Fit Gaussian Process
        if len(self.df) > 0 and len(self.param_cols) > 0:
            X = self.df[self.param_cols].values
            y = self.df['value'].values
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Simple GP kernel covering most parameter behavior combinations
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1)
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            try:
                self.gp.fit(X_scaled, y)
                self.gp_fitted = True
            except Exception as e:
                print(f"Warning: GP fitting failed: {e}")
                self.gp_fitted = False
        else:
            self.gp_fitted = False

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
        
        improvement_pct = (1.0 - self.best_lap_time) * 100.0
        sign = "+" if improvement_pct >= 0 else "-"
        lbl_time = QLabel(f"<b>Best objective:</b> {self.best_lap_time:.5f} &nbsp;&nbsp; (<b>{sign}{abs(improvement_pct):.2f}%</b> vs baseline)")
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
                "best_objective": self.best_lap_time,
                "improvement_pct": (1.0 - self.best_lap_time) * 100.0
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
        self._style_plot(plot, "Optimization history", "Objective function (normalized)", "Trial number")
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
        self.combo_x.currentIndexChanged.connect(self.update_sensitivity_plot)
        
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>X-Axis parameter:</b>"))
        header.addWidget(self.combo_x)
        header.addStretch()
        plot_layout.addLayout(header)
        
        self.plot_slice = pg.PlotWidget()
        
        self.gp_mean_curve = pg.PlotDataItem(pen=pg.mkPen('#e11d48', width=2, style=Qt.PenStyle.DashLine))
        self.gp_upper_curve = pg.PlotDataItem(pen=pg.mkPen(None))
        self.gp_lower_curve = pg.PlotDataItem(pen=pg.mkPen(None))
        self.gp_fill = pg.FillBetweenItem(self.gp_lower_curve, self.gp_upper_curve, brush=pg.mkBrush('#cffdff'))
        
        self.slice_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('#ffffff', width=0.5), brush=pg.mkBrush(self.C_SIM + 'AA'))
        
        self.plot_slice.addItem(self.gp_fill)
        self.plot_slice.addItem(self.gp_mean_curve)
        self.plot_slice.addItem(self.slice_scatter)
        plot_layout.addWidget(self.plot_slice)
        main_layout.addLayout(plot_layout, stretch=3)
        
        self.filter_group = QGroupBox("Slice filters")
        self.filter_layout = QVBoxLayout(self.filter_group)
        self.sliders = {}
        self.slider_labels = {}
        
        self.sens_group = QGroupBox("Sensitivity evaluation point")
        self.sens_layout = QVBoxLayout(self.sens_group)
        self.sens_sliders = {}
        self.sens_labels = {}
        
        for p_name in self.param_names:
            p_min, p_max = self.df[f"params_{p_name}"].min(), self.df[f"params_{p_name}"].max()
            
            # Slice Filter
            p_layout = QVBoxLayout()
            lbl = QLabel(f"<b>{p_name}</b>")
            val_lbl = QLabel(f"Range: {p_min:.3f} - {p_max:.3f}")
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100); slider.setValue(50)
            slider.valueChanged.connect(self.update_analysis_plot)
            
            p_layout.addWidget(lbl); p_layout.addWidget(slider); p_layout.addWidget(val_lbl)
            self.filter_layout.addLayout(p_layout)
            self.sliders[p_name] = slider; self.slider_labels[p_name] = val_lbl

            # Sensitivity Point
            s_layout = QVBoxLayout()
            s_lbl = QLabel(f"<b>{p_name}</b>")
            s_val_lbl = QLabel(f"Value: - (Sens: -)")
            
            s_slider = QSlider(Qt.Orientation.Horizontal)
            s_slider.setRange(0, 100); s_slider.setValue(50)
            s_slider.valueChanged.connect(self.update_sensitivity_plot)
            
            s_layout.addWidget(s_lbl); s_layout.addWidget(s_slider); s_layout.addWidget(s_val_lbl)
            self.sens_layout.addLayout(s_layout)
            self.sens_sliders[p_name] = s_slider; self.sens_labels[p_name] = s_val_lbl

        self.filter_layout.addStretch()
        self.sens_layout.addStretch()
        
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar_widget")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(self.filter_group)
        sidebar_layout.addWidget(self.sens_group)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(sidebar_widget)
        # Remove scroll area border and ensure white background everywhere inside
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #ffffff; }
            QWidget#qt_scrollarea_viewport { background-color: #ffffff; }
            QWidget#sidebar_widget { background-color: #ffffff; }
        """)
        
        main_layout.addWidget(scroll_area, stretch=1)
        
        self.tabs.addTab(widget, "Dynamic slice")
        self.update_analysis_plot()
        self.update_sensitivity_plot()

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
        
        # Update GP curve for current slice
        if getattr(self, 'gp_fitted', False):
            n_points = 100
            x_grid = np.linspace(x_min - x_margin, x_max + x_margin, n_points)
            
            X_pred = np.zeros((n_points, len(self.param_cols)))
            for i, p_name in enumerate(self.param_names):
                if p_name == x_name:
                    X_pred[:, i] = x_grid
                else:
                    slider = self.sliders[p_name]
                    p_col = f"params_{p_name}"
                    p_min_s, p_max_s = self.df[p_col].min(), self.df[p_col].max()
                    target_val = p_min_s + (slider.value() / 100.0) * (p_max_s - p_min_s)
                    X_pred[:, i] = target_val
                    
            try:
                X_pred_scaled = self.scaler.transform(X_pred)
                y_mean, y_std = self.gp.predict(X_pred_scaled, return_std=True)
                
                self.gp_mean_curve.setData(x=x_grid, y=y_mean)
                self.gp_upper_curve.setData(x=x_grid, y=y_mean + 1.96 * y_std)
                self.gp_lower_curve.setData(x=x_grid, y=y_mean - 1.96 * y_std)
                
                self.gp_mean_curve.show()
                self.gp_fill.show()
                
            except Exception as e:
                print(f"GP prediction failed: {e}")
                self.gp_mean_curve.hide()
                self.gp_fill.hide()
        else:
            self.gp_mean_curve.hide()
            self.gp_fill.hide()

        self._style_plot(self.plot_slice, f"Slice: {x_name} (Showing {len(filtered_df)} trials)", "Objective function (normalized)", x_name)

    def update_sensitivity_plot(self):
        if not getattr(self, 'gp_fitted', False):
            return

        x_name = self.combo_x.currentText()
        if not x_name: return

        # Construct N-dimensional evaluation point
        X_eval = np.zeros((1, len(self.param_cols)))
        for i, p_name in enumerate(self.param_names):
            slider = self.sens_sliders[p_name]
            p_col = f"params_{p_name}"
            p_min_s, p_max_s = self.df[p_col].min(), self.df[p_col].max()
            target_val = p_min_s + (slider.value() / 100.0) * (p_max_s - p_min_s)
            X_eval[0, i] = target_val

        # Calculate sensitivities
        sensitivities = self._calculate_sensitivities(X_eval)
        
        # Update labels
        if sensitivities is not None:
            for i, p_name in enumerate(self.param_names):
                slider = self.sens_sliders[p_name]
                p_col = f"params_{p_name}"
                p_min_s, p_max_s = self.df[p_col].min(), self.df[p_col].max()
                target_val = p_min_s + (slider.value() / 100.0) * (p_max_s - p_min_s)
                
                if p_name != x_name:
                    self.sens_labels[p_name].setText(f"Value: ~{target_val:.3f}   (Sens: {sensitivities[i]:+.3f} s/unit)")
                else:
                    self.sens_labels[p_name].setText(f"<b>[X-Axis]</b> Value: ~{target_val:.3f}   (Sens: {sensitivities[i]:+.3f} s/unit)")

    def _calculate_sensitivities(self, X_eval):
        """Calculates physical sensitivities analytically for the GP at X_eval (shape 1, D)"""
        if not getattr(self, 'gp_fitted', False):
            return None
            
        try:
            x_test_scaled = self.scaler.transform(X_eval)
            
            # Extract matern kernel and parameters
            matern_kernel = self.gp.kernel_.k1
            length_scale = matern_kernel.length_scale
            if isinstance(length_scale, float):
                length_scale = np.array([length_scale] * len(self.param_cols))
                
            X_train_scaled = self.gp.X_train_
            diff = x_test_scaled - X_train_scaled # shape (N, D)
            
            dist_sq = np.sum((diff / length_scale)**2, axis=1)
            r = np.sqrt(dist_sq)
            
            dk_dx_scaled = np.zeros((X_train_scaled.shape[0], len(self.param_cols)))
            sqrt5_r = np.sqrt(5) * r
            mask = r > 0
            coef = -(5.0 / 3.0) * (1.0 + sqrt5_r[mask]) * np.exp(-sqrt5_r[mask])
            
            for d in range(len(self.param_cols)):
                dk_dx_scaled[mask, d] = coef * diff[mask, d] / (length_scale[d]**2)
                
            alpha = self.gp.alpha_
            grad_scaled = np.dot(alpha, dk_dx_scaled) # shape (D,)
            
            y_std = self.gp._y_train_std
            x_scale = self.scaler.scale_
            
            grad_analytical = grad_scaled * y_std / x_scale
            return grad_analytical
            
        except Exception as e:
            print(f"Sensitivity calculation failed: {e}")
            return None

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