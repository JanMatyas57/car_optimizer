import os
import sys
import numpy as np
import casadi as ca
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSlider, QLabel, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QGroupBox,
    QDialog, QCheckBox, QDialogButtonBox, QRadioButton, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

class InterpolateDialog(QDialog):
    def __init__(self, parent, target_coords, lap_times, grids, dims):
        super().__init__(parent)
        self.setWindowTitle("Fix Outlier by Interpolation")
        self.setMinimumWidth(400)
        self.lap_times = lap_times
        self.target_coords = target_coords
        self.dims = dims
        self.grids = grids
        self.selected_value = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        coord_strs = []
        for d, dim in enumerate(self.dims):
            val = self.grids[dim][self.target_coords[d]]
            coord_strs.append(f"{dim}={val:.2f}")
            
        layout.addWidget(QLabel(f"<b>Target Point:</b> {', '.join(coord_strs)}"))
        
        current_val = self.lap_times[self.target_coords]
        layout.addWidget(QLabel(f"<b>Current Value:</b> {current_val:.3f} s" if not np.isnan(current_val) else "<b>Current Value:</b> NaN"))
        
        layout.addSpacing(10)
        self.radio_interpolate = QRadioButton("Interpolate from selected neighbours:")
        self.radio_interpolate.setChecked(True)
        layout.addWidget(self.radio_interpolate)
        
        self.neighbour_checkboxes = []
        
        for d, dim in enumerate(self.dims):
            idx = self.target_coords[d]
            for offset in [-1, 1]:
                n_idx = idx + offset
                if 0 <= n_idx < self.lap_times.shape[d]:
                    n_coords = list(self.target_coords)
                    n_coords[d] = n_idx
                    n_coords = tuple(n_coords)
                    
                    n_val = self.lap_times[n_coords]
                    if not np.isnan(n_val):
                        n_coord_val = self.grids[dim][n_idx]
                        cb = QCheckBox(f"{dim} offset to {n_coord_val:.2f}  ->  {n_val:.3f} s")
                        cb.n_val = n_val
                        cb.setChecked(True)
                        layout.addWidget(cb)
                        self.neighbour_checkboxes.append(cb)
                        
        if not self.neighbour_checkboxes:
            layout.addWidget(QLabel("<i>No valid immediate neighbours found!</i>"))
            
        layout.addSpacing(10)
        self.radio_manual = QRadioButton("Enter manual value (s):")
        layout.addWidget(self.radio_manual)
        
        self.spin_manual = QDoubleSpinBox()
        self.spin_manual.setDecimals(3)
        self.spin_manual.setRange(0.0, 1000.0)
        if not np.isnan(current_val):
            self.spin_manual.setValue(current_val)
        else:
            self.spin_manual.setValue(60.0)
        layout.addWidget(self.spin_manual)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
    def accept(self):
        if self.radio_manual.isChecked():
            self.selected_value = self.spin_manual.value()
        else:
            vals = [cb.n_val for cb in self.neighbour_checkboxes if cb.isChecked()]
            if vals:
                self.selected_value = float(np.mean(vals))
            else:
                self.selected_value = np.nan
        super().accept()


class DatasetCleanerTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battery Dataset Cleaner & Analyzer")
        self.resize(1400, 900)
        
        self.data = None
        self.grids = {}
        self.lap_times = None
        self.dims = []
        
        self.setup_ui()
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- Control Panel ---
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        self.btn_load = QPushButton("Load NPZ")
        self.btn_load.clicked.connect(self.load_file)
        
        self.btn_save = QPushButton("Save Cleaned NPZ")
        self.btn_save.setStyleSheet("background-color: #10b981;") # Green for save
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)
        
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_x.currentTextChanged.connect(self.update_axes_and_slider)
        self.combo_y.currentTextChanged.connect(self.update_axes_and_slider)
        
        self.lbl_slider = QLabel("Slice:")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.update_plot)
        self.lbl_slider_val = QLabel("")
        
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_save)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("X-Axis:"))
        control_layout.addWidget(self.combo_x)
        control_layout.addWidget(QLabel("Y-Axis:"))
        control_layout.addWidget(self.combo_y)
        control_layout.addWidget(self.lbl_slider)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.lbl_slider_val)
        control_layout.addStretch()
        
        main_layout.addWidget(control_group)
        
        # --- Content Area ---
        content_layout = QHBoxLayout()
        
        # Plot Area
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        content_layout.addWidget(plot_container, stretch=3)
        
        # Tables Area
        self.tabs = QTabWidget()
        self.table_min = QTableWidget()
        self.table_max = QTableWidget()
        self.table_nan = QTableWidget()
        
        self.table_min.cellDoubleClicked.connect(lambda r, c: self.on_cell_double_clicked(self.table_min, r))
        self.table_max.cellDoubleClicked.connect(lambda r, c: self.on_cell_double_clicked(self.table_max, r))
        self.table_nan.cellDoubleClicked.connect(lambda r, c: self.on_cell_double_clicked(self.table_nan, r))
        
        self.tabs.addTab(self.table_min, "Top 20 Min")
        self.tabs.addTab(self.table_max, "Top 20 Max")
        self.tabs.addTab(self.table_nan, "All NaNs")
        
        content_layout.addWidget(self.tabs, stretch=2)
        
        main_layout.addLayout(content_layout)
        
        self.apply_styles()
        
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                font-family: 'Segoe UI', system-ui;
                font-size: 10pt;
                background-color: #f8f9fa;
                color: #334155;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 6px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QGroupBox {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #475569;
                font-weight: bold;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
            }
            QHeaderView::section {
                background-color: #e2e8f0;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #cbd5e1;
            }
        """)
        
    def load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Battery Grid NPZ", "", "NPZ Files (*.npz);;All Files (*.*)"
        )
        if not filepath:
            return
            
        try:
            self.data = dict(np.load(filepath, allow_pickle=True))
            # Create a copy so we can edit it and resave without mutating the dict reference weirdly
            self.lap_times = self.data['lap_times'].copy()
            
            self.btn_save.setEnabled(True)
            
            # Identify dimensions
            self.grids = {}
            self.dims = []
            if 'soc_grid' in self.data:
                self.grids['SoC'] = self.data['soc_grid'] * 100  # convert to %
                self.dims.append('SoC')
            if 'energy_grid' in self.data:
                self.grids['Energy'] = self.data['energy_grid']
                self.dims.append('Energy')
            
            if self.lap_times.ndim == 3 and 'regen_grid' in self.data:
                self.grids['Regen'] = self.data['regen_grid'] * 100
                self.dims.append('Regen')
                
            # Interpolant for smooth plotting MUST be created before dropdowns trigger plots!
            flat_for_casadi = self.lap_times.ravel(order='F')
            grid_axes = [self.data['soc_grid'].tolist(), self.data['energy_grid'].tolist()]
            if self.lap_times.ndim == 3:
                grid_axes.append(self.data['regen_grid'].tolist())
                
            self.interpolant = ca.interpolant('laptime_map', 'bspline', grid_axes, flat_for_casadi.tolist())
                
            self.populate_dropdowns()
            self.populate_tables()
            
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            import traceback
            traceback.print_exc()
            
    def populate_dropdowns(self):
        self.combo_x.blockSignals(True)
        self.combo_y.blockSignals(True)
        
        self.combo_x.clear()
        self.combo_y.clear()
        
        self.combo_x.addItems(self.dims)
        self.combo_y.addItems(self.dims)
        
        if len(self.dims) >= 2:
            self.combo_x.setCurrentText(self.dims[1])  # Energy
            self.combo_y.setCurrentText(self.dims[0])  # SoC
            
        self.combo_x.blockSignals(False)
        self.combo_y.blockSignals(False)
        self.update_axes_and_slider()
        
    def update_axes_and_slider(self):
        x_dim = self.combo_x.currentText()
        y_dim = self.combo_y.currentText()
        
        if x_dim == y_dim:
            return  # Invalid, ignore until user changes
            
        # Find remaining dim
        remaining_dims = [d for d in self.dims if d not in (x_dim, y_dim)]
        
        if remaining_dims:
            self.slice_dim = remaining_dims[0]
            grid = self.grids[self.slice_dim]
            self.lbl_slider.setText(f"Slice by {self.slice_dim}:")
            
            self.slider.blockSignals(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(grid) - 1)
            # Find closest to sensible defaults
            default_val = grid[-1]
            if self.slice_dim == 'Regen':
                default_val = 80.0
            idx = np.abs(grid - default_val).argmin()
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            
            self.slider.setEnabled(True)
            self.lbl_slider.show()
            self.slider.show()
            self.lbl_slider_val.show()
        else:
            self.slice_dim = None
            self.slider.setEnabled(False)
            self.lbl_slider.hide()
            self.slider.hide()
            self.lbl_slider_val.hide()
            
        self.update_plot()
        
    def populate_tables(self):
        valid_mask = ~np.isnan(self.lap_times)
        valid_indices = np.where(valid_mask)
        nan_indices = np.where(~valid_mask)
        
        valid_lts = self.lap_times[valid_indices]
        
        sorted_idx = np.argsort(valid_lts)
        
        # Top 20 Min
        min_20_idx = sorted_idx[:20]
        self.fill_table(self.table_min, min_20_idx, valid_indices, valid_lts)
        
        # Top 20 Max
        max_20_idx = sorted_idx[-20:][::-1]
        self.fill_table(self.table_max, max_20_idx, valid_indices, valid_lts)
        
        # NaNs
        self.fill_table_nans(self.table_nan, nan_indices)
        
    def fill_table(self, table, selection_idx, all_valid_indices, valid_lts):
        table.clear()
        headers = ["Lap Time (s)"] + self.dims
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(selection_idx))
        
        for row, idx in enumerate(selection_idx):
            lt = valid_lts[idx]
            coord = tuple(all_valid_indices[col][idx] for col in range(len(self.dims)))
            
            item = QTableWidgetItem(f"{lt:.3f}")
            item.setData(Qt.ItemDataRole.UserRole, coord)
            table.setItem(row, 0, item)
            
            for col, dim in enumerate(self.dims):
                dim_idx = all_valid_indices[col][idx]
                val = self.grids[dim][dim_idx]
                table.setItem(row, col + 1, QTableWidgetItem(f"{val:.2f}"))
                
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
    def fill_table_nans(self, table, nan_indices):
        table.clear()
        if len(nan_indices) == 0 or len(nan_indices[0]) == 0:
            table.setRowCount(0)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Status"])
            return
            
        num_nans = len(nan_indices[0])
        headers = ["Status"] + self.dims
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(num_nans)
        
        for row in range(num_nans):
            coord = tuple(nan_indices[col][row] for col in range(len(self.dims)))
            
            item = QTableWidgetItem("NaN")
            item.setData(Qt.ItemDataRole.UserRole, coord)
            table.setItem(row, 0, item)
            
            for col, dim in enumerate(self.dims):
                dim_idx = nan_indices[col][row]
                val = self.grids[dim][dim_idx]
                table.setItem(row, col + 1, QTableWidgetItem(f"{val:.2f}"))
                
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def on_cell_double_clicked(self, table, row):
        item = table.item(row, 0)
        coord = item.data(Qt.ItemDataRole.UserRole)
        if coord is None:
            return
            
        dialog = InterpolateDialog(self, coord, self.lap_times, self.grids, self.dims)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_value is not None:
            self.lap_times[coord] = dialog.selected_value
            
            # Recreate interpolant
            flat_for_casadi = self.lap_times.ravel(order='F')
            grid_axes = [self.data['soc_grid'].tolist(), self.data['energy_grid'].tolist()]
            if self.lap_times.ndim == 3:
                grid_axes.append(self.data['regen_grid'].tolist())
            self.interpolant = ca.interpolant('laptime_map', 'bspline', grid_axes, flat_for_casadi.tolist())
            
            self.populate_tables()
            self.update_plot()
            
    def save_file(self):
        if self.data is None:
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned NPZ", "", "NPZ Files (*.npz)"
        )
        if not filepath:
            return
            
        save_dict = dict(self.data)
        save_dict['lap_times'] = self.lap_times
        
        try:
            np.savez(filepath, **save_dict)
            QMessageBox.information(self, "Success", f"Saved cleaned data to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def update_plot(self):
        if self.lap_times is None:
            return
            
        x_dim = self.combo_x.currentText()
        y_dim = self.combo_y.currentText()
        
        if x_dim == y_dim:
            return
            
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#ffffff')
        
        x_grid = self.data['energy_grid'] if x_dim == 'Energy' else (self.data['soc_grid'] if x_dim == 'SoC' else self.data['regen_grid'])
        y_grid = self.data['energy_grid'] if y_dim == 'Energy' else (self.data['soc_grid'] if y_dim == 'SoC' else self.data['regen_grid'])
        
        x_fine = np.linspace(x_grid.min(), x_grid.max(), 50)
        y_fine = np.linspace(y_grid.min(), y_grid.max(), 50)
        X, Y = np.meshgrid(x_fine, y_fine)
        
        pts = []
        for i, dim_name in enumerate(self.dims):
            if dim_name == x_dim:
                pts.append(X.ravel())
            elif dim_name == y_dim:
                pts.append(Y.ravel())
            else:
                slice_idx = self.slider.value()
                real_slice_val = self.data['energy_grid'][slice_idx] if self.slice_dim == 'Energy' else (self.data['soc_grid'][slice_idx] if self.slice_dim == 'SoC' else self.data['regen_grid'][slice_idx])
                self.lbl_slider_val.setText(f"{self.grids[self.slice_dim][slice_idx]:.2f}")
                pts.append(np.full_like(X.ravel(), real_slice_val))
                
                # Fetch raw grid slice for scatter points
                raw_X, raw_Y = np.meshgrid(x_grid, y_grid, indexing='ij')
                
                # Extract the 2D slice from the 3D lap_times array
                idx_map = [slice(None)] * len(self.dims)
                idx_map[i] = slice_idx
                raw_Z = self.lap_times[tuple(idx_map)]
                
                # Transpose if necessary
                if self.dims.index(x_dim) > self.dims.index(y_dim):
                    raw_Z = raw_Z.T
                    
                valid = ~np.isnan(raw_Z)
                if np.any(valid):
                    # Scale raw X/Y to %
                    raw_X_plot = raw_X * 100 if x_dim in ['SoC', 'Regen'] else raw_X
                    raw_Y_plot = raw_Y * 100 if y_dim in ['SoC', 'Regen'] else raw_Y
                    # We transpose raw_X_plot and raw_Y_plot because meshgrid 'ij' gives (nx, ny)
                    # and scatter expects flat arrays matching Z which has shape (nx, ny)
                    ax.scatter(raw_X_plot[valid], raw_Y_plot[valid], raw_Z[valid], 
                               color='#e11d48', s=30, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=10)

        pts = np.vstack(pts)
        Z = np.array(self.interpolant(pts)).reshape(X.shape)
        
        # Scale back to % for plot
        X_plot = X * 100 if x_dim in ['SoC', 'Regen'] else X
        Y_plot = Y * 100 if y_dim in ['SoC', 'Regen'] else Y
        
        # Determine global min/max for color scale
        z_min = float(np.nanmin(self.lap_times))
        z_max = float(np.nanmax(self.lap_times))
        
        surf = ax.plot_surface(X_plot, Y_plot, Z, cmap='viridis_r', edgecolor='none', alpha=0.85, vmin=z_min, vmax=z_max)
        
        # 2D case where there is no slice dim
        if len(self.dims) == 2:
            raw_X, raw_Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            raw_Z = self.lap_times
            if self.dims.index(x_dim) > self.dims.index(y_dim):
                raw_Z = raw_Z.T
            valid = ~np.isnan(raw_Z)
            if np.any(valid):
                raw_X_plot = raw_X * 100 if x_dim in ['SoC', 'Regen'] else raw_X
                raw_Y_plot = raw_Y * 100 if y_dim in ['SoC', 'Regen'] else raw_Y
                ax.scatter(raw_X_plot[valid], raw_Y_plot[valid], raw_Z[valid], 
                           color='#e11d48', s=30, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=10)

        ax.set_xlabel(f"{x_dim} {'(%)' if x_dim in ['SoC', 'Regen'] else '(kWh)'}", color='#475569', labelpad=10)
        ax.set_ylabel(f"{y_dim} {'(%)' if y_dim in ['SoC', 'Regen'] else '(kWh)'}", color='#475569', labelpad=10)
        ax.set_zlabel("Lap Time (s)", color='#475569', labelpad=10)
        
        ax.set_zlim(z_min, z_max)
        
        title_str = "Laptime Surface Map"
        if self.slice_dim:
            slice_idx = self.slider.value()
            slice_val = self.grids[self.slice_dim][slice_idx]
            title_str += f" ({self.slice_dim} = {slice_val:.2f}{'%' if self.slice_dim in ['SoC', 'Regen'] else ' kWh'})"
            
        ax.set_title(title_str, color='#475569', pad=15)
        
        ax.xaxis.set_tick_params(colors='#64748b')
        ax.yaxis.set_tick_params(colors='#64748b')
        ax.zaxis.set_tick_params(colors='#64748b')
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetCleanerTool()
    window.show()
    sys.exit(app.exec())
