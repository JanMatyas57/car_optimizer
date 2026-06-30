import sys
import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox, 
                             QLabel, QMessageBox)

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.car import Car

class PlotBatteryModel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battery Model CasADi Interpolant Viewer")
        self.resize(800, 600)
        self.car = None
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load .npz File")
        self.btn_load.clicked.connect(self.load_file)
        
        self.combo_param = QComboBox()
        self.combo_param.addItems(["OCV", "R0", "R1", "R2", "tau1", "tau2", "soc_to_power (W)"])
        self.combo_param.currentIndexChanged.connect(self.plot_param)
        self.combo_param.setEnabled(False)
        
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(QLabel("Parameter:"))
        controls_layout.addWidget(self.combo_param)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Plot
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
    def load_file(self):
        initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'vehicles', 'battery'))
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Battery NPZ", 
            initial_dir, 
            "NumPy Archives (*.npz)"
        )
        if not filepath:
            return
            
        try:
            # Create a dummy car just to hold the battery model
            params = {
                "name": "Dummy", "weight": 290, "rotating_mass": 22.0, "cd": 2.06, 
                "cl": 5.64, "cd_drs": 1.17, "cl_drs": 4.2, "drs_allowed": False, 
                "power_max": 80000, "regen_eff": 0.94, "soc_start": 1.0, 
                "M_max": 30.0, "model_type": "mass_point", "energy_limit": None,
                "cop": 0.53, "wb": 1.53, "tw": 1.2, "wd": 0.48, "cogz": 0.27, "i": 11.46
            }
            self.car = Car(**params)
            
            # The label name doesn't matter for custom, it will just read the NPZ
            self.car.load_battery(filepath, battery_type="custom")
            
            self.combo_param.setEnabled(True)
            self.plot_param()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def plot_param(self):
        if not self.car or not self.car.battery:
            return
            
        param_name = self.combo_param.currentText()
        bat = self.car.battery
        
        # Plot from 0 to 1.1 as requested
        soc_array = np.linspace(0.0, 1.01, 500)
        y_array = []
        
        # Evaluate the CasADi interpolant for each point
        for s in soc_array:
            try:
                if param_name == "OCV":
                    val = bat.OCV(s)
                elif param_name == "R0":
                    val = bat.get_R0(s)
                elif param_name == "R1":
                    val = bat.get_R1(s)
                elif param_name == "R2":
                    val = bat.get_R2(s)
                elif param_name == "tau1":
                    val = bat.get_tau1(s)
                elif param_name == "tau2":
                    val = bat.get_tau2(s)
                elif param_name == "soc_to_power (W)":
                    val = self.car.soc_to_power_interp(s)
                else:
                    val = 0.0
                
                # Extract numerical value from CasADi DM
                y_array.append(float(val))
            except Exception as e:
                # If CasADi throws an error, store NaN
                y_array.append(np.nan)
                
        self.ax.clear()
        
        # Draw vertical lines for context
        self.ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label="SoC = 1.0 (Boundary)")
        self.ax.axvline(x=0.0, color='r', linestyle='--', alpha=0.5)
        
        # Plot data
        self.ax.plot(soc_array, y_array, 'b-', linewidth=2, label="Evaluated Interpolant")
        
        # Explicitly reset and scale y-axis
        valid_y = [y for y in y_array if not np.isnan(y)]
        if valid_y:
            ymin, ymax = min(valid_y), max(valid_y)
            if ymin == ymax:
                self.ax.set_ylim(ymin - 0.1, ymax + 0.1)
            else:
                margin = (ymax - ymin) * 0.05
                self.ax.set_ylim(ymin - margin, ymax + margin)
        
        self.ax.set_xlim(-0.05, 1.15)
        self.ax.set_title(f"CasADi B-spline Evaluation: {param_name}")
        self.ax.set_xlabel("State of Charge (SoC)")
        self.ax.set_ylabel(param_name)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotBatteryModel()
    window.show()
    sys.exit(app.exec())
