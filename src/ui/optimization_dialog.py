import os
import glob
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QScrollArea, QWidget, QCheckBox, QComboBox, 
                             QLabel, QDoubleSpinBox, QPushButton, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt
from src.utils.config_manager import ConfigManager
from src.ui.optimization_results_dialog import OptimizationResultsDialog
from src.ui.load_optimization_dialog import LoadOptimizationDialog

class OptimizationDialog(QDialog):
    def __init__(self, current_vehicle_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Car parameters optimization")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        self.current_vehicle_name = current_vehicle_name
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
       
        group_car = QGroupBox("1. Base vehicle")
        group_car.setStyleSheet("QGroupBox { border: 1px solid #3b82f6; }")
        layout_car = QHBoxLayout(group_car)
        
        self.combo_vehicle = QComboBox()
        vehicles = ConfigManager.get_available_vehicles()
        self.combo_vehicle.addItems(vehicles)
        if self.current_vehicle_name in vehicles:
            self.combo_vehicle.setCurrentText(self.current_vehicle_name)
            
        layout_car.addWidget(QLabel("Select base vehicle profile:"))
        layout_car.addWidget(self.combo_vehicle)
        
        layout_car.addSpacing(20)
        layout_car.addWidget(QLabel("Trials:"))
        self.spin_trials = QDoubleSpinBox()
        self.spin_trials.setDecimals(0)
        self.spin_trials.setRange(1.0, 10000.0)
        self.spin_trials.setValue(100.0)
        self.spin_trials.setSingleStep(10.0)
        layout_car.addWidget(self.spin_trials)
        
        layout.addWidget(group_car)
        
      
        group_tracks = QGroupBox("2. Tracks to evaluate")
        group_tracks.setStyleSheet("QGroupBox { border: 1px solid #3b82f6; }")
        layout_tracks = QVBoxLayout(group_tracks)
        
        scroll_tracks = QScrollArea()
        scroll_tracks.setWidgetResizable(True)
        scroll_tracks.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_tracks.viewport().setStyleSheet("background: transparent;")
        
        container_tracks = QWidget()
        container_tracks.setStyleSheet("background-color: #ffffff;")
        self.layout_track_list = QVBoxLayout(container_tracks)
        
        self.track_checkboxes = {}
        self.track_weights = {}
        self._load_tracks()
        
        self.layout_track_list.addStretch()
        scroll_tracks.setWidget(container_tracks)
        layout_tracks.addWidget(scroll_tracks)
        layout.addWidget(group_tracks)
        
      
        group_params = QGroupBox("3. Parameters bounds")
        group_params.setStyleSheet("QGroupBox { border: 1px solid #3b82f6; }")
        layout_params = QGridLayout(group_params)
        
        layout_params.addWidget(QLabel("Optimize"), 0, 0)
        layout_params.addWidget(QLabel("Parameter"), 0, 1)
        layout_params.addWidget(QLabel("Min bound"), 0, 2)
        layout_params.addWidget(QLabel("Max bound"), 0, 3)
        
        self.param_widgets = {}
      
        params_setup = [
            ("cop", "Center of pressure [0-1]", 0.40, 0.60, 0.01),
            ("wb", "Wheelbase [m]", 1.40, 1.65, 0.01),
            ("tw", "Car track width [m]", 1.15, 1.35, 0.01),
            ("wd", "Weight dist. front [0-1]", 0.45, 0.55, 0.01),
            ("cogz", "CoG height [m]", 0.20, 0.35, 0.01)
        ]
        
        for i, (key, label, def_min, def_max, step) in enumerate(params_setup, start=1):
            cb_enable = QCheckBox()
            cb_enable.setChecked(True) 
            
            lbl_name = QLabel(label)
            
            spin_min = QDoubleSpinBox()
            spin_min.setRange(0.0, 5.0)
            spin_min.setSingleStep(step)
            spin_min.setValue(def_min)
            
            spin_max = QDoubleSpinBox()
            spin_max.setRange(0.0, 5.0)
            spin_max.setSingleStep(step)
            spin_max.setValue(def_max)
            
            cb_enable.toggled.connect(spin_min.setEnabled)
            cb_enable.toggled.connect(spin_max.setEnabled)
            
            layout_params.addWidget(cb_enable, i, 0, alignment=Qt.AlignmentFlag.AlignCenter)
            layout_params.addWidget(lbl_name, i, 1)
            layout_params.addWidget(spin_min, i, 2)
            layout_params.addWidget(spin_max, i, 3)
            
            self.param_widgets[key] = {"enable": cb_enable, "min": spin_min, "max": spin_max}
            
        layout.addWidget(group_params)
        
   
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        
        self.btn_load_run = QPushButton("Load Previous Run")
        self.btn_load_run.setObjectName("BtnOptimize") 
        self.btn_load_run.clicked.connect(self.load_previous_run)
        
        self.btn_start = QPushButton("Start optimization")
        self.btn_start.setObjectName("BtnRunSingle") 
        self.btn_start.clicked.connect(self.validate_and_accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_load_run) 
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

    def _load_tracks(self):
        track_dir = "data/tracks"
        if os.path.exists(track_dir):
            files = glob.glob(os.path.join(track_dir, "*npz"))
            track_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
            for t_name in track_names:
                row_layout = QHBoxLayout()
                
                cb = QCheckBox(t_name)
                
                spin_weight = QDoubleSpinBox()
                spin_weight.setRange(0.01, 100.0)
                spin_weight.setValue(1.0)
                spin_weight.setSingleStep(0.1)
                spin_weight.setSuffix(" (Weight)")
                
                # Only enable weight input if track is selected
                cb.toggled.connect(spin_weight.setEnabled)
                spin_weight.setEnabled(cb.isChecked())
                
                row_layout.addWidget(cb)
                row_layout.addStretch()
                row_layout.addWidget(spin_weight)
                
                self.layout_track_list.addLayout(row_layout)
                self.track_checkboxes[t_name] = cb
                self.track_weights[t_name] = spin_weight

    def validate_and_accept(self):
    
        selected = [n for n, cb in self.track_checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one track for optimization.")
            return
            
      
        for key, w in self.param_widgets.items():
            if w["enable"].isChecked() and w["min"].value() >= w["max"].value():
                QMessageBox.warning(self, "Warning", f"Min bound must be strictly less than Max bound for {key}!")
                return
                
        self.accept()

    def get_optimization_data(self):
        selected_tracks = [name for name, cb in self.track_checkboxes.items() if cb.isChecked()]
        track_weights = {name: self.track_weights[name].value() for name in selected_tracks}
        bounds = {}
        for key, widgets in self.param_widgets.items():
            if widgets["enable"].isChecked():
                bounds[key] = (widgets["min"].value(), widgets["max"].value())
                
        return {
            "vehicle_name": self.combo_vehicle.currentText(),
            "tracks": selected_tracks,
            "track_weights": track_weights,
            "bounds": bounds,
            "trials": int(self.spin_trials.value())
        }
    
    def load_previous_run(self):
        dialog = LoadOptimizationDialog(self)
        if dialog.exec():
          
            json_data = dialog.selected_data
            
            
            res_dialog = OptimizationResultsDialog(payload=None, json_data=json_data, parent=self)
            res_dialog.exec()