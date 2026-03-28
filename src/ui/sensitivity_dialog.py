import os
import glob
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QScrollArea, QWidget, QCheckBox, 
                             QLabel, QDoubleSpinBox, QPushButton, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt

class SensitivityDialog(QDialog):
    def __init__(self, vehicle_name, car_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensitivity test")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        self.vehicle_name = vehicle_name
        self.car_params = car_params
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # 1. Tracks to evaluate
        group_tracks = QGroupBox("1. Tracks to evaluate")
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
        self._load_tracks()
        
        self.layout_track_list.addStretch()
        scroll_tracks.setWidget(container_tracks)
        layout_tracks.addWidget(scroll_tracks)
        layout.addWidget(group_tracks)
        
        # 2. Parameters
        group_params = QGroupBox("2. Parameters to calculate sensitivities")
        group_params.setStyleSheet("QGroupBox { border: 1px solid #3b82f6; }")
        layout_params = QGridLayout(group_params)
        
        layout_params.addWidget(QLabel("Select"), 0, 0)
        layout_params.addWidget(QLabel("Parameter"), 0, 1)
        layout_params.addWidget(QLabel("Base Value"), 0, 2)
        
        self.param_widgets = {}
        
        params_setup = [
            ("weight", "Total weight [kg]", 280.0, 0.0, 1000.0, 1.0, 1),
            ("wb", "Wheelbase [m]", 1.53, 0.1, 5.0, 0.01, 3),
            ("tw", "Car track width [m]", 1.20, 0.1, 3.0, 0.01, 3),
            ("wd", "Weight dist. front [0-1]", 0.53, 0.0, 1.0, 0.01, 2),
            ("cogz", "CoG height [m]", 0.25, 0.0, 1.5, 0.01, 2),
            ("cd", "Drag coefficient (CdA)", 1.83, 0.0, 10.0, 0.01, 2),
            ("cd_drs", "CdA (DRS open)", 1.17, 0.0, 10.0, 0.01, 2),
            ("cl", "Lift coefficient (ClA)", 5.64, 0.0, 15.0, 0.05, 2),
            ("cl_drs", "ClA (DRS open)", 4.20, 0.0, 15.0, 0.05, 2),
            ("cop", "Center of pressure [0-1]", 0.53, 0.0, 1.0, 0.01, 2)
        ]
        
        for i, (key, label, default_val, p_min, p_max, step, decimals) in enumerate(params_setup, start=1):
            cb_enable = QCheckBox()
            cb_enable.setChecked(True)
            
            lbl_name = QLabel(label)
            
            val = self.car_params.get(key, default_val)
            spin_val = QDoubleSpinBox()
            spin_val.setRange(p_min, p_max)
            spin_val.setSingleStep(step)
            spin_val.setDecimals(decimals)
            spin_val.setValue(val)
            
            cb_enable.toggled.connect(spin_val.setEnabled)
            
            layout_params.addWidget(cb_enable, i, 0, alignment=Qt.AlignmentFlag.AlignCenter)
            layout_params.addWidget(lbl_name, i, 1)
            layout_params.addWidget(spin_val, i, 2)
            
            self.param_widgets[key] = {"enable": cb_enable, "value": spin_val}
            
        layout.addWidget(group_params)
        
        btn_layout = QHBoxLayout()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("BtnSensitivity")
        self.btn_run.clicked.connect(self.on_run_clicked)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)

    def _load_tracks(self):
        track_dir = "data/tracks"
        if os.path.exists(track_dir):
            files = glob.glob(os.path.join(track_dir, "*npz"))
            track_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
            for t_name in track_names:
                cb = QCheckBox(t_name)
                self.layout_track_list.addWidget(cb)
                self.track_checkboxes[t_name] = cb
        else:
            cb = QCheckBox("No tracks found")
            cb.setEnabled(False)
            self.layout_track_list.addWidget(cb)

    def on_run_clicked(self):
        selected_tracks = [n for n, cb in self.track_checkboxes.items() if cb.isChecked()]
        if not selected_tracks:
            QMessageBox.warning(self, "Warning", "Please select at least one track.")
            return

        parameters = {}
        for key, widgets in self.param_widgets.items():
            if widgets["enable"].isChecked():
                # For now step size is fixed to 1%
                parameters[key] = {"dx": 0.01}
                
        if not parameters:
            QMessageBox.warning(self, "Warning", "Please select at least one parameter.")
            return
            
        self.accept()
        # Return true and data so the caller can trigger the request
        self.result_data = {
            "track_names": selected_tracks,
            "parameters": parameters
        }

    def get_request_data(self):
        return getattr(self, "result_data", None)
