import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QSpinBox, QDoubleSpinBox, QRadioButton,
                             QGroupBox, QMessageBox)

class TrackGeneratorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Track generation")
        self.resize(400, 350)
        
        self.selected_file = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- 1. telemetry data ---
        file_group = QGroupBox("1. Telemetry data (.mat)")
        file_layout = QVBoxLayout()
        self.btn_load_file = QPushButton("Select .mat file")
        self.btn_load_file.clicked.connect(self.load_file)
        
        self.lbl_file_path = QLabel("No file selected.")
        self.lbl_file_path.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_path.setWordWrap(True)
        
        file_layout.addWidget(self.btn_load_file)
        file_layout.addWidget(self.lbl_file_path)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- 2. track mode ---
        mode_group = QGroupBox("2. Track mode")
        mode_layout = QVBoxLayout()
        mode_group.setStyleSheet("color: black;")
        self.radio_closed = QRadioButton("Closed circuit (Endurance / Autocross)")
        self.radio_manual = QRadioButton("Manual section (Acceleration / Skidpad)")
        self.radio_closed.setChecked(True) 
        
        mode_layout.addWidget(self.radio_closed)
        mode_layout.addWidget(self.radio_manual)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # --- 3. smoothing parameters ---
        param_group = QGroupBox("3. Smoothing parameters")
        param_layout = QVBoxLayout()
        
        # r parameter
        layout_r = QHBoxLayout()
        lbl_r = QLabel("Smoothness factor (r):")
        self.spin_r = QDoubleSpinBox()
        self.spin_r.setRange(100, 10000)
        self.spin_r.setSingleStep(100)
        self.spin_r.setValue(3000)
        layout_r.addWidget(lbl_r)
        layout_r.addWidget(self.spin_r)
        
        # w parameter
        layout_w = QHBoxLayout()
        lbl_w = QLabel("Position weight (w):")
        self.spin_w = QDoubleSpinBox()
        self.spin_w.setRange(1.0, 500.0)
        self.spin_w.setSingleStep(10.0)
        self.spin_w.setValue(50.0)
        layout_w.addWidget(lbl_w)
        layout_w.addWidget(self.spin_w)

        param_layout.addLayout(layout_r)
        param_layout.addLayout(layout_w)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- 4. start button ---
        self.btn_generate = QPushButton("Continue to lap selection")
        self.btn_generate.setStyleSheet("font-weight: bold; background-color: #2E86C1; color: white; padding: 10px;")
        self.btn_generate.setEnabled(False) 
        self.btn_generate.clicked.connect(self.start_generation)
        
        layout.addSpacing(10)
        layout.addWidget(self.btn_generate)
        self.setLayout(layout)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select telemetry file", "", "MAT Files (*.mat);;All Files (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.lbl_file_path.setText(os.path.basename(file_path))
            self.lbl_file_path.setStyleSheet("color: black;")
            self.btn_generate.setEnabled(True)

    def start_generation(self):
        if not self.selected_file:
            return
        # close and accept dialog
        self.accept()