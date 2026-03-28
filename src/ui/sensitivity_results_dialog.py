import os
import json
import math
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QBrush

class SensitivityResultsDialog(QDialog):
    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self.payload = payload
        
        self.setWindowTitle("Sensitivity Analysis Results")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
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
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                background-color: white;
                alternate-background-color: white;
                gridline-color: #cbd5e1;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                color: #475569;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #e2e8f0;
            }
        """)

        self.results_data = self.payload.data
        self.meta = self.payload.meta
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        tracks = list(self.results_data.keys())
        
        # Summary
        summary = QGroupBox(f"Sensitivity Analysis Summary")
        s_layout = QVBoxLayout(summary)
        
        lbl_tracks = QLabel(f"<b>Tracks evaluated:</b> {', '.join(tracks)}")
        lbl_tracks.setStyleSheet("font-size: 11pt; color: #334155;")
        s_layout.addWidget(lbl_tracks)
        
        layout.addWidget(summary)

        # Table
        table_group = QGroupBox("Results per Track")
        t_layout = QVBoxLayout(table_group)
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(False)
        t_layout.addWidget(self.table)
        
        self._populate_table()
        
        layout.addWidget(table_group)

        # Button row
        btn_layout = QHBoxLayout()
        
        btn_save = QPushButton("Save JSON")
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

    def _populate_table(self):
        tracks = list(self.results_data.keys())
        if not tracks:
            return
            
        params = list(self.results_data[tracks[0]].keys())
        
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Track", "Parameter", "Base Value", "dT per +1% change [s]", "Sensitivity / Gradient [s/unit]"
        ])
        
        self.table.setRowCount(len(tracks) * len(params))
        
        row = 0
        for track_name in tracks:
            for param_name in params:
                data = self.results_data[track_name][param_name]
                
                base_val = data["base_value"]
                sens = data["sensitivity"]
                time_diff = data["time_diff_per_dx"] # Approximation of lap time delta per step size
                
                # Format
                t_item = QTableWidgetItem(track_name)
                p_item = QTableWidgetItem(param_name)
                b_item = QTableWidgetItem(f"{base_val:.4f}")
                
                diff_str = f"{time_diff:+.4f}" if not np.isnan(time_diff) else "NaN"
                td_item = QTableWidgetItem(diff_str)
                
                sens_str = f"{sens:+.4f}" if not np.isnan(sens) else "NaN"
                s_item = QTableWidgetItem(sens_str)
                
                # Align numbers right
                b_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                td_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                s_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
                # Colorize positive vs negative delta laptime
                if not np.isnan(time_diff):
                    if time_diff < -0.001:
                        td_item.setForeground(QBrush(QColor("#10b981"))) # Green (improvement)
                    elif time_diff > 0.001:
                        td_item.setForeground(QBrush(QColor("#ef4444"))) # Red (worse)
                        
                if not np.isnan(sens):
                    if sens < -0.001:
                        s_item.setForeground(QBrush(QColor("#10b981"))) # Green (negative gradient)
                    elif sens > 0.001:
                        s_item.setForeground(QBrush(QColor("#ef4444"))) # Red (positive gradient)

                self.table.setItem(row, 0, t_item)
                self.table.setItem(row, 1, p_item)
                self.table.setItem(row, 2, b_item)
                self.table.setItem(row, 3, td_item)
                self.table.setItem(row, 4, s_item)
                
                row += 1
                
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

    def save_results(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"sensitivity_analysis_{timestamp}.json"
        
        save_dir = os.path.join("data", "sensitivities")
        os.makedirs(save_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save sensitivity results", os.path.join(save_dir, default_filename), "JSON Files (*.json)")
        
        if not file_path: return
            
        # Serialize format
        export_data = {
            "metadata": {
                "timestamp": timestamp,
                "type": "Sensitivity Analysis",
                "tracks": [t.name for t in self.meta["tracks"]]
            },
            "parameters": self.meta["parameters"],
            "results": self.results_data
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f: 
                # sanitize nan to null for JSON standard
                json_str = json.dumps(export_data, indent=4).replace("NaN", "null")
                f.write(json_str)
            QMessageBox.information(self, "Success", f"Results saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results:\n{str(e)}")
