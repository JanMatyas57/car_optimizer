import os
import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                             QPushButton, QHBoxLayout, QHeaderView, QMessageBox)
from PyQt6.QtCore import Qt
class LoadOptimizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load optimization run")
        self.resize(700, 400)
        self.selected_data = None
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Date", "Base Vehicle", "Tracks", "Best objective"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load selected")
        btn_load.setObjectName("BtnOptimize")
        btn_load.clicked.connect(self.load_selected)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        self.files_map = {} 
        self.populate_table()
        
    def populate_table(self):
        save_dir = os.path.join("data", "optimizations")
        if not os.path.exists(save_dir):
            return
            
        row = 0
    
        for filename in sorted(os.listdir(save_dir), reverse=True):
            if filename.endswith(".json"):
                filepath = os.path.join(save_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        meta = data.get("metadata", {})
                        
                        date_str = meta.get("timestamp", filename)
                        vehicle = meta.get("base_vehicle_name", "N/A")
                        tracks = ", ".join(meta.get("tracks_used", []))
                        # Handle both old format (best_lap_time) and new normalized format (best_objective)
                        raw = meta.get('best_objective', meta.get('best_lap_time'))
                        if raw is not None and raw < 10:  # new normalized format (ratio ~1.0)
                            improvement_pct = (1.0 - raw) * 100.0
                            display_str = f"{raw:.5f}  ({improvement_pct:+.2f}%)"
                        elif raw is not None:  # old format (absolute seconds)
                            display_str = f"{raw:.3f} s (legacy)"
                        else:
                            display_str = "N/A"
                        
                        self.table.insertRow(row)
                        self.table.setItem(row, 0, QTableWidgetItem(date_str))
                        self.table.setItem(row, 1, QTableWidgetItem(vehicle))
                        self.table.setItem(row, 2, QTableWidgetItem(tracks))
                  
                        obj_item = QTableWidgetItem(display_str)
                        obj_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        self.table.setItem(row, 3, obj_item)
                        
                        self.files_map[row] = filepath
                        row += 1
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    
    def load_selected(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a run to load.")
            return
            
        row = selected[0].row()
        filepath = self.files_map.get(row)
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.selected_data = json.load(f)
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")