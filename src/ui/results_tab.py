import os
import glob
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                             QLabel, QHeaderView, QPushButton, QMessageBox, QHBoxLayout)
from PyQt6.QtCore import Qt
from src.ui.compare_results import ComparisonPlotWindow

class CompareWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Compare simulation logs")
        self.resize(1000, 700)
        
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                color: #334155;
                background-color: #f8f9fa;
            }
            QLabel#TitleLabel {
                font-size: 18px; 
                font-weight: bold; 
                color: #0f172a;
                margin-bottom: 2px;
            }
            QLabel#HintLabel {
                color: #64748b; 
                margin-bottom: 10px;
            }
            
            /* --- TABULKA --- */
            QTreeWidget {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                outline: none;
            }

            /* Jemný modrý border při focusu */
            QTreeWidget:focus {
                border: 1px solid #3b82f6;
            }

            QTreeWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f1f5f9;
                color: #334155;
            }
            
            /* Moderní selection bez systémové modré */
            QTreeWidget::item:selected {
                background-color: #eff6ff; 
                color: #2563eb;            
                outline: none;
            }
            
            /* Stav kdy okno není aktivní */
            QTreeWidget::item:selected:!active {
                background-color: #f8fafc;
                color: #64748b;
            }

            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: bold;
                color: #475569;
            }
            
            /* --- TLAČÍTKA --- */
            QPushButton {
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton#BtnRefresh {
                background-color: #64748b;
                color: white;
            }
            QPushButton#BtnRefresh:hover {
                background-color: #475569;
            }
            QPushButton#BtnCompare {
                background-color: #3b82f6;
                color: white;
            }
            QPushButton#BtnCompare:hover {
                background-color: #2563eb;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # 1. Header
        lbl_title = QLabel("Select simulations to compare")
        lbl_title.setObjectName("TitleLabel")
        layout.addWidget(lbl_title)
        
        lbl_hint = QLabel("Check the boxes on the left. All selected logs must be from the SAME track.")
        lbl_hint.setObjectName("HintLabel")
        layout.addWidget(lbl_hint)
        
        # 2. Table (TreeWidget)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Track Name", "Lap Time (s)", "Car Config", "File / Date"])
        self.tree.setIndentation(10)
        
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        # VYPNUTO: Už žádné střídavé barvy, které by si systém mohl obarvit po svém
        self.tree.setAlternatingRowColors(False)
        self.tree.setRootIsDecorated(False) 
        layout.addWidget(self.tree)
        
        # 3. Buttons Area
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 10, 0, 0)
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.setObjectName("BtnRefresh")
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.load_logs)
        
        self.btn_compare = QPushButton("Compare Selected")
        self.btn_compare.setObjectName("BtnCompare")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.clicked.connect(self.on_compare_clicked)
        
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_compare)
        
        layout.addLayout(btn_layout)

        self.load_logs()

    def load_logs(self):
        self.tree.clear()
        results_dir = "results"
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            return

        files = glob.glob(os.path.join(results_dir, "*.npz"))
        logs_data = []

        for filepath in files:
            try:
                with np.load(filepath, allow_pickle=True) as data:
                    track_name = str(data['track_name']) if 'track_name' in data else "Unknown"
                    lap_time = float(data['lap_time']) if 'lap_time' in data else 0.0
                    
                    # CAR CONFIG LOGIKA:
                    # Pokud v budoucnu přidáš do npz 'car_config' (např. string s aero nastavením), 
                    # zobrazí se přednostně, jinak jen jméno.
                    car_config = str(data['car_config']) if 'car_config' in data else str(data.get('car_name', '-'))
                    
                    filename = os.path.basename(filepath)
                    
                    logs_data.append({
                        'track': track_name,
                        'time': lap_time,
                        'car': car_config,
                        'file': filename,
                        'path': filepath
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

        logs_data.sort(key=lambda x: (x['track'], x['time']))

        for item in logs_data:
            row = QTreeWidgetItem([
                item['track'],
                f"{item['time']:.3f}",
                item['car'],
                item['file']
            ])
            
            row.setCheckState(0, Qt.CheckState.Unchecked)
            row.setData(0, Qt.ItemDataRole.UserRole, item['path'])
            
            # Vynutíme transparentní background, aby neprosvítala systémová barva
            for col in range(self.tree.columnCount()):
                row.setBackground(col, Qt.GlobalColor.transparent)
            
            self.tree.addTopLevelItem(row)

        self.tree.setCurrentItem(None)

    def on_compare_clicked(self):
        # ... (zbytek logiky zůstává stejný)
        selected_data = []
        track_names_set = set()
        
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                path = item.data(0, Qt.ItemDataRole.UserRole)
                track_name = item.text(0)
                lap_time = item.text(1)
                filename = item.text(3)
                
                selected_data.append({
                    'path': path,
                    'track': track_name,
                    'legend': f"{filename} ({lap_time}s)"
                })
                track_names_set.add(track_name)

        if len(selected_data) < 2:
            QMessageBox.warning(self, "Selection Required", "Please select at least 2 logs to compare.")
            return

        if len(track_names_set) > 1:
            QMessageBox.critical(self, "Track Mismatch", "Please select logs from the same track only.")
            return

        self.comparison_window = ComparisonPlotWindow(selected_data)
        self.comparison_window.show()