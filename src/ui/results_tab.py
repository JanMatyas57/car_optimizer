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
        self.resize(900, 600)
        
        layout = QVBoxLayout(self)
        
        # 1. Header
        lbl_title = QLabel("Select simulations to compare")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(lbl_title)
        
        lbl_hint = QLabel("Check the boxes on the left. All selected logs must be from the SAME track.")
        lbl_hint.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(lbl_hint)
        
        # 2. Table (TreeWidget)
        self.tree = QTreeWidget()
        # První sloupec bude mít checkbox
        self.tree.setHeaderLabels(["Track Name", "Lap Time (s)", "Car Config", "File / Date"])
        
        # Nastavení šířky sloupců
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Track
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Time
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Car
        
        self.tree.setAlternatingRowColors(True)
        # Root decoration schováme (nemáme zanořování, je to plochý seznam)
        self.tree.setRootIsDecorated(False) 
        
        layout.addWidget(self.tree)
        
        # 3. Buttons Area
        btn_layout = QHBoxLayout()
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.load_logs)
        
        self.btn_compare = QPushButton("Compare Selected")
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.setStyleSheet("""
            QPushButton {
                background-color: #007bff; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.btn_compare.clicked.connect(self.on_compare_clicked)
        
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch() # Mezera
        btn_layout.addWidget(self.btn_compare)
        
        layout.addLayout(btn_layout)

        # Načtení dat při startu
        self.load_logs()

    def load_logs(self):
        """Proskenuje složku results a naplní tabulku s checkboxy."""
        self.tree.clear()
        results_dir = "results"
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            return

        files = glob.glob(os.path.join(results_dir, "*.npz"))
        logs_data = []

        # A) Načtení metadat
        for filepath in files:
            try:
                data = np.load(filepath, allow_pickle=True)
                
                track_name = str(data['track_name']) if 'track_name' in data else "Unknown"
                lap_time = float(data['lap_time']) if 'lap_time' in data else 0.0
                car_name = str(data['car_name']) if 'car_name' in data else "-"
                
                # Zformátujeme datum ze jména souboru (pokud to jde), jinak jen filename
                filename = os.path.basename(filepath)
                
                logs_data.append({
                    'track': track_name,
                    'time': lap_time,
                    'car': car_name,
                    'file': filename,
                    'path': filepath
                })
                data.close()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

        # B) Třídění (Track -> Time)
        logs_data.sort(key=lambda x: (x['track'], x['time']))

        # C) Vykreslení
        for item in logs_data:
            row = QTreeWidgetItem([
                item['track'],
                f"{item['time']:.3f}",
                item['car'],
                item['file']
            ])
            
            # --- PŘIDÁNÍ CHECKBOXU ---
            # Nastavíme checkbox do sloupce 0 (Track Name)
            row.setCheckState(0, Qt.CheckState.Unchecked)
            
            # Uložíme si cestu k souboru do skrytých dat
            row.setData(0, Qt.ItemDataRole.UserRole, item['path'])
            
            self.tree.addTopLevelItem(row)

    def on_compare_clicked(self):
        """Zkontroluje výběr a spustí porovnání."""
        selected_data = []
        track_names_set = set()
        
        # 1. Iterace přes všechny řádky
        root = self.tree.invisibleRootItem()
        count = root.childCount()
        
        for i in range(count):
            item = root.child(i)
            
            # Je checkbox zaškrtnutý?
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

        # 2. VALIDACE: Počet logů
        if len(selected_data) < 2:
            QMessageBox.warning(self, "Not enough data", 
                                "Please select at least 2 logs to compare.")
            return

        # 3. VALIDACE: Shoda tratí
        if len(track_names_set) > 1:
            tracks_list = ", ".join(track_names_set)
            QMessageBox.critical(self, "Track Mismatch", 
                                 f"You cannot compare logs from different tracks!\n\n"
                                 f"Selected tracks: {tracks_list}\n"
                                 f"Please select logs from the same track only.")
            return

        # 4. Success -> Spustit vizualizaci
        # (Zatím jen print, příště uděláme okno)
        print(f"Comparing {len(selected_data)} logs form track '{list(track_names_set)[0]}'...")
        print(f"Comparing {len(selected_data)} logs...")
        
        # Uložíme do self, aby okno nezmizelo
        self.comparison_window = ComparisonPlotWindow(selected_data)
        self.comparison_window.show()