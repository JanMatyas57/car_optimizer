import os

class StyleManager:
    @staticmethod
    def get_global_stylesheet():
        # 1. generate physical icons
        os.makedirs("data/icons", exist_ok=True)
        
        down_svg = "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#334155' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>"
        up_svg = "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#334155' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='18 15 12 9 6 15'/></svg>"
        
        with open("data/icons/down_arrow.svg", "w", encoding="utf-8") as f:
            f.write(down_svg)
        with open("data/icons/up_arrow.svg", "w", encoding="utf-8") as f:
            f.write(up_svg)

        # 2. central stylesheet QSS
        return """
            /* --- BASE WINDOWS AND TEXTS --- */
            QMainWindow, QDialog {
                background-color: #f8f9fa; 
                font-family: 'Segoe UI', system-ui, sans-serif;
                color: #334155;
            }
            QLabel {
                color: #1e293b;
                font-size: 10pt;
                font-weight: 500;
            }
            
            /* tabs */
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                background-color: #ffffff;
                border-radius: 8px;
                margin-top: -1px; 
            }
            QTabBar::tab {
                background-color: #e2e8f0;
                color: #64748b;
                border: 1px solid #cbd5e1;
                border-bottom: none; 
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #0f172a;
                border-top: 3px solid #3b82f6;
            }
            QTabBar::tab:hover:!selected {
                background-color: #cbd5e1;
                color: #0f172a;
            }

            /* boxes */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 8px;
                margin-top: 15px; 
                padding-top: 10px;
                font-weight: 600;
                font-size: 11pt;
                color: #334155;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #64748b;
            }
            
    
            QGroupBox#HighlightedGroup {
                background-color: #f1f5f9; 
                border: 1px solid #cbd5e1;
            }
            QGroupBox#HighlightedGroup::title {
                color: #334155;
            }

            QWidget#CanvasContainer {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }

      
            QComboBox, QDoubleSpinBox, QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 5px 10px;
                min-height: 22px;
                font-size: 10pt;
                color: #0f172a;
            }
            QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
                border: 1px solid #94a3b8;
            }
            QLineEdit:focus {
                border: 1px solid #3b82f6;
            }
            
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #0f172a;
                selection-background-color: #e2e8f0;
                selection-color: #0f172a;
                border: 1px solid #cbd5e1;
                outline: none;
            }
            QComboBox::drop-down {
                border-left: 1px solid #cbd5e1;
                width: 24px;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                background: #f1f5f9;
            }
            QComboBox::down-arrow {
                image: url("data/icons/down_arrow.svg");
                width: 14px;
                height: 14px;
            }
            
            QSpinBox,QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #f1f5f9; 
                border-left: 1px solid #cbd5e1; 
                width: 24px; 
            }
            QSpinBox,QDoubleSpinBox::up-button { border-top-right-radius: 4px; }
            QSpinBox,QDoubleSpinBox::down-button {
                border-bottom-right-radius: 4px;
                border-top: 1px solid #cbd5e1;
            }
            QSpinBox,QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: #e2e8f0; }
            QSpinBox,QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background-color: #cbd5e1; }
            QSpinBox,QDoubleSpinBox::up-arrow { image: url("data/icons/up_arrow.svg"); width: 12px; height: 12px; }
            QSpinBox,QDoubleSpinBox::down-arrow { image: url("data/icons/down_arrow.svg"); width: 12px; height: 12px; }

            /* --- CHECKBOXY --- */
            QCheckBox { font-size: 10pt; color: #334155; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #cbd5e1; background-color: #f8fafc; }
            QCheckBox::indicator:hover { border-color: #94a3b8; }
            QCheckBox::indicator:checked { background-color: #3b82f6; border-color: #3b82f6; }

            QPushButton {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                font-weight: 600;
                border-radius: 6px;
                padding: 10px 15px;
                border: none;
                color: white;
            }

            /* Default */
            QPushButton {
                background-color: #64748B;
            }
            QPushButton:hover {
                background-color: #475569;
            }

            QPushButton:disabled {
                background-color: #CBD5E1;
                color: #94A3B8;
            }

            /* Config (neutral) */
            QPushButton#BtnConfig {
                background-color: #475569;
            }
            QPushButton#BtnConfig:hover {
                background-color: #334155;
            }

            /* Track creation (creative action) */
            QPushButton#BtnCreateTrack {
                background-color: #6366F1;
            }
            QPushButton#BtnCreateTrack:hover {
                background-color: #4F46E5;
            }

            /* Compare (analysis) */
            QPushButton#BtnCompare {
                background-color: #3B82F6;
            }
            QPushButton#BtnCompare:hover {
                background-color: #2563EB;
            }

            /* Run simulation */
            QPushButton#BtnRunSingle {
                background-color: #10B981;
            }
            QPushButton#BtnRunSingle:hover {
                background-color: #059669;
            }

            /* Endurance run */
            QPushButton#BtnEndurance {
                background-color: #0EA5A4;
            }
            QPushButton#BtnEndurance:hover {
                background-color: #0F766E;
            }

            /* Map utility */
            QPushButton#BtnLapMap {
                background-color: #64748B;
            }
            QPushButton#BtnLapMap:hover {
                background-color: #475569;
            }

            /* Heavy computation */
            QPushButton#BtnOptimize {
                background-color: #3B82F6;
            }
            QPushButton#BtnOptimize:hover {
                background-color: #2563EB;
            }

            /* Sensitivity analysis */
            QPushButton#BtnSensitivity {
                background-color: #4F46E5;
            }
            QPushButton#BtnSensitivity:hover {
                background-color: #4338CA;
            }

            /* Tool buttons */
            QPushButton#BtnTool {
                background-color: #F1F5F9;
                color: #334155;
                font-weight: 600;
                font-size: 12pt;
                border-radius: 6px;
                border: 1px solid #CBD5E1;
                padding: 6px;
            }

            QPushButton#BtnTool:hover {
                background-color: #E2E8F0;
            }

         
            

            /* --- DIALOGY A SCROLLBARY --- */
            QMessageBox, QInputDialog { background-color: #f1f5f9; }
            QMessageBox QLabel, QInputDialog QLabel { color: #1e293b; background-color: transparent; }
            QDialogButtonBox QPushButton { background-color: #3b82f6; min-width: 80px; padding: 6px 16px; }
            QDialogButtonBox QPushButton:hover { background-color: #2563eb; }

            QScrollArea#LeftScrollArea { background-color: transparent; border: none; }
            QWidget#ControlPanel { background-color: transparent; }
            
            QScrollBar:vertical { border: none; background-color: #f1f5f9; width: 8px; border-radius: 4px; margin: 0px; }
            QScrollBar::handle:vertical { background-color: #cbd5e1; min-height: 30px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background-color: #94a3b8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { border: none; background: none; }
            
            QScrollBar:horizontal { border: none; background-color: #f1f5f9; height: 8px; border-radius: 4px; margin: 0px; }
            QScrollBar::handle:horizontal { background-color: #cbd5e1; min-width: 30px; border-radius: 4px; }
            QScrollBar::handle:horizontal:hover { background-color: #94a3b8; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { border: none; background: none; }
            QSplitter::handle { background-color: transparent; }

            /* Vynutí, aby samotný obsah záložky byl vždy čistě bílý */
            QTabWidget > QWidget {
                background-color: #ffffff;
            }

            /* --- MODERN SLIDERS --- */
            /* Horizontal slider */
            QSlider::groove:horizontal {
                border: 1px solid #cbd5e1;
                height: 6px;
                background: #e2e8f0;
                border-radius: 3px;
                margin: 2px 0;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6; /* blue fill from left to handle */
                border-radius: 3px;
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #3b82f6; /* white circle with blue border */
                width: 16px;
                height: 16px;
                margin: -6px 0; /* center on groove */
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #eff6ff;
                border: 2px solid #2563eb;
            }

            /* Vertical slider */
            QSlider::groove:vertical {
                border: 1px solid #cbd5e1;
                width: 6px;
                background: #e2e8f0;
                border-radius: 3px;
                margin: 0 2px;
            }
            QSlider::add-page:vertical {
                background: #3b82f6; /* blue fill from bottom to handle */
                border-radius: 3px;
                width: 6px;
            }
            QSlider::handle:vertical {
                background: #ffffff;
                border: 2px solid #3b82f6;
                height: 16px;
                width: 16px;
                margin: 0 -6px; /* center on groove */
                border-radius: 8px;
            }
            QSlider::handle:vertical:hover {
                background: #eff6ff;
                border: 2px solid #2563eb;
            }

            /* --- TABLES (QTableWidget) --- */
            QTableWidget {
                background-color: #ffffff;
                color: #0f172a; /* dark text color in cells */
                gridline-color: #e2e8f0; /* gridline color */
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                outline: none; /* remove dotted border on focus */
            }
            QTableWidget::item {
                padding: 4px 8px;
                color: #0f172a; /* cell text color protection */
                border-bottom: 1px solid #f1f5f9;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff; /* light blue selection */
                color: #1d4ed8; /* dark blue text for selected row */
                font-weight: 500;
            }
            
            /* --- TABLE HEADERS (QHeaderView) --- */
            QHeaderView {
                background-color: #ffffff;
                border: none;
            }
            QHeaderView::section {
                background-color: #f8fafc; /* light gray for headers */
                color: #475569; /* dark gray for header text */
                font-weight: bold;
                padding: 6px 10px;
                border: none;
                border-bottom: 2px solid #cbd5e1;
                border-right: 1px solid #e2e8f0;
            }
            QHeaderView::section:last {
                border-right: none;
            }
            
            /* Levý horní roh tabulky (pokud jsou zobrazená čísla řádků) */
            QTableCornerButton::section {
                background-color: #f8fafc;
                border: none;
                border-bottom: 2px solid #cbd5e1;
                border-right: 1px solid #e2e8f0;
            }
        """