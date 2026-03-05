import os

class StyleManager:
    @staticmethod
    def get_global_stylesheet():
        # 1. Vygenerování fyzických ikon
        os.makedirs("data/icons", exist_ok=True)
        
        down_svg = "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#334155' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>"
        up_svg = "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#334155' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='18 15 12 9 6 15'/></svg>"
        
        with open("data/icons/down_arrow.svg", "w", encoding="utf-8") as f:
            f.write(down_svg)
        with open("data/icons/up_arrow.svg", "w", encoding="utf-8") as f:
            f.write(up_svg)

        # 2. Centrální QSS
        return """
            /* --- ZÁKLADNÍ OKNA A TEXTY --- */
            QMainWindow, QDialog {
                background-color: #f8f9fa; /* Změněno na čistší světlou z ResultsWindow */
                font-family: 'Segoe UI', system-ui, sans-serif;
                color: #334155;
            }
            QLabel {
                color: #1e293b;
                font-size: 10pt;
                font-weight: 500;
            }
            
            /* --- ZÁLOŽKY (TABS) --- */
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

            /* --- KONTEJNERY A BOXY --- */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
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
            
            /* Speciální šedý GroupBox z ResultsWindow */
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

            /* --- VSTUPY A COMBOBOXY --- */
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
            
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #f1f5f9; 
                border-left: 1px solid #cbd5e1; 
                width: 24px; 
            }
            QDoubleSpinBox::up-button { border-top-right-radius: 4px; }
            QDoubleSpinBox::down-button {
                border-bottom-right-radius: 4px;
                border-top: 1px solid #cbd5e1;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: #e2e8f0; }
            QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background-color: #cbd5e1; }
            QDoubleSpinBox::up-arrow { image: url("data/icons/up_arrow.svg"); width: 12px; height: 12px; }
            QDoubleSpinBox::down-arrow { image: url("data/icons/down_arrow.svg"); width: 12px; height: 12px; }

            /* --- CHECKBOXY --- */
            QCheckBox { font-size: 10pt; color: #334155; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #cbd5e1; background-color: #f8fafc; }
            QCheckBox::indicator:hover { border-color: #94a3b8; }
            QCheckBox::indicator:checked { background-color: #3b82f6; border-color: #3b82f6; }

            /* --- OBECNÁ TLAČÍTKA --- */
            QPushButton {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
                font-weight: 600;
                border-radius: 6px;
                padding: 10px 15px;
                border: none;
                color: white;
                background-color: #64748b; 
            }
            QPushButton:hover { background-color: #475569; }
            QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; }

            /* Specifická tlačítka podle ID */
            QPushButton#BtnConfig { background-color: #475569; }
            QPushButton#BtnConfig:hover { background-color: #334155; }
            QPushButton#BtnCreateTrack { background-color: #8b5cf6; }
            QPushButton#BtnCreateTrack:hover { background-color: #7c3aed; }
            QPushButton#BtnCompare { background-color: #3b82f6; }
            QPushButton#BtnCompare:hover { background-color: #2563eb; }
            QPushButton#BtnRunSingle { background-color: #10b981; }
            QPushButton#BtnRunSingle:hover { background-color: #059669; }
            QPushButton#BtnEndurance { background-color: #06b6d4; }
            QPushButton#BtnEndurance:hover { background-color: #0891b2; }
            QPushButton#BtnLapMap { background-color: #64748b; }
            QPushButton#BtnLapMap:hover { background-color: #475569; }
            
            QPushButton#BtnTool {
                background-color: #e2e8f0;
                color: #334155;
                font-weight: bold;
                font-size: 14pt;
                border-radius: 5px;
                border: 1px solid #cbd5e1;
                padding: 5px;
            }
            QPushButton#BtnTool:hover { background-color: #cbd5e1; }

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
            QSplitter::handle { background-color: transparent; }

            /* Vynutí, aby samotný obsah záložky byl vždy čistě bílý */
            QTabWidget > QWidget {
                background-color: #ffffff;
            }

            /* --- MODERNÍ SLIDERY (POSUVNÍKY) --- */
            /* Horizontální slider */
            QSlider::groove:horizontal {
                border: 1px solid #cbd5e1;
                height: 6px;
                background: #e2e8f0;
                border-radius: 3px;
                margin: 2px 0;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6; /* Modrá výplň zleva po držátko */
                border-radius: 3px;
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #3b82f6; /* Bílé kolečko s modrým okrajem */
                width: 16px;
                height: 16px;
                margin: -6px 0; /* Vycentrování na střed drážky */
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #eff6ff;
                border: 2px solid #2563eb;
            }

            /* Vertikální slider */
            QSlider::groove:vertical {
                border: 1px solid #cbd5e1;
                width: 6px;
                background: #e2e8f0;
                border-radius: 3px;
                margin: 0 2px;
            }
            QSlider::add-page:vertical {
                background: #3b82f6; /* Modrá výplň zdola po držátko */
                border-radius: 3px;
                width: 6px;
            }
            QSlider::handle:vertical {
                background: #ffffff;
                border: 2px solid #3b82f6;
                height: 16px;
                width: 16px;
                margin: 0 -6px; /* Vycentrování na střed drážky */
                border-radius: 8px;
            }
            QSlider::handle:vertical:hover {
                background: #eff6ff;
                border: 2px solid #2563eb;
            }
        """