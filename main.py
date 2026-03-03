import os
import glob
import sys
from src.core.track import Track
from src.ui.main_window import MainWindow
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QGroupBox, QComboBox, 
                             QDoubleSpinBox, QPushButton, QSplitter)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import ctypes
from PyQt6.QtGui import QIcon

if __name__ == "__main__":
    try:
        myappid = 'optisim' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    except ImportError:
        pass # Not on windows

    app = QApplication(sys.argv)

    icon_path = os.path.join("data", "logo.png")
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

#TO ADD
#add elevation change 
#add warm start
#redesign to use one event bus
#add twintrack
#add optuna to find twintrack parameters and UI
