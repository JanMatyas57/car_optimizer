from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPalette, QColor, QMovie

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False) 
        self.hide()
        
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
     
        # Poloprůhledné černé pozadí
        self.setObjectName("LoadingOverlay")
        self.setStyleSheet("""
            QWidget#LoadingOverlay {
                background-color: rgba(0, 0, 0, 200);
            }
            QLabel {
                background: transparent;
            }
        """)
        
        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20) # Mezera mezi GIFem a textem
        
        # --- 1. GIF LABEL ---
        self.lbl_gif = QLabel()
        self.lbl_gif.setAlignment(Qt.AlignmentFlag.AlignCenter)
     
        gif_path = "data/vejsi_loading.gif" 
        self.movie = QMovie(gif_path)
     
        self.movie.setScaledSize(QSize(300, 300)) 
        self.movie.finished.connect(self.movie.start)
        self.lbl_gif.setMovie(self.movie)
        
        # --- 2. TEXT LABEL ---
        self.lbl_text = QLabel("Calculating...")
        self.lbl_text.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- 3. PROGRESS BAR 
        self.progress = QProgressBar()
        self.progress.setFixedWidth(300)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #333;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                width: 20px; 
            }
        """)
        
        # Přidání do layoutu
        layout.addWidget(self.lbl_gif)   # GIF nahoře
        layout.addWidget(self.lbl_text)  # Text uprostřed
        layout.addWidget(self.progress)  # Bar dole
        
        self.setLayout(layout)
        
        self.current_progress_fmt = None

    def show_loading(self, text="Calculating...", indefinite=True, progress_text_fmt=None):
        """
        Zobrazí overlay.
        Args:
            text: Úvodní text.
            indefinite: 
                True = Single Lap / ACP -> Ukáže GIF, skryje Bar.
                False = Endurance -> Ukáže Bar, skryje GIF.
        """
        if self.parent():
            self.resize(self.parent().size())
            
        self.lbl_text.setText(text)
        
        # Uložení formátu textu
        if progress_text_fmt:
            self.current_progress_fmt = progress_text_fmt
        else:
            self.current_progress_fmt = f"{text} ({{}} / {{}})"

        if indefinite:
            # --- REŽIM GIF (Točící se kolečko / Animace) ---
            self.progress.hide()       # Schováme progress bar
            self.lbl_gif.show()        # Ukážeme GIF
            self.movie.start()         # Spustíme animaci
        else:
            # --- REŽIM PROGRESS BAR (Endurance) ---
            self.lbl_gif.hide()        # Schováme GIF (nebo ho necháme, pokud chceš obojí)
            self.movie.stop()          # Zastavíme animaci (šetří CPU)
            
            self.progress.show()       # Ukážeme progress bar
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

        self.show()
        self.raise_()

    def update_progress(self, current, total):
        # ... (Tato metoda zůstává stejná jako předtím) ...
        if self.current_progress_fmt:
            try:
                self.lbl_text.setText(self.current_progress_fmt.format(current, total))
            except:
                self.lbl_text.setText(f"{current} / {total}")
        
        if self.progress.minimum() == 0 and self.progress.maximum() == 0:
            self.progress.setRange(0, total)
            self.progress.setTextVisible(True)
            self.progress.setFormat("%v / %m") 
            
        self.progress.setValue(current)

    def hide_loading(self):
        self.movie.stop() # Důležité: zastavit GIF při skrytí
        self.hide()