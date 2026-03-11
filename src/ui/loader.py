from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPalette, QColor, QMovie

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False) 
        self.hide()
        
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
     
       
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
        
        # gif label
        self.lbl_gif = QLabel()
        self.lbl_gif.setAlignment(Qt.AlignmentFlag.AlignCenter)
     
        gif_path = "data/loading1.gif" 
        self.movie = QMovie(gif_path)
     
        self.lbl_gif.setFixedSize(300, 300)
        self.lbl_gif.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.movie.frameChanged.connect(self.update_gif_frame)
        self.movie.finished.connect(self.movie.start)
        
        # text label
        self.lbl_text = QLabel("Calculating...")
        self.lbl_text.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # progress bar
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
        
       
        layout.addWidget(self.lbl_gif)   # GIF nahoře
        layout.addWidget(self.lbl_text)  # Text uprostřed
        layout.addWidget(self.progress)  # Bar dole
        
        self.setLayout(layout)
        
        self.current_progress_fmt = None

    def show_loading(self, text="Calculating...", indefinite=True, progress_text_fmt=None):
        """
        shows overlay
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
            # gif 
            self.progress.hide()      
            self.lbl_gif.show()        
            self.movie.start()        
        else:
            # progress bar
            self.lbl_gif.hide()       
            self.movie.stop()          
            
            self.progress.show()       
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

        self.show()
        self.raise_()

    def update_gif_frame(self, frame_number):
        pixmap = self.movie.currentPixmap()
        if pixmap.isNull():
            return
            
        ratio = self.devicePixelRatioF()
        physical_size = int(300 * ratio)
        
        # Smoothly scale the frame to the target physical size
        scaled_pixmap = pixmap.scaled(
            physical_size, physical_size, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Tell Qt this is a High-DPI pixmap before displaying it
        scaled_pixmap.setDevicePixelRatio(ratio)
        self.lbl_gif.setPixmap(scaled_pixmap)

    def update_progress(self, current, total):
       
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
        self.movie.stop() 
        self.hide()