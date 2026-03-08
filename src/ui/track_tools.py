import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from scipy.signal import find_peaks

# ==========================================
# math helpers
# ==========================================
def get_tangent_vector(data, latitude_ref, longitude_ref):
    lon, lat = data["Longitude"], data["Latitude"]
    distances = np.sqrt((lon - longitude_ref)**2 + (lat - latitude_ref)**2)
    closest_idx = np.argmin(distances)
    vector = np.array([0.0, 0.0])
    for i in range(1, 51):
        if closest_idx + i < len(lon):
            vector = np.array([lon[closest_idx + i] - lon[closest_idx],
                               lat[closest_idx + i] - lat[closest_idx]])
            if not np.allclose(vector, [0, 0]): break
    return closest_idx, vector

def get_lap_indices(data, point, tangent_vector):
    lat, lon = data["Latitude"], data["Longitude"]
    rel_long, rel_lat = lon - point[1], lat - point[0]
    vec_norm = np.dot(tangent_vector, tangent_vector)
    projected = np.array([np.dot([rel_long[i], rel_lat[i]], tangent_vector) / vec_norm for i in range(len(lon))])
    minima_indices, _ = find_peaks(-np.abs(projected))
    return [i for i in minima_indices if (abs(rel_long[i]) + abs(rel_lat[i])) < 0.0001]

# ==========================================
# 1. start / finish selection dialog
# ==========================================
class StartFinishDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.selected_idx = None
        self.selected_vec = None
        
        self.setWindowTitle("1. Select Start/Finish Line")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        lbl = QLabel("Click on the track to set the Start/Finish line. Press <b>ENTER</b> to confirm.")
        lbl.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(lbl)
        
        self.plot = pg.PlotWidget()
        mean_lat = np.mean(self.data["Latitude"])
        aspect_ratio = 1.0 / np.cos(np.radians(mean_lat))
        self.plot.setAspectLocked(True, ratio=aspect_ratio)
        self.plot.setBackground('w')
        self.plot.plot(self.data["Longitude"], self.data["Latitude"], pen=None, symbol='o', symbolSize=2, symbolPen=None, symbolBrush=(150, 150, 150, 100))
        
        self.marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen('k'), brush=pg.mkBrush('r'))
        self.plot.addItem(self.marker)
        layout.addWidget(self.plot)
        
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_click)

    def on_mouse_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
            idx, vec = get_tangent_vector(self.data, pos.y(), pos.x())
            self.selected_idx = idx
            self.selected_vec = vec
            
            self.marker.setData([self.data["Longitude"][idx]], [self.data["Latitude"][idx]])

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            if self.selected_idx is not None:
                self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()

# ==========================================
# 2. lap browser dialog
# ==========================================
# ==========================================
# 2. lap browser dialog (single lap only)
# ==========================================
class LapSelectorDialog(QDialog):
    def __init__(self, data, laps_data, parent=None):
        super().__init__(parent)
        self.data = data
        self.laps_data = laps_data
        self.current_idx = 0
        self.selected_lap = None
        
        self.setWindowTitle("2. Browse and select a single lap")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        self.lbl = QLabel()
        self.lbl.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.lbl)
        
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)

        self.plot.setBackground('w')
        self.plot.plot(self.data["Longitude"], self.data["Latitude"], pen=None, symbol='o', symbolSize=2, symbolPen=None, symbolBrush=(150, 150, 150, 100))
        
        self.highlight_curve = self.plot.plot([], [], pen=pg.mkPen('b', width=4))
        layout.addWidget(self.plot)
        self.update_ui()

    def update_ui(self):
        lap = self.laps_data[self.current_idx]
        s, e = lap['indices']
        self.highlight_curve.setData(self.data["Longitude"][s:e], self.data["Latitude"][s:e])
        
        text = (f"Lap {self.current_idx + 1} / {len(self.laps_data)} | Time: {lap['time']:.3f}s<br>"
                f"<b>ARROWS</b>: Browse | <b>ENTER</b>: Select this lap | <b>ESC</b>: Cancel")
        self.lbl.setText(text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            self.current_idx = (self.current_idx + 1) % len(self.laps_data)
            self.update_ui()
        elif event.key() == Qt.Key.Key_Left:
            self.current_idx = (self.current_idx - 1) % len(self.laps_data)
            self.update_ui()
        elif event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.selected_lap = [self.laps_data[self.current_idx]] # return as list of 1 element
            self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()

def run_lap_selector(filtered_data, parent=None):
    dlg1 = StartFinishDialog(filtered_data, parent)
    if dlg1.exec() != QDialog.DialogCode.Accepted:
        return None
        
    pt = [filtered_data["Latitude"][dlg1.selected_idx], filtered_data["Longitude"][dlg1.selected_idx]]
    lap_indices = get_lap_indices(filtered_data, pt, dlg1.selected_vec)
    
    if len(lap_indices) < 2:
        return None

    laps_data = [{'indices': (lap_indices[i], lap_indices[i+1]), 
                  'time': filtered_data["Time"][lap_indices[i+1]] - filtered_data["Time"][lap_indices[i]]} 
                 for i in range(len(lap_indices) - 1)]

    dlg2 = LapSelectorDialog(filtered_data, laps_data, parent)
    if dlg2.exec() != QDialog.DialogCode.Accepted:
        return None
        
    return dlg2.selected_lap

# ==========================================
# 3. manual section selector
# ==========================================
class ManualLapDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.selected_indices = None
        
        self.setWindowTitle("Select track section")
        self.resize(1000, 800)
        
        layout = QVBoxLayout(self)
        lbl = QLabel(
            "Drag the yellow region to select the desired section. "
            "<b><span style='color: red;'>Avoid areas with red speed </span></b>.<br>"
            "Press <b>ENTER</b> or click <b>Done</b> to confirm."
        )
        lbl.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(lbl)
        
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')
        layout.addWidget(self.win)
        
        t = self.data["Time"]
        v = self.data["Speed"]
        
        # speed plot
        p1 = self.win.addPlot(row=0, col=0, title="Speed")
        p1.showGrid(x=True, y=True)
        
        # 1. draw the base blue line for the entire valid speed
        p1.plot(t, v, pen='b')
        
        # 2. highlight invalid speeds (< 3 m/s) in red

        v_invalid = np.copy(v)
        v_invalid[v >= 3.0] = np.nan 
        p1.plot(t, v_invalid, pen=pg.mkPen('r', width=3))
        
        # region selector on the first plot
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.setRegion([t[int(len(t)*0.1)], t[int(len(t)*0.9)]]) # default selection
        p1.addItem(self.region)
        
        # power plot
        p2 = self.win.addPlot(row=1, col=0, title="Power")
        p2.plot(t, self.data["Power"], pen=pg.mkPen('orange'))
        p2.setXLink(p1)
        p2.showGrid(x=True, y=True)
        
        # brake plot
        p3 = self.win.addPlot(row=2, col=0, title="Brake")
        p3.plot(t, self.data["Brake"], pen='r')
        p3.setXLink(p1)
        p3.showGrid(x=True, y=True)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_done = QPushButton("Done")
        btn_done.setFixedWidth(150)
        btn_done.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 10px;")
        btn_done.clicked.connect(self.accept)
        btn_layout.addWidget(btn_done)
        layout.addLayout(btn_layout)

    def get_selection(self):
        min_t, max_t = self.region.getRegion()
        t = self.data["Time"]
        s_idx = np.searchsorted(t, min_t)
        e_idx = np.searchsorted(t, max_t)
        return [{"indices": (s_idx, e_idx), "time": t[e_idx] - t[s_idx]}]

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()

def manual_lap_selector(data, parent=None):
    dlg = ManualLapDialog(data, parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return dlg.get_selection()
    return None

# ==========================================
# 4. drs zone editor
# ==========================================
class DRSEditorDialog(QDialog):
    def __init__(self, x, y, s, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.s = s
        self.drs_mask = np.zeros_like(self.s)
        self.temp_start_idx = None
        self.zones = []
        
        self.setWindowTitle("Define DRS Zones")
        self.resize(900, 700)
        
        layout = QVBoxLayout(self)
        
        lbl = QLabel(
            "<b>1.</b> Click to set ZONE START (Green) &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>2.</b> Click to set ZONE END (Red) &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>U</b>: Undo last zone &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>ENTER</b>: Finish"
        )
        lbl.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f1f5f9; border-radius: 5px;")
        layout.addWidget(lbl)
        
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setBackground('w')
        layout.addWidget(self.plot)
        
        # draw base track
        self.plot.plot(self.x, self.y, pen=pg.mkPen(color=(150, 150, 150, 100), width=3))
        
        # dynamic scatters
        self.scatter_active = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush('g'))
        self.plot.addItem(self.scatter_active)
        
        self.marker_start = pg.ScatterPlotItem(size=12, pen=pg.mkPen('k'), brush=pg.mkBrush('g'))
        self.plot.addItem(self.marker_start)
        
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_click)

    def get_nearest_index(self, mx, my):
        dist = np.sqrt((self.x - mx)**2 + (self.y - my)**2)
        return np.argmin(dist)

    def on_mouse_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
            idx = self.get_nearest_index(pos.x(), pos.y())
            
            if self.temp_start_idx is None:
                self.temp_start_idx = idx
                self.marker_start.setData([self.x[idx]], [self.y[idx]])
            else:
                self.add_zone(self.temp_start_idx, idx)
                self.temp_start_idx = None
                self.marker_start.setData([], [])
                self.update_visuals()

    def add_zone(self, start, end):
        track_length = self.s[-1]
        dist_direct = abs(self.s[end] - self.s[start])
        dist_wrap = track_length - dist_direct
        
        idx_min, idx_max = min(start, end), max(start, end)
        
        if dist_direct <= dist_wrap:
            self.drs_mask[idx_min:idx_max] = 1.0
            self.zones.append((idx_min, idx_max, False))
        else:
            self.drs_mask[:idx_min] = 1.0
            self.drs_mask[idx_max:] = 1.0
            self.zones.append((idx_min, idx_max, True))

    def update_visuals(self):
        mask_bool = self.drs_mask > 0.5
        if np.any(mask_bool):
            self.scatter_active.setData(self.x[mask_bool], self.y[mask_bool])
        else:
            self.scatter_active.setData([], [])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_U:
            if not self.zones:
                if self.temp_start_idx is not None:
                    self.temp_start_idx = None
                    self.marker_start.setData([], [])
                return
            
            idx_min, idx_max, is_wrap = self.zones.pop()
            if not is_wrap:
                self.drs_mask[idx_min:idx_max] = 0.0
            else:
                self.drs_mask[:idx_min] = 0.0
                self.drs_mask[idx_max:] = 0.0
            self.update_visuals()
            
        elif event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.accept()
            
    def get_mask(self):
        return self.drs_mask