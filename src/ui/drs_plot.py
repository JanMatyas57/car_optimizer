from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np


class DRSResultsWindow(QWidget):
    def __init__(self, drs_data, nodrs_data):
        super().__init__()

        self.drs_data = drs_data
        self.nodrs_data = nodrs_data

        self.setWindowTitle("DRS analysis")
        self.resize(1100, 800)

        layout = QVBoxLayout(self)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        self.plot_results()

    def plot_results(self):
        # Extract arrays
        E_drs = np.array([d["target"] for d in self.drs_data])
        T_drs = np.array([d["laptime"] for d in self.drs_data])
        U_drs = np.array([d["drs_active_ratio"] for d in self.drs_data])
        Ef_drs = np.array([d["ef_score"] for d in self.drs_data])

        E_no = np.array([d["target"] for d in self.nodrs_data])
        T_no = np.array([d["laptime"] for d in self.nodrs_data])
        Ef_no = np.array([d["ef_score"] for d in self.nodrs_data])

        # Sort by energy
        idx_drs = np.argsort(E_drs)
        idx_no = np.argsort(E_no)

        E_drs, T_drs, U_drs, Ef_drs = E_drs[idx_drs], T_drs[idx_drs], U_drs[idx_drs], Ef_drs[idx_drs]
        E_no, T_no, Ef_no = E_no[idx_no], T_no[idx_no], Ef_no[idx_no]

        # Delta time (positive = DRS helped)
        T_delta = T_no - T_drs

        # --- Plot 1: Delta Time ---
        p1 = self.win.addPlot(title="DRS time gain per lap vs energy target")
        p1.setLabel("bottom", "Endurance energy target", units="kWh")
        p1.setLabel("left", "Delta time", units="s")
        p1.showGrid(x=True, y=True, alpha=0.1)
        p1.invertX(True)

        p1.plot(
            E_drs, T_delta,
            pen=pg.mkPen("b", width=3),
            symbol="o",
            symbolSize=8
        )

        #efficiency score plot
        self.win.nextRow()
        p2 = self.win.addPlot(title="Efficiency score (lower better)")
        p2.setLabel("bottom", "Endurance energy target", units="kWh/lap")
        p2.setLabel("left", "Efficiency score", units="kWh·s²")
        p2.showGrid(x=True, y=True, alpha=0.1)
        p2.setXLink(p1)
        p2.invertX(True)
        p2.addLegend(offset=(30, 30))
      
        p2.plot(
            E_drs, Ef_drs,
            pen=pg.mkPen("r", width=3),
            symbol="o",
            symbolSize=8,
            name="With DRS"
        )
        p2.plot(
            E_no, Ef_no,
            pen=pg.mkPen("orange", width=3),
            symbol="s",
            symbolSize=8,
            name="No DRS"
        )

        self.win.nextRow()

        p3 = self.win.addPlot(title="DRS usage")
        p3.setLabel("bottom", "Endurance energy target", units="kWh")
        p3.setLabel("left", "DRS active ratio", units="%")
        p3.showGrid(x=True, y=True, alpha=0.1)
        p3.setXLink(p1)
        p3.invertX(True)
   

        p3.plot(
            E_drs, U_drs,
            pen=pg.mkPen("g", width=3),
            symbol="o",
            symbolSize=8
        )

     