from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
import pyqtgraph as pg
from PyQt6.QtGui import QColor  
from PyQt6.QtCore import Qt

class Universal3DWindow(QWidget):
    def __init__(self, x_grid, y_grid, z_grid, x_label, y_label, z_label, title, parent=None):
        super().__init__(parent)
       
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.plot_title = title
        
        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        

        self.plot_3d(x_grid, y_grid, z_grid)

    def plot_3d(self, X_raw, Y_raw, Z_raw):
        self.fig.clear()
        
        ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
        
        x_flat = X_raw.flatten()
        y_flat = Y_raw.flatten()
        z_flat = Z_raw.flatten()
        
        mask = ~np.isnan(z_flat)
        x_valid = x_flat[mask]
        y_valid = y_flat[mask]
        z_valid = z_flat[mask]
        
        if len(z_valid) < 4:
            print("Not enough points for interpolation")
            return

        resolution = 100 
        xi = np.linspace(min(x_valid), max(x_valid), resolution)
        yi = np.linspace(min(y_valid), max(y_valid), resolution)
        XI, YI = np.meshgrid(xi, yi)
        
        # Cubic interpolation
        ZI = griddata((x_valid, y_valid), z_valid, (XI, YI), method='cubic')
        
        # Plot 3D Surface
        surf = ax3d.plot_surface(XI, YI, ZI, 
                               cmap='viridis', edgecolor='none', 
                               linewidth=0, antialiased=True, alpha=0.9,
                               rcount=resolution, ccount=resolution)
        
      
        ax3d.set_xlabel(self.x_label)
        ax3d.set_ylabel(self.y_label)
        ax3d.set_zlabel(self.z_label)
        ax3d.set_title(self.plot_title)
        
        z_min = np.nanmin(ZI)
        ax3d.contourf(XI, YI, ZI, zdir='z', offset=z_min, cmap='viridis', alpha=0.3)
        ax3d.set_zlim(z_min, np.nanmax(ZI))

        self.fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label=self.z_label)
        ax3d.view_init(elev=25, azim=-135)

        self.fig.tight_layout()
        self.canvas.draw()


class LaptimeSensitivityPlot(QWidget):
    def __init__(self, results, laps):
        super().__init__()
        self.setWindowTitle("Laptime sensitivity results")
        self.resize(1200, 900)
        
        self.all_results = results

        self.results = [r for r in results if getattr(r, 'meta_soc', 0) and abs(r.meta_soc - 1.0) < 1e-5]
        self.laps = laps
        
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

    def open_windows(self):
        if not self.results:
            print("Warning: No results provided to LaptimeSensitivityPlot.")
            return

        acp_widget = self.create_acp_plot()
        lt_widget = self.create_laptimes_energy()
        opt_cap_3d_widget = self.create_optimal_capacity_3d()
        opt_cap_2d_widget = self.create_optimal_capacity_2d() 
        soc_widget = self.create_soc_vs_laptime_plot()

       
        self.tabs.addTab(acp_widget, "Laptime map")
        self.tabs.addTab(lt_widget, "Energy vs points")
        self.tabs.addTab(opt_cap_3d_widget, "Optimal capacity map")
        self.tabs.addTab(opt_cap_2d_widget, "Optimal capacity slices") 
        self.tabs.addTab(soc_widget,"SoC vs laptime")
        
        self.show()

    def get_laptime_functions(self):
        x_vals = sorted(list(set([r.meta_energy*self.laps for r in self.results]))) 
        y_vals = sorted(list(set([r.meta_regen for r in self.results])))  
        
        nx = len(x_vals)
        ny = len(y_vals)
        
        Z_time = np.zeros((nx, ny))
        Z_eff = np.zeros((nx, ny))
        Z_time[:] = np.nan
        Z_eff[:] = np.nan
        
        x_map = {val: i for i, val in enumerate(x_vals)}
        y_map = {val: i for i, val in enumerate(y_vals)}
        
        for r in self.results:
            ix = x_map.get(r.meta_energy*self.laps)
            iy = y_map.get(r.meta_regen)
            if ix is not None and iy is not None:
                t = r.lap_time
                Z_time[ix, iy] = t
                Z_eff[ix, iy] = (t**2) * (r.meta_energy*self.laps)
        
        if np.isnan(Z_time).any():
            Z_time = np.nan_to_num(Z_time, nan=np.nanmax(Z_time))
            Z_eff = np.nan_to_num(Z_eff, nan=np.nanmax(Z_eff))

        interp_method = 'cubic' if (nx >= 4 and ny >= 4) else 'linear'

        interp_time_obj = RegularGridInterpolator((x_vals, y_vals), Z_time, method=interp_method, bounds_error=False, fill_value=None)
        
        def laptime_func(E_target, Regen):
            E_arr = np.atleast_1d(E_target)
            R_arr = np.atleast_1d(Regen)
            if E_arr.shape != R_arr.shape:
                E_arr, R_arr = np.broadcast_arrays(E_arr, R_arr)
                
            pts = np.column_stack((E_arr.ravel(), R_arr.ravel()))
            res = interp_time_obj(pts).reshape(E_arr.shape)
            return float(res) if res.size == 1 else res

        return laptime_func, None

    def create_acp_plot(self):
        x_vals = sorted(list(set([r.meta_energy*self.laps for r in self.results])))
        y_vals = sorted(list(set([r.meta_regen for r in self.results])))
        
        nx = len(x_vals)
        ny = len(y_vals)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros((ny, nx))
        Z[:] = np.nan
        
        x_map = {val: i for i, val in enumerate(x_vals)}
        y_map = {val: i for i, val in enumerate(y_vals)}
        
        for r in self.results:
            ix = x_map.get(r.meta_energy*self.laps)
            iy = y_map.get(r.meta_regen)
            if ix is not None and iy is not None:
                Z[iy, ix] = r.lap_time

        return Universal3DWindow(X, Y, Z, 
                                 x_label='Total energy target [kWh]', 
                                 y_label='Regen percentage [%]', 
                                 z_label='Lap Time [s]', 
                                 title='Lap time sensitivity')

    def create_laptimes_energy(self):
        laptime_func, _ = self.get_laptime_functions()
        
        all_times = np.array([r.lap_time for r in self.results])
        all_energies = np.array([r.meta_energy*self.laps for r in self.results])
        all_eff_factors = (all_times**2) * all_energies
        
        T_min_global = np.min(all_times) * (1 - 0.07)
        Best_Eff_global = np.min(all_eff_factors)
        
        lt_widget = pg.GraphicsLayoutWidget(title="Energy vs laptime and points")
        lt_widget.setBackground('w')
        
        p1 = lt_widget.addPlot(title="Lap Time [s]")
        p1.setLabel('left', "Time", units='s')
        p1.addLegend(offset=(30, 30))

        lt_widget.nextRow()

        p2 = lt_widget.addPlot(title="Total points vs energy used")
        p2.setLabel('left', "Points", units='pts')
        p2.setLabel('bottom', "Total Energy target for endurance", units='kWh')
        p2.setXLink(p1)
        p2.addLegend(offset=(30, 30))
        p2.plot([0], [0], pen=pg.mkPen('k', width=2, style=Qt.PenStyle.SolidLine), name="Total Score")

        min_e, max_e = np.min(all_energies), np.max(all_energies)
        regens_set = sorted(list(set([r.meta_regen for r in self.results])))
        min_r, max_r = min(regens_set), max(regens_set)
        
        x_smooth = np.linspace(min_e, max_e, 200)
        regen_levels = np.linspace(min_r, max_r, 4) 
        
        global_max_points = -1
        global_opt_energy = 0
        
        COLORS = [QColor(0, 114, 178), QColor(213, 94, 0), QColor(0, 158, 115), QColor(204, 121, 167)]
        
        for i, r_val in enumerate(regen_levels):
            y_time = laptime_func(x_smooth, np.full_like(x_smooth, r_val))
            y_eff_factor = (y_time**2) * x_smooth
            
            pts_time = np.zeros_like(y_time)
            T_max = T_min_global * 1.5
            mask_t = y_time <= T_max
            pts_time[mask_t] = (250 - 25) * ((T_max - y_time[mask_t]) / (T_max - T_min_global))**2 + 25
            
            pts_eff = np.zeros_like(y_eff_factor)
            Ef_max = (Best_Eff_global * 2) * 1.2
            mask_e = y_eff_factor <= Ef_max
            pts_eff[mask_e] = 75 * ((Ef_max - y_eff_factor[mask_e]) / (Ef_max - Best_Eff_global))**2
            
            pts_total = pts_time + pts_eff
            
            local_idx = np.argmax(pts_total)
            local_opt_energy = x_smooth[local_idx]
            local_max_points = pts_total[local_idx]
            
            if local_max_points > global_max_points:
                global_max_points = local_max_points
                global_opt_energy = local_opt_energy
            
            base_color = COLORS[i % len(COLORS)]
            label_text = f"Regen {r_val*100:.0f}%"
            
            p1.plot(x_smooth, y_time, pen=pg.mkPen(color=base_color, width=3), name=label_text)
            p2.plot(x_smooth, pts_total, pen=pg.mkPen(color=base_color, width=3, style=Qt.PenStyle.SolidLine))
            
            p2.plot([local_opt_energy], [local_max_points], symbol='o', symbolSize=10, 
                    symbolBrush=base_color, symbolPen='k', name=None)

        v_line = pg.InfiniteLine(pos=global_opt_energy, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashDotLine))
        p2.addItem(v_line)
        p1.addItem(pg.InfiniteLine(pos=global_opt_energy, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashDotLine)))

        info_html = (f"<div style='color:black; background-color: rgba(220,220,255,0.85); "
                     f"padding:5px; border:2px solid blue; font-weight:bold;'>"
                     f"Global Max Points: {global_max_points:.1f} pts<br>"
                     f"Opt Total Energy: {global_opt_energy:.2f} kWh </div>")
        
        text = pg.TextItem(html=info_html, anchor=(0.5, 0))
        text.setPos(global_opt_energy, global_max_points)
        p2.addItem(text)
        
        return lt_widget

    def create_optimal_capacity_3d(self):
        laptime_func, _ = self.get_laptime_functions()
        
        all_times = np.array([r.lap_time for r in self.results])
        all_energies = np.array([r.meta_energy*self.laps for r in self.results])
        all_eff_factors = (all_times**2) * all_energies
        
        base_t_min = np.min(all_times)
        best_eff_glob = np.min(all_eff_factors)
        
        regens_set = sorted(list(set([r.meta_regen for r in self.results])))
        min_r, max_r = min(regens_set), max(regens_set)
        min_e, max_e = np.min(all_energies), np.max(all_energies)
        
        regen_grid = np.linspace(min_r, max_r, 40)
        offset_grid = np.linspace(0.0, 25.0, 40) 
        
        X_regen, Y_offset = np.meshgrid(regen_grid, offset_grid)
        Z_opt_cap = np.zeros_like(X_regen)
        
        e_test = np.linspace(min_e, max_e, 200)
        
        for i in range(X_regen.shape[0]):
            for j in range(X_regen.shape[1]):
                r_val = X_regen[i, j]
                offset_pct = Y_offset[i, j]
                
                t_min_glob = base_t_min * (1.0 - (offset_pct / 100.0))
                
                y_time = laptime_func(e_test, np.full_like(e_test, r_val))
                y_eff_factor = (y_time**2) * e_test
                
                pts_time = np.zeros_like(y_time)
                T_max = t_min_glob * 1.5
                mask_t = y_time <= T_max
                if np.any(mask_t):
                    pts_time[mask_t] = (250 - 25) * ((T_max - y_time[mask_t]) / (T_max - t_min_glob))**2 + 25
                
                pts_eff = np.zeros_like(y_eff_factor)
                Ef_max = (best_eff_glob * 2) * 1.2
                mask_e = y_eff_factor <= Ef_max
                if np.any(mask_e):
                    pts_eff[mask_e] = 75 * ((Ef_max - y_eff_factor[mask_e]) / (Ef_max - best_eff_glob))**2
                
                pts_total = pts_time + pts_eff
                best_idx = np.argmax(pts_total)
                Z_opt_cap[i, j] = e_test[best_idx]

        return Universal3DWindow(
            X_regen, Y_offset, Z_opt_cap, 
            x_label='Regen percentage [%]', 
            y_label='T_min Offset [%]', 
            z_label='Optimal Capacity [kWh]', 
            title='Optimal Capacity vs Regen & Time Offset'
        )

    
    def create_optimal_capacity_2d(self):
        """Vygeneruje 2D plot: Osa X = Regen, Osa Y = Optimal Capacity pro různé Time Offsets."""
        laptime_func, _ = self.get_laptime_functions()
        
        all_times = np.array([r.lap_time for r in self.results])
        all_energies = np.array([r.meta_energy*self.laps for r in self.results])
        all_eff_factors = (all_times**2) * all_energies
        
        base_t_min = np.min(all_times)
        best_eff_glob = np.min(all_eff_factors)
        
        regens_set = sorted(list(set([r.meta_regen for r in self.results])))
        min_r, max_r = min(regens_set), max(regens_set)
        min_e, max_e = np.min(all_energies), np.max(all_energies)
        
        regen_test = np.linspace(min_r, max_r, 50)
        e_test = np.linspace(min_e, max_e, 200)
        
        # Testované offsety (0% až 25% pomalejší konkurence)
        offsets = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        
        plot_widget = pg.PlotWidget(title="Optimal capacity vs regen for different times")
        plot_widget.setBackground('w')
        plot_widget.setLabel('bottom', "Regen percentage", units="%")
        plot_widget.setLabel('left', "Optimal capacity", units="kWh")
        plot_widget.addLegend(offset=(30, 30))
        plot_widget.showGrid(x=True, y=True, alpha=0.1)
        
        for i, offset_pct in enumerate(offsets):
            opt_caps = []
            
            
            t_min_glob = base_t_min * (1.0 - (offset_pct / 100.0))
            T_max = t_min_glob * 1.5
            Ef_max = (best_eff_glob * 2) * 1.2
            
            for r_val in regen_test:
                y_time = laptime_func(e_test, np.full_like(e_test, r_val))
                y_eff_factor = (y_time**2) * e_test
                
                pts_time = np.zeros_like(y_time)
                mask_t = y_time <= T_max
                if np.any(mask_t):
                    pts_time[mask_t] = (250 - 25) * ((T_max - y_time[mask_t]) / (T_max - t_min_glob))**2 + 25
                
                pts_eff = np.zeros_like(y_eff_factor)
                mask_e = y_eff_factor <= Ef_max
                if np.any(mask_e):
                    pts_eff[mask_e] = 75 * ((Ef_max - y_eff_factor[mask_e]) / (Ef_max - best_eff_glob))**2
                
                pts_total = pts_time + pts_eff
                best_idx = np.argmax(pts_total)
                opt_caps.append(e_test[best_idx])
            
            # Vygenerování unikátní barvy pro každý offset
            color = pg.intColor(i, hues=len(offsets))
            
           
            plot_widget.plot(regen_test * 100, opt_caps, pen=pg.mkPen(color=color, width=3), name=f"Time offset {offset_pct:.0f}%")
            
        return plot_widget
    
    def create_soc_vs_laptime_plot(self):
        # 1. Zjistíme unikátní hodnoty z celého datasetu
        e_targets = sorted(list(set([r.meta_energy for r in self.all_results])))
        regens = sorted(list(set([r.meta_regen for r in self.all_results])))
        socs = sorted(list(set([r.meta_soc for r in self.all_results])))
        
        if not e_targets or not regens or not socs:
            return QWidget()
            
        # 2. Vybereme fixní hodnoty (prostřední E_target a maximální Regen)
        fixed_e_target = e_targets[len(e_targets) // 2]
        fixed_regen = regens[-1]
        
        # 3. Vyfiltrujeme data
        plot_data = []
        for r in self.all_results:
            # Porovnáváme floaty s tolerancí (abs < 1e-5)
            if abs(r.meta_energy - fixed_e_target) < 1e-5 and abs(r.meta_regen - fixed_regen) < 1e-5:
                plot_data.append((r.meta_soc, r.lap_time))
                
        # Seřadíme podle SoC (od nejmenšího po největší)
        plot_data.sort(key=lambda x: x[0])
        
        x_vals = [d[0] * 100 for d in plot_data] # Převod SoC na procenta (Osa X)
        y_vals = [d[1] for d in plot_data]       # Laptime (Osa Y)
        
        # 4. Vytvoření PyQtGraph widgetu
        title = f"SoC vs Laptime (Fixed Target = {fixed_e_target*self.laps:.2f} kWh, Regen = {fixed_regen*100:.0f}%)"
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setBackground('w')
        
        plot_widget.setLabel('bottom', "State of Charge (SoC)", units="%")
        plot_widget.setLabel('left', "Lap Time", units="s")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # 5. Vykreslení čáry a bodů
        line_pen = pg.mkPen(color=QColor(213, 94, 0), width=3) # Oranžová čára
        plot_widget.plot(x_vals, y_vals, pen=line_pen, symbol='o', symbolSize=10, 
                         symbolBrush=QColor(0, 114, 178), symbolPen='k') # Modré body s černým okrajem
        
        # 6. Invertování osy X (aby 100% bylo vlevo a 0% vpravo, jak se baterie vybíjí)
        plot_widget.getViewBox().invertX(True)
        
        return plot_widget