import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import casadi as ca
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator, interp1d
from matplotlib.widgets import SpanSelector

# ==========================================
# 1. OPTIMIZATION & SMOOTHING CLASS
# ==========================================

class TrackGenerator:
    @staticmethod
    def smooth_track(x_smpl, y_smpl, z_smpl, ds=0.05, r=1000, is_closed=True):
        x_smpl = np.array(x_smpl).flatten()
        y_smpl = np.array(y_smpl).flatten()
        z_smpl = np.array(z_smpl).flatten()
        
        dx = np.diff(x_smpl)
        dy = np.diff(y_smpl)
        ds_tmp = np.sqrt(dx**2 + dy**2)
        s_tmp = np.concatenate(([0], np.cumsum(ds_tmp)))
        
        total_length = s_tmp[-1]
        N = int(np.round(total_length / ds))
        if N < 2: N = 2 
        ds_real = total_length / N
        s_traj = np.linspace(0, total_length, N) 
        
        # Initial Interpolation
        fx = PchipInterpolator(s_tmp, x_smpl)
        fy = PchipInterpolator(s_tmp, y_smpl)
        fz = PchipInterpolator(s_tmp, z_smpl)
        x_init = fx(s_traj)
        y_init = fy(s_traj)
        z_init = fz(s_traj)
        
        # Init alpha, theta, C
        dz_init = np.diff(z_init) / ds_real
        alpha_init = np.arcsin(np.clip(dz_init, -0.2, 0.2))
        alpha_init = np.append(alpha_init, alpha_init[-1])

        th_init = np.unwrap(np.arctan2(np.diff(y_init), np.diff(x_init)))
        th_init = np.append(th_init, th_init[-1])
        C_init = np.append(np.diff(th_init) / ds_real, 0)
        
        # --- CasADi Optimization ---
        opti = ca.Opti()
        
        # State: [kappa, theta, x, y, alpha, z]
        Z = opti.variable(N, 6) 
        z_C, z_th, z_x, z_y, z_alp, z_z = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4], Z[:,5]
        
        # Control: [d_kappa/ds, d_alpha/ds]
        u = opti.variable(N-1, 2) 
        
        # Cost Function
        x_dev = ((z_x[:-1] - x_init[:-1])**2 + (z_x[1:] - x_init[1:])**2) / 2
        y_dev = ((z_y[:-1] - y_init[:-1])**2 + (z_y[1:] - y_init[1:])**2) / 2
        z_dev = ((z_z[:-1] - z_init[:-1])**2 + (z_z[1:] - z_init[1:])**2) / 2
        
        reg_term = r * u[:,0]**2 + 10*r * u[:,1]**2
        
        cost = ca.sum1(ds_real * (reg_term + x_dev + y_dev + z_dev))
        opti.minimize(cost)
        
        # Dynamics
        def f_dyn(z_in, u_in):
            return ca.vertcat(u_in[0],              # dC/ds
                              z_in[0],              # dth/ds = kappa
                              ca.cos(z_in[1]),      # dx/ds = cos(theta)
                              ca.sin(z_in[1]),      # dy/ds = sin(theta)
                              u_in[1],              # dalp/ds
                              ca.sin(z_in[4]))      # dz/ds = sin(alpha)

        for k in range(N-1):
            z_k, u_k = Z[k, :].T, u[k, :].T
            k1 = f_dyn(z_k, u_k)
            k2 = f_dyn(z_k + ds_real/2*k1, u_k)
            k3 = f_dyn(z_k + ds_real/2*k2, u_k)
            k4 = f_dyn(z_k + ds_real*k3, u_k)
            opti.subject_to(Z[k+1, :].T == z_k + ds_real/6 * (k1 + 2*k2 + 2*k3 + k4))

        # Constraints
        if is_closed:
            opti.subject_to(Z[-1, 2] == Z[0, 2]) # x
            opti.subject_to(Z[-1, 3] == Z[0, 3]) # y
            opti.subject_to(Z[-1, 5] == Z[0, 5]) # z
            opti.subject_to(Z[-1, 0] == Z[0, 0]) # kappa
            opti.subject_to(Z[-1, 4] == Z[0, 4]) # alpha
            n_loops = np.round((th_init[-1] - th_init[0]) / (2*np.pi))
            opti.subject_to(z_th[-1] == z_th[0] + 2*np.pi*n_loops)

        # Init guess
        opti.set_initial(z_x, x_init)
        opti.set_initial(z_y, y_init)
        opti.set_initial(z_z, z_init)
        opti.set_initial(z_th, th_init)
        opti.set_initial(z_C, C_init)
        opti.set_initial(z_alp, alpha_init)
        
        opti.solver('ipopt', {'expand': True}, {'max_iter': 500, 'print_level': 0})
        
        try:
            sol = opti.solve()
            Z_opt = sol.value(Z)
            return {
                's': s_traj, 'kappa': Z_opt[:, 0], 'psi': Z_opt[:, 1], 
                'x': Z_opt[:, 2], 'y': Z_opt[:, 3], 'alpha': Z_opt[:, 4], 'z': Z_opt[:, 5]
            }
        except Exception as e:
            print(f"Optimalizace selhala: {e}")
            return {'s': s_traj, 'kappa': C_init, 'psi': th_init, 'x': x_init, 'y': y_init, 'alpha': alpha_init, 'z': z_init}


    @staticmethod
    def smooth_track2(x_smpl, y_smpl, z_smpl, ds=0.1,w_pos=50.0, r=1e1, is_closed=True):
        """
        Vyhladi trajektorii. Matematika cost function je sjednocena s MATLAB OCP.
        """
        x_smpl = np.array(x_smpl).flatten()
        y_smpl = np.array(y_smpl).flatten()
        z_smpl = np.array(z_smpl).flatten()
        
        dx = np.diff(x_smpl)
        dy = np.diff(y_smpl)
        ds_tmp = np.sqrt(dx**2 + dy**2)
        s_tmp = np.concatenate(([0], np.cumsum(ds_tmp)))
        
        total_length = s_tmp[-1]
        N = int(np.round(total_length / ds))
        if N < 2: N = 2 
        ds_real = total_length / N
        s_traj = np.linspace(0, total_length, N) 
        
        # Prvotní interpolace
        fx = PchipInterpolator(s_tmp, x_smpl)
        fy = PchipInterpolator(s_tmp, y_smpl)
        fz = PchipInterpolator(s_tmp, z_smpl)
        x_init = fx(s_traj)
        y_init = fy(s_traj)
        z_init = fz(s_traj)
        
        # Init alpha, theta, C
        dz_init = np.diff(z_init) / ds_real
        alpha_init = np.arcsin(np.clip(dz_init, -0.2, 0.2))
        alpha_init = np.append(alpha_init, alpha_init[-1])

        th_init = np.unwrap(np.arctan2(np.diff(y_init), np.diff(x_init)))
        th_init = np.append(th_init, th_init[-1])
        C_init = np.append(np.diff(th_init) / ds_real, 0)
        
        # --- CasADi Optimalizace ---
        opti = ca.Opti()
        
        # Stav: [kappa, theta, x, y, alpha, z]
        Z = opti.variable(N, 6) 
        z_C, z_th, z_x, z_y, z_alp, z_z = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4], Z[:,5]
        
        # Control: [d_kappa/ds, d_alpha/ds]
        u = opti.variable(N-1, 2) 
        

        # 1. Odchylky pro X a Y (přesně jako Matlab)
        x_dev = w_pos*((z_x[:-1] - x_init[:-1])**2 + (z_x[1:] - x_init[1:])**2) / 2
        y_dev = w_pos*((z_y[:-1] - y_init[:-1])**2 + (z_y[1:] - y_init[1:])**2) / 2
        
        # 2. Odchylka pro Z (přidáno navíc oproti Matlabu, ale stejnou logikou)
        z_dev = w_pos*((z_z[:-1] - z_init[:-1])**2 + (z_z[1:] - z_init[1:])**2) / 2
        
        
        # 3. Regularizace (smoothing) - Matlab má r*u^2
        # Zde máme dvě u (křivost a sklon), oběma dáváme váhu 'r'
        reg_term = r * u[:,0]**2 + 10*r * u[:,1]**2
        
        # Celková cost function
        cost = ca.sum1(ds_real * (reg_term + x_dev + y_dev + z_dev))
        
        opti.minimize(cost)
        
        # --- DYNAMIKA (RK4) ---
        def f_dyn(z_in, u_in):
            return ca.vertcat(u_in[0],           # dC/ds
                              z_in[0],           # dth/ds = kappa
                              ca.cos(z_in[1]),   # dx/ds = cos(theta)
                              ca.sin(z_in[1]),   # dy/ds = sin(theta)
                              u_in[1],           # dalp/ds
                              ca.sin(z_in[4]))   # dz/ds = sin(alpha)

        for k in range(N-1):
            z_k, u_k = Z[k, :].T, u[k, :].T
            k1 = f_dyn(z_k, u_k)
            k2 = f_dyn(z_k + ds_real/2*k1, u_k)
            k3 = f_dyn(z_k + ds_real/2*k2, u_k)
            k4 = f_dyn(z_k + ds_real*k3, u_k)
            opti.subject_to(Z[k+1, :].T == z_k + ds_real/6 * (k1 + 2*k2 + 2*k3 + k4))

        # --- OMEZENÍ (Constraints) ---
        if is_closed:
            opti.subject_to(Z[-1, 2] == Z[0, 2]) # x
            opti.subject_to(Z[-1, 3] == Z[0, 3]) # y
            opti.subject_to(Z[-1, 5] == Z[0, 5]) # z
            opti.subject_to(Z[-1, 0] == Z[0, 0]) # kappa
            opti.subject_to(Z[-1, 4] == Z[0, 4]) # alpha
            n_loops = np.round((th_init[-1] - th_init[0]) / (2*np.pi))
            opti.subject_to(z_th[-1] == z_th[0] + 2*np.pi*n_loops)

        # Inicializace
        opti.set_initial(z_x, x_init)
        opti.set_initial(z_y, y_init)
        opti.set_initial(z_z, z_init)
        opti.set_initial(z_th, th_init)
        opti.set_initial(z_C, C_init)
        opti.set_initial(z_alp, alpha_init)
        
        # Nastavení solveru (stejně jako v Matlabu)
        opti.solver('ipopt', {'expand': True}, {'max_iter': 500, 'print_level': 0})
        
        try:
            sol = opti.solve()
            Z_opt = sol.value(Z)
            return {
                's': s_traj, 'kappa': Z_opt[:, 0], 'psi': Z_opt[:, 1], 
                'x': Z_opt[:, 2], 'y': Z_opt[:, 3], 'alpha': Z_opt[:, 4], 'z': Z_opt[:, 5]
            }
        except Exception as e:
            print(f"Optimalizace selhala: {e}")
            return {'s': s_traj, 'kappa': C_init, 'psi': th_init, 'x': x_init, 'y': y_init, 'alpha': alpha_init, 'z': z_init}
# ==========================================
# 2. HELPERS & DATA LOADERS
# ==========================================
    @staticmethod
    def latlon_to_meters(lat, lon):
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        lat0, lon0 = lat_rad[0], lon_rad[0]
        R = 6371000 
        x = R * (lon_rad - lon0) * np.cos(lat0)
        y = R * (lat_rad - lat0)
        return x, y
    
    @staticmethod
    def filter_data(mat_file):
        print("Loading MAT data...")
        mat_data = scipy.io.loadmat(mat_file)
        
        # Extract data (zkráceno pro přehlednost - logika stejná)
        try:
            ins = mat_data['INS'][0,0]
            ams = mat_data['AMS'][0,0]
            pdl = mat_data['PDL'][0,0]
            
            gps = ins['EKF_POS'][0,0]
            vel = ins['EKF_VEL_BODY'][0,0]
            alt = ins['EKF_ALTITUDE'][0,0]
            acc = ins['EKF_POS_ACC'][0,0]
            
            longitude = np.array(gps['Longitude']).flatten()
            latitude = np.array(gps['Latitude']).flatten()
            gps_time = np.array(gps['t']).flatten()
            acc_time = np.array(acc['t']).flatten()
            
            speed = np.array(vel['VELOCITY_X']).flatten()
            speed_time = np.array(vel['t']).flatten()
            
            altitude = np.array(alt['Altitude']).flatten()
            altitude_time = np.array(alt['t']).flatten()

            long_acc = np.array(acc['Longitude_acc']).flatten()
            lat_acc = np.array(acc['Latitude_acc']).flatten()
            altitude_acc = np.array(acc['Altitude_acc']).flatten()
            # Interpolace
            lat_interp = np.interp(speed_time, gps_time, latitude)
            lat_acc_interp = np.interp(speed_time, acc_time, lat_acc)
            lon_interp = np.interp(speed_time, gps_time, longitude)
            long_acc_interp = np.interp(speed_time, acc_time, long_acc)
            alt_interp = np.interp(speed_time, altitude_time, altitude)
            altitude_acc_interp = np.interp(speed_time, acc_time, altitude_acc)
            
            # Power & Brake
            p_out = np.array(ams['TsMeas'][0,0]['Pout']).flatten()
            p_time = np.array(ams['TsMeas'][0,0]['t']).flatten()
            p_interp = np.interp(speed_time, p_time, p_out)
            
            b_front = np.array(pdl['Brakes'][0,0]['Front']).flatten()
            b_rear = np.array(pdl['Brakes'][0,0]['Rear']).flatten()
            b_time = np.array(pdl['Brakes'][0,0]['t']).flatten()
            b_interp = np.interp(speed_time, b_time, b_front + b_rear)
            
            mask = (lat_acc_interp < 1.5) & (long_acc_interp < 1.5) & (altitude_acc_interp < 1.5)
            valid_indices = np.where(mask)[0]
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            
            return {
                'Speed': speed[start_idx:end_idx], 
                'Latitude': lat_interp[start_idx:end_idx],
                'Longitude': lon_interp[start_idx:end_idx], 
                'Altitude': alt_interp[start_idx:end_idx],
                'Power': p_interp[start_idx:end_idx],
                'Brake': b_interp[start_idx:end_idx], 
                'Time': speed_time[start_idx:end_idx]
            }
        except Exception as e:
            print(f"Error parsing MAT file structure: {e}")
            return None

# ==========================================
# 3. INTERACTIVE SELECTORS (Matplotlib)
# ==========================================
    @staticmethod
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

    @staticmethod
    def get_lap_indices(data, point, tangent_vector):
        lat, lon = data["Latitude"], data["Longitude"]
        rel_long, rel_lat = lon - point[1], lat - point[0]
        vec_norm = np.dot(tangent_vector, tangent_vector)
        projected = np.array([np.dot([rel_long[i], rel_lat[i]], tangent_vector) / vec_norm for i in range(len(lon))])
        minima_indices, _ = find_peaks(-np.abs(projected))
        return [i for i in minima_indices if (abs(rel_long[i]) + abs(rel_lat[i])) < 0.0002]

 
    @staticmethod
    def run_lap_selector(filtered_data):
        # --- FÁZE 1: Výběr startovní čáry ---
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.plot(filtered_data["Longitude"], filtered_data["Latitude"], ",", color='gray', alpha=0.3)
        ax1.set_title("1. Click on start/finish (Direction matters!)\n2. Press ENTER to confirm")
        
        selected = {'idx': None, 'vec': None, 'laps': []}
        temp_plot = []

        def on_click(event):
            if event.inaxes != ax1: return
            for e in temp_plot: e.remove()
            temp_plot.clear()
            # Předpokládám, že get_tangent_vector vrací (index, vektor)
            idx, vec = TrackGenerator.get_tangent_vector(filtered_data, event.ydata, event.xdata)
            selected.update({'idx': idx, 'vec': vec})
            
            p, = ax1.plot(filtered_data["Longitude"][idx], filtered_data["Latitude"][idx], "ro", markersize=10)
            # Volitelně: vykreslit šipku směru, aby uživatel viděl orientaci
            arrow = ax1.arrow(filtered_data["Longitude"][idx], filtered_data["Latitude"][idx], 
                              vec[0]*0.0001, vec[1]*0.0001, head_width=0.00001, color='red')
            temp_plot.append(p)
            temp_plot.append(arrow)
            fig1.canvas.draw()

        def on_key_init(event):
            if event.key == "enter" and selected['idx'] is not None:
                pt = [filtered_data["Latitude"][selected['idx']], filtered_data["Longitude"][selected['idx']]]
                selected['laps'] = TrackGenerator.get_lap_indices(filtered_data, pt, selected['vec'])
                
                # DŮLEŽITÉ: Zastavit smyčku a zavřít
                fig1.canvas.stop_event_loop()
                plt.close(fig1)

        # Bezpečnostní handler pro zavření křížkem
        def handle_close1(evt):
            fig1.canvas.stop_event_loop()

        fig1.canvas.mpl_connect("button_press_event", on_click)
        fig1.canvas.mpl_connect("key_press_event", on_key_init)
        fig1.canvas.mpl_connect("close_event", handle_close1)
        
        # --- FIX PRO QT EVENT LOOP ---
        plt.show(block=False)
        fig1.canvas.start_event_loop(timeout=-1)
        # -----------------------------

        if len(selected['laps']) < 2:
            return None

        # --- FÁZE 2: Listování koly ---
        lap_indices = selected['laps']
        laps_data = []
        for i in range(len(lap_indices) - 1):
            s, e = lap_indices[i], lap_indices[i+1]
            laps_data.append({
                'indices': (s, e),
                'time': filtered_data["Time"][e] - filtered_data["Time"][s]
            })

        state = {'current': 0, 'confirmed_lap': None}

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        def update_plot():
            ax2.clear()
            idx = state['current']
            s, e = laps_data[idx]['indices']
            
            ax2.plot(filtered_data["Longitude"], filtered_data["Latitude"], ",", color='gray', alpha=0.1)
            ax2.plot(filtered_data["Longitude"][s:e], filtered_data["Latitude"][s:e], "b-", linewidth=2)
            
            ax2.set_title(f"Lap {idx+1}/{len(laps_data)} | Time: {laps_data[idx]['time']:.3f}s\n"
                        f"Arrows: browse laps | Enter: Save and close | Esc: Cancel")
            fig2.canvas.draw()

        def on_key_browse(event):
            if event.key == "right":
                state['current'] = (state['current'] + 1) % len(laps_data)
            elif event.key == "left":
                state['current'] = (state['current'] - 1) % len(laps_data)
            elif event.key == "enter":
                state['confirmed_lap'] = laps_data[state['current']]
                fig2.canvas.stop_event_loop()
                plt.close(fig2)
            elif event.key == "escape":
                fig2.canvas.stop_event_loop()
                plt.close(fig2)
            update_plot()

        def handle_close2(evt):
            fig2.canvas.stop_event_loop()

        fig2.canvas.mpl_connect("key_press_event", on_key_browse)
        fig2.canvas.mpl_connect("close_event", handle_close2)
        
        update_plot()
        
        # --- FIX PRO QT EVENT LOOP ---
        plt.show(block=False)
        fig2.canvas.start_event_loop(timeout=-1)
        # -----------------------------

        return state['confirmed_lap']
 
    @staticmethod
    def manual_lap_selector(data):
        t = data["Time"]
        v = data["Speed"]
        p = data["Power"]
        
        state = {"idx": None, "confirmed": False}
        
        
        current_selection = [None] 

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        
       
        axs[0].plot(t, v, label="Speed [m/s]", color='#1f77b4')
        axs[1].plot(t, p, label="Power [kW]", color='#ff7f0e')
        
 
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(t, ymin, ymax, where=(v < 3.0), 
                            facecolor='red', alpha=0.15, label="Speed < 3 m/s (don't use, method creating track will crash))", zorder=0)
        
      
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        axs[1].grid(True, alpha=0.3)

        def onselect(t_min, t_max):
            # Přepočet času na indexy
            s = np.searchsorted(t, t_min)
            e = np.searchsorted(t, t_max)
            state["idx"] = (s, e)
            
            # --- FIX: VIZUÁLNÍ PERSISTENCE ---
            # 1. Smazat starý výběr, pokud existuje
            if current_selection[0] is not None:
                current_selection[0].remove()
            
            # 2. Vykreslit nový trvalý obdélník přes oba grafy
            # Použijeme žlutou barvu (stejnou jako SpanSelector)
            rect = axs[0].axvspan(t_min, t_max, color='yellow', alpha=0.3)
            axs[1].axvspan(t_min, t_max, color='yellow', alpha=0.3) # Volitelně i dole
            
            # Uložíme referenci (stačí na jeden, protože se mažou překreslením canvasu nebo referencí)
            # Zde pro jednoduchost mažeme jen ten horní přes referenci, 
            # spodní se smaže překreslením celé figury nebo bychom museli držet list.
            current_selection[0] = rect
            
            print(f"Selected: {t[e]-t[s]:.2f}s Press ENTER to save")
            fig.canvas.draw()

        # Nastavení SpanSelectoru
        # props=dict(alpha=0) znamená, že samotný selector bude při tažení neviditelný 
        # (nebo průhledný), pokud chceme spoléhat jen na naše vykreslování, 
        # ale nechme ho viditelný pro UX při tažení.
        span = SpanSelector(
            axs[0], 
            onselect, 
            "horizontal", 
            useblit=True,
            props=dict(facecolor="yellow", alpha=0.3)
        )
        
        plt.suptitle("1. Drag with mouse to select the segment\n2. Press ENTER to save")
        
        def on_key(event):
            if event.key == "enter" and state["idx"] is not None:
                state["confirmed"] = True
               
                fig.canvas.stop_event_loop()
                plt.close(fig)
            elif event.key == "escape":
             
                fig.canvas.stop_event_loop()
                plt.close(fig)

        def handle_close(evt):
            fig.canvas.stop_event_loop()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", handle_close)
        
        
        plt.show(block=False)
        fig.canvas.start_event_loop(timeout=-1)
        
        if not state["confirmed"] or state["idx"] is None:
            return None
            
        s, e = state["idx"]
        return {"indices": (s, e), "time": t[e] - t[s]}

 

    @staticmethod
    def process_and_save_lap(all_data, lap_info, is_closed, save_path):
        s, e = lap_info['indices']
        
        # 1. Extrakce surových dat
        raw_lat = all_data['Latitude'][s:e]
        raw_lon = all_data['Longitude'][s:e]
        raw_alt = all_data['Altitude'][s:e]
        raw_speed = all_data['Speed'][s:e]
        raw_power = all_data['Power'][s:e]
        raw_brake = all_data['Brake'][s:e]

        # 2. Příprava masek
        # A) Speed Mask
        speed_mask = raw_speed > 1.0
        
        # B) Distance Mask (převod na metry pro výpočet vzdáleností)
        x_m_temp, y_m_temp = TrackGenerator.latlon_to_meters(raw_lat, raw_lon)
        dx = np.diff(x_m_temp, prepend=x_m_temp[0])
        dy = np.diff(y_m_temp, prepend=y_m_temp[0])
        dist_sq = dx**2 + dy**2
        dist_mask = np.concatenate(([True], dist_sq[1:] > 1e-6)) # 1mm tolerance

        # 3. Aplikace kombinované masky na VŠECHNA DATA
        final_mask = speed_mask & dist_mask
        
        # DŮLEŽITÉ: Musíme zkrátit všechna pole, aby seděla délka
        raw_lat = raw_lat[final_mask]
        raw_lon = raw_lon[final_mask]
        raw_alt = raw_alt[final_mask]
        raw_speed = raw_speed[final_mask]
        raw_power = raw_power[final_mask]
        raw_brake = raw_brake[final_mask]

        # Znovu přepočítáme x_m, y_m už z vyfiltrovaných Lat/Lon
        # (Nebo bychom mohli filtrovat x_m_temp, ale takhle je to čistší)
        x_m, y_m = TrackGenerator.latlon_to_meters(raw_lat, raw_lon)

        if len(x_m) < 10:
            msg = "Error: Not enough valid data points. Car must be strictly in motion (> 1 m/s)."
            return False, msg # <--- VRACÍ TUPLE
        
        # 4. Smoothing
        tg = TrackGenerator()
        smoothed = tg.smooth_track(x_m, y_m, raw_alt, ds=0.1, r=1e-3, is_closed=is_closed)
        
        if smoothed is None: 
            return False, "Smoothing optimization failed (IPOPT error)." # <--- VRACÍ TUPLE
        
        # 5. Interpolace Telemetrie
        # s_raw počítáme z FILTROVANÝCH dat (x_m, y_m mají nyní stejnou délku jako raw_speed atd.)
        s_raw = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))))
        
        def interp_on_s(data_arr):
            # data_arr musí mít stejnou délku jako s_raw!
            f = interp1d(s_raw, data_arr, kind='linear', fill_value="extrapolate")
            return f(smoothed['s'])

        final_data = {
            's': smoothed['s'],
            'x': smoothed['x'],
            'y': smoothed['y'],
            'z': smoothed['z'],
            'kappa': smoothed['kappa'],
            'psi': smoothed['psi'],
            'alpha': smoothed['alpha'],
            'speed': interp_on_s(raw_speed), 
            'power': interp_on_s(raw_power),
            'brake': interp_on_s(raw_brake),
        }
        
        np.savez(save_path, **final_data)
        
        return True, "Track generated successfully." 

# ==========================================
# 4. PROCESSING & SAVING
# ==========================================
    @staticmethod
    def process_and_save_lap(all_data, lap_info, is_closed, save_path):
        s, e = lap_info['indices']
        
        # 1. Extrakce surových dat
        raw_lat = all_data['Latitude'][s:e]
        raw_lon = all_data['Longitude'][s:e]
        raw_alt = all_data['Altitude'][s:e]
        raw_speed = all_data['Speed'][s:e]
        raw_power = all_data['Power'][s:e]
        raw_brake = all_data['Brake'][s:e]

        # 2. Vytvoření masek (Filtry)
        # A) Speed Mask (Auto musí jet > 1 m/s)
        speed_mask = raw_speed > 1.0
        
        # B) Distance Mask (převod na metry pro zjištění duplicit)
        x_m_temp, y_m_temp = TrackGenerator.latlon_to_meters(raw_lat, raw_lon)
        dx = np.diff(x_m_temp, prepend=x_m_temp[0])
        dy = np.diff(y_m_temp, prepend=y_m_temp[0])
        dist_sq = dx**2 + dy**2
        dist_mask = np.concatenate(([True], dist_sq[1:] > 1e-6)) # tolerance 1mm

        # 3. Aplikace kombinované masky na VŠECHNA DATA
        final_mask = speed_mask & dist_mask
        
        # Filtrujeme souřadnice
        x_m = x_m_temp[final_mask]
        y_m = y_m_temp[final_mask]
        
        # Filtrujeme telemetrii (DŮLEŽITÉ pro pozdější interpolaci!)
        raw_alt = raw_alt[final_mask]
        raw_speed = raw_speed[final_mask]
        raw_power = raw_power[final_mask]
        raw_brake = raw_brake[final_mask]

        # --- FIX 1: Return Tuple ---
        if len(x_m) < 10:
            msg = "Error: Not enough valid data points. Car must be moving (> 1 m/s)."
            return False, msg 
        
        # 4. Smoothing
        tg = TrackGenerator()
        smoothed = tg.smooth_track(x_m, y_m, raw_alt, ds=0.1, r=1000, is_closed=is_closed)
        
        # --- FIX 2: Return Tuple ---
        if smoothed is None: 
            return False, "Smoothing optimization failed (IPOPT error)."
        
        # 5. Interpolace Telemetrie
        # s_raw počítáme z FILTROVANÝCH dat
        s_raw = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))))
        
        def interp_on_s(data_arr):
            # data_arr musí mít stejnou délku jako s_raw!
            f = interp1d(s_raw, data_arr, kind='linear', fill_value="extrapolate")
            return f(smoothed['s'])

        final_data = {
            's': smoothed['s'],
            'x': smoothed['x'],
            'y': smoothed['y'],
            'z': smoothed['z'],
            'kappa': smoothed['kappa'],
            'psi': smoothed['psi'],
            'alpha': smoothed['alpha'],
            'speed': interp_on_s(raw_speed), 
            'power': interp_on_s(raw_power),
            'brake': interp_on_s(raw_brake),
        }
        
        np.savez(save_path, **final_data)
        
        # --- FIX 3: Return Tuple ---
        return True, "Track generated successfully."