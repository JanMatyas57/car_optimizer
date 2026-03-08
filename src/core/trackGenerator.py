import numpy as np
import casadi as ca
from scipy.interpolate import PchipInterpolator, interp1d

def latlon_to_meters(lat, lon):

    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat0, lon0 = lat_rad[0], lon_rad[0]
    R = 6371000 
    x = R * (lon_rad - lon0) * np.cos(lat0)
    y = R * (lat_rad - lat0)
    return x, y

class TrackGenerator:
    @staticmethod
    def smooth_track(x_smpl, y_smpl, z_smpl, ds=0.1, w_pos=50.0, r=3000, is_closed=True):
        """smooth the track with rk4 and casadi."""
        x_smpl, y_smpl, z_smpl = np.array(x_smpl).flatten(), np.array(y_smpl).flatten(), np.array(z_smpl).flatten()
        dx, dy = np.diff(x_smpl), np.diff(y_smpl)
        s_tmp = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy**2))))
        
        total_length = s_tmp[-1]
        N = max(2, int(np.round(total_length / ds)))
        ds_real = total_length / N
        s_traj = np.linspace(0, total_length, N) 
        
        fx, fy, fz = PchipInterpolator(s_tmp, x_smpl), PchipInterpolator(s_tmp, y_smpl), PchipInterpolator(s_tmp, z_smpl)
        x_init, y_init, z_init = fx(s_traj), fy(s_traj), fz(s_traj)
        
        dz_init = np.diff(z_init) / ds_real
        alpha_init = np.append(np.arcsin(np.clip(dz_init, -0.2, 0.2)), 0) 
        alpha_init[-1] = alpha_init[-2]

        th_init = np.unwrap(np.arctan2(np.diff(y_init), np.diff(x_init)))
        th_init = np.append(th_init, th_init[-1])
        C_init = np.append(np.diff(th_init) / ds_real, 0)
        
        opti = ca.Opti()
        Z = opti.variable(N, 6) 
        z_C, z_th, z_x, z_y, z_alp, z_z = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4], Z[:,5]
        u = opti.variable(N-1, 2) 
        
        x_dev = w_pos*((z_x[:-1] - x_init[:-1])**2 + (z_x[1:] - x_init[1:])**2) / 2
        y_dev = w_pos*((z_y[:-1] - y_init[:-1])**2 + (z_y[1:] - y_init[1:])**2) / 2
        z_dev = w_pos*((z_z[:-1] - z_init[:-1])**2 + (z_z[1:] - z_init[1:])**2) / 2
        reg_term = r * u[:,0]**2 + 10*r * u[:,1]**2
        
        opti.minimize(ca.sum1(ds_real * (reg_term + x_dev + y_dev + z_dev)))
        
        def f_dyn(z_in, u_in):
            return ca.vertcat(u_in[0], z_in[0], ca.cos(z_in[1]), ca.sin(z_in[1]), u_in[1], ca.sin(z_in[4]))

        for k in range(N-1):
            z_k, u_k = Z[k, :].T, u[k, :].T
            k1 = f_dyn(z_k, u_k)
            k2 = f_dyn(z_k + ds_real/2*k1, u_k)
            k3 = f_dyn(z_k + ds_real/2*k2, u_k)
            k4 = f_dyn(z_k + ds_real*k3, u_k)
            opti.subject_to(Z[k+1, :].T == z_k + ds_real/6 * (k1 + 2*k2 + 2*k3 + k4))

        if is_closed:
            opti.subject_to(Z[-1, 2] == Z[0, 2]) 
            opti.subject_to(Z[-1, 3] == Z[0, 3]) 
            opti.subject_to(Z[-1, 5] == Z[0, 5]) 
            opti.subject_to(Z[-1, 0] == Z[0, 0]) 
            opti.subject_to(Z[-1, 4] == Z[0, 4]) 
            n_loops = np.round((th_init[-1] - th_init[0]) / (2*np.pi))
            opti.subject_to(z_th[-1] == z_th[0] + 2*np.pi*n_loops)

        opti.set_initial(z_x, x_init); opti.set_initial(z_y, y_init); opti.set_initial(z_z, z_init)
        opti.set_initial(z_th, th_init); opti.set_initial(z_C, C_init); opti.set_initial(z_alp, alpha_init)
        
        opti.solver('ipopt', {'expand': True}, {'max_iter': 500, 'print_level': 0, 'sb': 'yes'})
        
        try:
            sol = opti.solve()
            Z_opt = sol.value(Z)
            return {'s': s_traj, 'kappa': Z_opt[:, 0], 'psi': Z_opt[:, 1], 'x': Z_opt[:, 2], 'y': Z_opt[:, 3], 'alpha': Z_opt[:, 4], 'z': Z_opt[:, 5]}
        except Exception as e:
            print(f"Optimalizace selhala: {e}")
            return None



def generate_smoothed_track(all_data, lap_infos, is_closed, r_penalty, w_penalty):
    """processes raw data, smooths via casadi and averages if multiple laps are provided."""
    tg = TrackGenerator()
    smoothed_list, telemetry_list = [], []

    print(f"processing {len(lap_infos)} laps with r={r_penalty}, w={w_penalty}...")

    for lap in lap_infos:
        s_idx, e_idx = lap['indices']
        raw_lat, raw_lon = all_data['Latitude'][s_idx:e_idx], all_data['Longitude'][s_idx:e_idx]
        raw_power, raw_brake = all_data['Power'][s_idx:e_idx], all_data['Brake'][s_idx:e_idx]
        raw_speed, raw_alt = all_data['Speed'][s_idx:e_idx], all_data['Altitude'][s_idx:e_idx]

        x_m, y_m = latlon_to_meters(raw_lat, raw_lon)
        # pass the w_penalty as w_pos
        sm = tg.smooth_track(x_m, y_m, raw_alt, ds=0.1, w_pos=w_penalty, r=r_penalty, is_closed=is_closed)
        if sm is None: raise RuntimeError("smoothing failed for one of the laps.")

        s_raw = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))))
        _, unique_indices = np.unique(s_raw, return_index=True)
        s_raw = s_raw[unique_indices]
        
        from scipy.interpolate import interp1d
        f_power = interp1d(s_raw, raw_power[unique_indices], kind='linear', fill_value='extrapolate')
        f_brake = interp1d(s_raw, raw_brake[unique_indices], kind='linear', fill_value='extrapolate')
        f_speed = interp1d(s_raw, raw_speed[unique_indices], kind='linear', fill_value='extrapolate')
        f_alt = interp1d(s_raw, raw_alt[unique_indices], kind='linear', fill_value='extrapolate')

        smoothed_list.append(sm)
        telemetry_list.append({'power': f_power, 'brake': f_brake, 'speed': f_speed, 'alt': f_alt})

    if len(smoothed_list) == 1:
        res = smoothed_list[0]
        fns = telemetry_list[0]
        final_s, final_kappa, final_psi, final_x, final_y, final_alpha = res['s'], res['kappa'], res['psi'], res['x'], res['y'], res['alpha']
        final_z, final_power, final_brake, final_speed = fns['alt'](final_s), fns['power'](final_s), fns['brake'](final_s), fns['speed'](final_s)
        avg_lap_time = lap_infos[0]['time']
    else:
        min_len = min([s_arr['s'][-1] for s_arr in smoothed_list])
        N_common = min(len(s_arr['s']) for s_arr in smoothed_list)
        final_s = np.linspace(0, min_len, N_common)

        vals = {k: [] for k in ['x', 'y', 'z', 'kappa', 'alpha', 'psi_sin', 'psi_cos', 'p', 'b', 'v']}
        for sm, fns in zip(smoothed_list, telemetry_list):
            for k in ['x', 'y', 'kappa', 'alpha']: vals[k].append(interp1d(sm['s'], sm[k])(final_s))
            vals['z'].append(fns['alt'](final_s))
            psi_curr = interp1d(sm['s'], sm['psi'])(final_s)
            vals['psi_sin'].append(np.sin(psi_curr))
            vals['psi_cos'].append(np.cos(psi_curr))
            vals['p'].append(fns['power'](final_s)); vals['b'].append(fns['brake'](final_s)); vals['v'].append(fns['speed'](final_s))

        final_x, final_y, final_z = np.mean(vals['x'], axis=0), np.mean(vals['y'], axis=0), np.mean(vals['z'], axis=0)
        final_kappa, final_alpha = np.mean(vals['kappa'], axis=0), np.mean(vals['alpha'], axis=0)
        final_psi = np.arctan2(np.mean(vals['psi_sin'], axis=0), np.mean(vals['psi_cos'], axis=0))
        final_power, final_brake, final_speed = np.mean(vals['p'], axis=0), np.mean(vals['b'], axis=0), np.mean(vals['v'], axis=0)
        avg_lap_time = float(np.mean([l['time'] for l in lap_infos]))

    return {
        's': final_s, 'kappa': final_kappa, 'psi': final_psi, 'x': final_x, 'y': final_y, 'z': final_z,
        'alpha': final_alpha, 'power': final_power, 'brake': final_brake, 'speed': final_speed, 'lap_time': avg_lap_time
    }