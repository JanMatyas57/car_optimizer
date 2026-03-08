import os
import math
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from src.optimization.solvers.base_solver import BaseSolver, SolverResults, InitConditions



class Tire:
    def __init__(self, filename, Rl=0.2, rr=0.015):
        # Default fallback if the path is relative to the execution root
        if not os.path.exists(filename):
            filename = os.path.join("src", "core", "pacejka94.txt")
            
        self.data = np.loadtxt(filename, delimiter=',')
        self.Br = self.data[0]
        self.Bt = self.data[1]
        self.Bx = self.data[2]
        self.By = self.data[3]
        self.Ct = self.data[4]
        self.Cx = self.data[5]
        self.Cy = self.data[6]
        self.Dr = self.data[7]
        self.Dt = self.data[8]
        self.Dx = self.data[9]
        self.Dy = self.data[10]
        self.Et = self.data[11]
        self.Ex = self.data[12]
        self.Ey = self.data[13]
        self.mu_y_grad = self.data[14]
        self.mu_x_grad = self.data[15]
        self.Rl = Rl                 
        self.rr = rr                 

    def Fy(self, alpha, Fz):
        Dy = self.Dy - self.mu_y_grad * Fz
        Fy = Fz * Dy * np.sin(self.Cy * np.arctan(self.By * alpha - self.Ey * (self.By * alpha - np.arctan(self.By * alpha))))
        return Fy
    
    def Fx(self, kappa, Fz):
        Dx = self.Dx - self.mu_x_grad * Fz
        resistance = Fz * self.rr
        Fx = Fz * Dx * np.sin(self.Cx * np.arctan(self.Bx * kappa - self.Ex * (self.Bx * kappa - np.arctan(self.Bx * kappa)))) - resistance
        return Fx
    
    def Mz(self, alpha, Fz):
        Fy = self.Fy(alpha, Fz)
        t = self.Dt * np.cos(self.Ct * np.arctan(self.Bt * alpha - self.Et * (self.Bt * alpha - np.arctan(self.Bt * alpha))))
        Mzr = self.Dr * np.cos(np.arctan(self.Br * alpha))
        Mz = t * Fy + Mzr
        return Mz


class QS_Car:
    
    def __init__(self, tire_file, ay_max, ax_max, v_max, m, CoGz, WB, TW, wd, CdA, ClA, CoP, P_max, gear_res, M_max, i, Rl):
        self.tire = Tire(tire_file, Rl=Rl)
        self.ay_max = ay_max         
        self.ax_max = ax_max         
        self.v_max = v_max           
        self.m = m                   
        self.CoGz = CoGz             
        self.WB = WB                 
        self.TW = TW                 
        self.wd = wd                 
        self.CdA = CdA               
        self.ClA = ClA               
        self.CoP = CoP               
        self.P_max = P_max           
        self.gear_res = gear_res     
        self.M_max = M_max          
        self.i = i                   

    def Fd(self, v, rho=1.225): 
        return 0.5 * rho * self.CdA * v**2
    
    def Frr(self): 
        return self.m * 9.81 * self.tire.rr
    
    def Fgear(self, v): 
        return v * self.gear_res 
    
    def Fdown(self, v, rho=1.225): 
        return 0.5 * rho * self.ClA * v**2

    def Favailable(self, v): 
        v = max(v, 1.0) # avoid division by zero
        omega_wheel = v / self.tire.Rl 
        omega_motor = omega_wheel * self.i  
        if omega_motor * 4 * self.M_max < self.P_max * 1000: 
            return 4 * self.M_max * self.i / self.tire.Rl  
        else:
            M = (self.P_max * 1000) / (omega_motor * 4)  
            return 4 * M * self.i / self.tire.Rl       

    def N_all(self, v, ax, ay):
        Fdown = self.Fdown(v)
        mf = self.m * self.wd 
        mr = self.m * (1 - self.wd)
        N_aero_f = Fdown * self.CoP
        N_aero_r = Fdown * (1 - self.CoP)
        
        N_FL = mf * 9.81 / 2 + N_aero_f / 2 - (mf * ay * self.CoGz) / self.TW - (self.m/2 * ax * self.CoGz) / self.WB
        N_FR = mf * 9.81 / 2 + N_aero_f / 2 + (mf * ay * self.CoGz) / self.TW - (self.m/2 * ax * self.CoGz) / self.WB
        N_RL = mr * 9.81 / 2 + N_aero_r / 2 - (mr * ay * self.CoGz) / self.TW + (self.m/2 * ax * self.CoGz) / self.WB
        N_RR = mr * 9.81 / 2 + N_aero_r / 2 + (mr * ay * self.CoGz) / self.TW + (self.m/2 * ax * self.CoGz) / self.WB

        return N_FL, N_FR, N_RL, N_RR
    
    def slip_angles(self, v, delta, beta, r):
        a1 = self.WB * (1 - self.wd)  
        a2 = self.WB * self.wd          
        vy = v * np.sin(-beta)

        vx_FL = (v - r * self.TW / 2) * np.cos(delta) + (vy + r * a1) * np.sin(delta)
        vx_FR = (v + r * self.TW / 2) * np.cos(delta) + (vy - r * a1) * np.sin(delta)
        vx_RL = (v - r * self.TW / 2)
        vx_RR = (v + r * self.TW / 2)

        vy_FL = -(v - r * self.TW / 2) * np.sin(delta) + (vy + r * a1) * np.cos(delta)
        vy_FR = -(v + r * self.TW / 2) * np.sin(delta) + (vy + r * a1) * np.cos(delta)
        vy_RL = vy - r * a2
        vy_RR = vy - r * a2

        alpha_FL = -np.arctan2(vy_FL, vx_FL)
        alpha_FR = -np.arctan2(vy_FR, vx_FR)
        alpha_RL = -np.arctan2(vy_RL, vx_RL)
        alpha_RR = -np.arctan2(vy_RR, vx_RR)

        return alpha_FL, alpha_FR, alpha_RL, alpha_RR
    
    def friction_ellipse_limit(self, Fx, Fy, Fz):
        if Fz <= 0.0:
            return 0.0, 0.0
            
        mu_y = self.tire.Dy - self.tire.mu_y_grad * Fz
        mu_x = self.tire.Dx - self.tire.mu_x_grad * Fz

        mu_x = max(mu_x, 1e-6)
        mu_y = max(mu_y, 1e-6)

        Fx_max = mu_x * Fz
        Fy_max = mu_y * Fz

        g = math.sqrt((Fx / Fx_max)**2 + (Fy / Fy_max)**2)

        if g <= 1.0:
            return Fx, Fy

        s = 1.0 / g
        return Fx * s, Fy * s
    
    def tire_forces(self, v, delta, beta, r, ay, ax, kappa):
        alpha_FL, alpha_FR, alpha_RL, alpha_RR = self.slip_angles(v, delta, beta, r)
        N_FL, N_FR, N_RL, N_RR = self.N_all(v, ax, ay)
        
        Fy_FL = self.tire.Fy(alpha_FL, N_FL)
        Fy_FR = self.tire.Fy(alpha_FR, N_FR)
        Fy_RL = self.tire.Fy(alpha_RL, N_RL)
        Fy_RR = self.tire.Fy(alpha_RR, N_RR)
        Favailable = self.Favailable(v)

        Fx_FL = min(self.tire.Fx(kappa, N_FL), Favailable / 4.0) 
        Fx_FR = min(self.tire.Fx(kappa, N_FR), Favailable / 4.0) 
        Fx_RL = min(self.tire.Fx(kappa, N_RL), Favailable / 4.0) 
        Fx_RR = min(self.tire.Fx(kappa, N_RR), Favailable / 4.0) 

        Fx_FL_w, Fy_FL_w = self.friction_ellipse_limit(Fx_FL, Fy_FL, N_FL)  
        Fx_FR_w, Fy_FR_w = self.friction_ellipse_limit(Fx_FR, Fy_FR, N_FR)
        Fx_RL_w, Fy_RL_w = self.friction_ellipse_limit(Fx_RL, Fy_RL, N_RL)
        Fx_RR_w, Fy_RR_w = self.friction_ellipse_limit(Fx_RR, Fy_RR, N_RR)

        Fx_FL = Fx_FL_w * np.cos(delta) - Fy_FL_w * np.sin(delta)
        Fx_FR = Fx_FR_w * np.cos(delta) - Fy_FR_w * np.sin(delta)
        Fx_RL = Fx_RL_w
        Fx_RR = Fx_RR_w

        Fy_FL = Fx_FL_w * np.sin(delta) + Fy_FL_w * np.cos(delta)
        Fy_FR = Fx_FR_w * np.sin(delta) + Fy_FR_w * np.cos(delta)
        Fy_RL = Fy_RL_w
        Fy_RR = Fy_RR_w

        return Fx_FL, Fy_FL, Fx_FR, Fy_FR, Fx_RL, Fy_RL, Fx_RR, Fy_RR 

    def find_GGV(self, delta_max, beta_max, kappa_max, tol, N=9):
        # note: reduced N from 21 to 9 for decent calculation speed before lap
        delta = np.linspace(0, delta_max, N)  
        beta = np.linspace(0, beta_max, N)
        kappa = np.linspace(-kappa_max, kappa_max, N)  
        v = np.linspace(2, self.v_max, N)

        self.GGV_points = []
        point = []
        point_mirror = []

        print("calculating ggv diagram (in memory)...")
        for v_i in v:
            for kappa_i in kappa:
                for beta_i in beta:
                    for delta_i in delta:
                        ay_old = 0 
                        ay = 10
                        ax_old = 10
                        ax = 10
                        r_old = 0 
                        r = 0
                        i_while = 0

                        while abs(ay - ay_old) > tol:
                            ay_old = ay
                            r_old = r
                            ax_old = ax
                            i_while += 1 

                            Fx_FL, Fy_FL, Fx_FR, Fy_FR, Fx_RL, Fy_RL, Fx_RR, Fy_RR = self.tire_forces(v_i, delta_i, beta_i, r, ay, ax, kappa_i)   
                            F_drag = self.Fd(v_i)

                            Fy = Fy_FL + Fy_FR + Fy_RL + Fy_RR
                            Fx = Fx_FL + Fx_FR + Fx_RL + Fx_RR - F_drag

                            ax = Fx / self.m 
                            ay = Fy / self.m
                            r = ay / v_i

                            # numerical damping
                            r = (1 * r + 2 * r_old) / 3
                            ay = (1 * ay + 2 * ay_old) / 3
                            ax = (1 * ax + 2 * ax_old) / 3

                            if i_while > 50:
                                ay_old = (ay_old + ay) / 2
                                ay = ay_old

                        point.append([ay, ax, v_i])
                        point_mirror.append([-ay, ax, v_i])

        data1 = np.array(point)  
        data_mirrored = np.array(point_mirror)
        self.GGV_points = np.vstack((data1, data_mirrored)) 

    def create_lookup_tables(self, steps=50):
        data = self.GGV_points
        unique_v = np.unique(data[:, 2])
        final_rows = []

        for v in unique_v:
            subset = data[data[:, 2] == v, :2] 
            if len(subset) < 3: continue

            subset_mirror = subset.copy()
            subset_mirror[:, 0] *= -1 
            full_cloud = np.vstack((subset, subset_mirror))

            try:
                hull = ConvexHull(full_cloud)
                hull_pts = full_cloud[hull.vertices]
            except:
                continue

            min_ay_idx = np.argmin(hull_pts[:, 0])
            max_ay_idx = np.argmax(hull_pts[:, 0])
            
            sorted_indices = np.argsort(hull_pts[:, 0])
            sorted_pts = hull_pts[sorted_indices]

            max_ay_abs = np.max(hull_pts[:, 0])
            input_ay = np.linspace(0, max_ay_abs, steps)
            
            limit_ax_throttle = []
            limit_ax_brake = []

            for ay_val in input_ay:
                y_candidates = []
                for i in range(len(hull_pts)):
                    p1 = hull_pts[i]
                    p2 = hull_pts[(i + 1) % len(hull_pts)]
                    x1, x2 = p1[0], p2[0]
                    
                    if (ay_val >= min(x1, x2)) and (ay_val <= max(x1, x2)):
                        if x1 != x2:
                            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                            y_on_line = p1[1] + slope * (ay_val - p1[0])
                            y_candidates.append(y_on_line)
                        else:
                            y_candidates.append(max(p1[1], p2[1])) 

                if y_candidates:
                    limit_ax_throttle.append(max(y_candidates)) 
                    limit_ax_brake.append(min(y_candidates))    
                else:
                    limit_ax_throttle.append(0)
                    limit_ax_brake.append(0)

            max_ax = np.max(hull_pts[:, 1])
            min_ax = np.min(hull_pts[:, 1])
            
            input_ax_pos = np.linspace(0, max_ax, steps)
            input_ax_neg = np.linspace(0, min_ax, steps)
            
            limit_ay_throttle = [] 
            limit_ay_brake = []    

            def get_max_ay_for_ax(ax_val):
                x_candidates = []
                for i in range(len(hull_pts)):
                    p1 = hull_pts[i]
                    p2 = hull_pts[(i + 1) % len(hull_pts)]
                    y1, y2 = p1[1], p2[1]
                    if (ax_val >= min(y1, y2)) and (ax_val <= max(y1, y2)):
                        if y1 != y2:
                            slope_inv = (p2[0] - p1[0]) / (p2[1] - p1[1])
                            x_on_line = p1[0] + slope_inv * (ax_val - p1[1])
                            x_candidates.append(x_on_line)
                        else:
                            x_candidates.append(max(p1[0], p2[0]))
                if x_candidates: return np.max(np.abs(x_candidates))
                return 0.0

            for val in input_ax_pos:
                limit_ay_throttle.append(get_max_ay_for_ax(val))
                
            for val in input_ax_neg:
                limit_ay_brake.append(get_max_ay_for_ax(val))

            for i in range(steps):
                final_rows.append({
                    'v': v,
                    'input_ay': input_ay[i],
                    'input_ax_pos': input_ax_pos[i],
                    'input_ax_neg': input_ax_neg[i],
                    'limit_ax_throttle': limit_ax_throttle[i],
                    'limit_ax_brake': limit_ax_brake[i],
                    'limit_ay_throttle': limit_ay_throttle[i],
                    'limit_ay_brake': limit_ay_brake[i]
                })

        df = pd.DataFrame(final_rows)
        self.lookup_df = df
        return df


# ==========================================================
# Optimization Solver Wrapper
# ==========================================================

class QuasiSteadySolver(BaseSolver):
    def __init__(self, car, track, laps=1):
        super().__init__(car, track, laps)
        
     
        # assuming cogz = 0.3, TW (track width) = 1.2 by default if not strictly in Car
        wd = 1.0 - self.car.weight_distrib 
        cop = 1.0 - self.car.cop           
        
        self.qs_car = QS_Car(
            tire_file="src/core/pacejka94.txt",
            ay_max=25.0,
            ax_max=15.0,
            v_max=40.0,
            m=self.car.weight,
            CoGz=self.car.cogz,
            WB=self.car.wheelbase,
            TW=1.2, 
            wd=wd,
            CdA=self.car.cd,
            ClA=self.car.cl,
            CoP=cop,
            P_max=self.car.power_max / 1000.0, # kW
            gear_res=0.0,
            M_max=self.car.max_torque,
            i=self.car.gear_ratio,
            Rl=self.car.wheel_radius
        )
        
        # 2. Run GGV and build LUT in memory
        self.qs_car.find_GGV(delta_max=0.2, beta_max=0.07, kappa_max=0.15, tol=0.1, N=9)
        self.df = self.qs_car.create_lookup_tables(steps=40)
        
        # 3. Setup slices for solver
        self._setup_luts()

    def _setup_luts(self):
        self.v_levels = np.sort(self.df['v'].unique())

        self.slices_ax_pos = {}
        self.slices_ax_neg = {}
        self.slices_ay_throttle = {}
        self.slices_ay_brake = {}

        for v_i in self.v_levels:
            # Ax pos -> Ay max
            s1 = self.df[self.df['v'] == v_i].sort_values('input_ax_pos')
            ax1, inv1 = np.unique(s1['input_ax_pos'].to_numpy(), return_inverse=True)
            ay1 = np.bincount(inv1, weights=s1['limit_ay_throttle'].to_numpy()) / np.bincount(inv1)
            self.slices_ax_pos[v_i] = (ax1, ay1)

            # Ax neg -> Ay max
            s2 = self.df[self.df['v'] == v_i].sort_values('input_ax_neg')
            ax2, inv2 = np.unique(s2['input_ax_neg'].to_numpy(), return_inverse=True)
            ay2 = np.bincount(inv2, weights=s2['limit_ay_brake'].to_numpy()) / np.bincount(inv2)
            self.slices_ax_neg[v_i] = (ax2, ay2)

            # Ay -> Ax throttle
            s3 = self.df[self.df['v'] == v_i].sort_values('input_ay')
            ax3, inv3 = np.unique(s3['input_ay'].to_numpy(), return_inverse=True)
            ay3 = np.bincount(inv3, weights=s3['limit_ax_throttle'].to_numpy()) / np.bincount(inv3)
            self.slices_ay_throttle[v_i] = (ax3, ay3)

            # Ay -> Ax brake
            s4 = self.df[self.df['v'] == v_i].sort_values('input_ay')
            ax4, inv4 = np.unique(s4['input_ay'].to_numpy(), return_inverse=True)
            ay4 = np.bincount(inv4, weights=s4['limit_ax_brake'].to_numpy()) / np.bincount(inv4)
            self.slices_ay_brake[v_i] = (ax4, ay4)

    def _query_lut(self, slices, accel_input, v_input):
        vq = float(v_input)
        if vq <= self.v_levels[0]:
            v0 = v1 = self.v_levels[0]
        elif vq >= self.v_levels[-1]:
            v0 = v1 = self.v_levels[-1]
        else:
            j = np.searchsorted(self.v_levels, vq)
            v0 = self.v_levels[j-1]
            v1 = self.v_levels[j]

        def interp_on_v(v_i, axq):
            ax, ay = slices[v_i]
            axq_c = np.clip(axq, ax.min(), ax.max())
            return np.interp(axq_c, ax, ay)

        y0 = interp_on_v(v0, accel_input)
        y1 = interp_on_v(v1, accel_input)

        if v0 == v1: return float(y0)
        t = (vq - v0) / (v1 - v0)
        return float(y0 + t * (y1 - y0))

    def _find_v_ay_lim(self, radius_array, tol=0.1):
        v_test = np.ones(len(radius_array)) * 2.0
        ay_lim = np.zeros(len(radius_array))

        for i, R in enumerate(radius_array):
            if R > 5000:
                v_test[i] = self.qs_car.v_max
                ay_lim[i] = 0
                continue

            while True:
                ay_max = self._query_lut(self.slices_ax_pos, 0, v_test[i])
                ay_test = (v_test[i]**2) / R

                if ay_test > (ay_max - tol):
                    break

                v_test[i] += np.abs(ay_max - ay_test) * 0.5
                v_test[i] = min(v_test[i], self.qs_car.v_max)
                
                if v_test[i] >= self.qs_car.v_max:
                    break

            ay_lim[i] = (v_test[i]**2) / R

        return v_test, ay_lim

    def solve(self, init_cond: InitConditions, init_guess=None, ds=0.5) -> SolverResults:
        # 1. evaluate track geometry
        total_length = self.track.length
        N = int(np.round(total_length / ds))
        s_eval = np.linspace(0, total_length, N + 1)
        
        kappa = np.array(self.track.get_curvature(s_eval)).flatten()
        radius = np.zeros_like(kappa)
        for i in range(len(kappa)):
            if abs(kappa[i]) < 1e-4:
                radius[i] = 10000.0
            else:
                radius[i] = 1.0 / abs(kappa[i])
        
        # 2. apply limit finding logic from colleague's runLapTime.py
        v_limit, ay_lim = self._find_v_ay_lim(radius, 0.1)
        
        n_points = len(v_limit)
        v_fwd = np.zeros(n_points)
        v_bwd = np.zeros(n_points)
        
        v_fwd[0] = max(init_cond.v, 1.0)
        v_bwd[-1] = v_limit[-1]

        # forward pass
        for i in range(n_points - 1):
            ay = min((v_fwd[i]**2) / radius[i], ay_lim[i])
            ax_max = self._query_lut(self.slices_ay_throttle, ay, v_fwd[i])
            v_next = np.sqrt(max(v_fwd[i]**2 + 2 * ax_max * ds, 0))
            v_fwd[i+1] = min(v_next, v_limit[i+1])

        # backward pass
        for i in range(n_points - 1, 0, -1):
            ay = min((v_bwd[i]**2) / radius[i], ay_lim[i])
            ax_min = np.abs(self._query_lut(self.slices_ay_brake, ay, v_bwd[i]))
            v_prev = np.sqrt(max(v_bwd[i]**2 + 2 * ax_min * ds, 0)) 
            v_bwd[i-1] = min(v_prev, v_limit[i-1])

        v_final = np.minimum(v_fwd, v_bwd)
        
        # 3. post process arrays to match user's expected format
        v_avg = np.maximum(1e-6, 0.5 * (v_final[:-1] + v_final[1:]))
        dt_array = ds / v_avg
        t_final = np.concatenate(([init_cond.T], init_cond.T + np.cumsum(dt_array)))

        acc_long = np.diff(v_final**2) / (2 * ds)
        acc_long = np.append(acc_long, 0)
        
        f_net = self.car.weight * acc_long
        f_drag = np.array([self.car.get_aero_drag(v) for v in v_final]).flatten()
        f_long_total = f_net + f_drag
        
        f_acc = np.maximum(f_long_total, 0)
        f_brake = np.minimum(f_long_total, 0)
        
        empty_arr = np.zeros_like(v_final)

        return SolverResults(
            success=True,
            lap_time=t_final[-1] - init_cond.T,
            t=t_final,
            v=v_final,
            s=s_eval,
            f_acc=f_acc,
            f_brake=f_brake,
            E=empty_arr,
            SoC=empty_arr,
            recuperation=empty_arr,
            drs_usage=empty_arr,
            sens={}
        )