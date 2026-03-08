# src/optimization/solvers/utils.py
import numpy as np
from src.optimization.solvers.base_solver import InitConditions

def calculate_initial_guess(car,track, init_cond: InitConditions, ds=1.0):
        #creates warm start for the solver based on simple forward-backward integration and curve speed limits
       
        total_length = track.length
        N = int(np.round(total_length / ds))
        s_sim = np.linspace(0, total_length, N + 1)
        
        ds_const = total_length / N 
        
        kappa_dm = track.get_curvature(s_sim)
        kappa = np.array(kappa_dm).flatten()
        
        m = car.weight
        g = 9.81
        rho = 1.225
        Cl = car.cl
        Cd = car.cd
        mu_x = car.mu_x - 0.1
        mu_y = car.mu_y - 0.1
        P_max = car.power_max-5000
        F_max_abs = 8000
        
    
        #curve limits vectorized
        denom = m * np.abs(kappa) - mu_y * 0.5 * rho * Cl
        
      
        v_max_car = 40.0 
        v_curve = np.full(N+1, v_max_car)
        
        
        mask = denom > 0
        v_sq_limit = (mu_y * m * g) / denom[mask]
        v_curve[mask] = np.sqrt(v_sq_limit)
        
    
        v_curve = np.minimum(v_curve, v_max_car)

        #forward integration (acceleration)
        v_fwd = np.zeros(N+1)
        v_fwd[0] = max(init_cond.v, 3.0) 
        
        for k in range(N):
            v_curr = v_fwd[k]
            
            # Aero Drag
            F_drag = 0.5 * rho * Cd * v_curr**2
            
            # Grip: F_z = m*g + AeroDown
            F_z = m*g + 0.5 * rho * Cl * v_curr**2
            F_grip = mu_x * F_z
            
            # Power: P = F*v -> F = P/v
            F_power = P_max / max(v_curr, 2.0) * car.power_eff
            
            # Useable accel force
            F_acc_max = min(F_grip, F_power, F_max_abs)
            
            # Net force & accel
            F_net = F_acc_max - F_drag
            acc = F_net / m
            
            # kinematics
            v_next = np.sqrt(max(v_curr**2 + 2 * acc * ds_const, 1.0))
            
            # limit by curve
            v_fwd[k+1] = min(v_next, v_curve[k+1])

        # backward integration (braking)
        v_bwd = np.zeros(N+1)
        v_bwd[-1] = v_curve[-1] 
        
        for k in range(N, 0, -1):
            v_curr = v_bwd[k] 
            
            # Aero
            F_drag = 0.5 * rho * Cd * v_curr**2
            F_z = m*g + 0.5 * rho * Cl * v_curr**2
            
            # Max Brake Force
            F_grip = mu_x * F_z
            F_brake_max = min(F_grip, F_max_abs) 
                    
            acc_back = (F_brake_max + F_drag) / m
            
            v_prev = np.sqrt(v_curr**2 + 2 * acc_back * ds_const)
            
            # Min (Backward, Forward, Curve)
            v_bwd[k-1] = min(v_prev, v_curve[k-1], v_fwd[k-1])

        # Final velocity profile
        v_guess = v_bwd 
        
        #smoothing for IPOPT
        window_size = 5
        v_smooth = np.convolve(v_guess, np.ones(window_size)/window_size, mode='same')
        v_smooth[0] = v_guess[0] 
        v_smooth[-1] = v_guess[-1] 
        
        #calculate forces and energy for the guess
        v_sq_diff = np.diff(v_smooth**2)
        acc_req = v_sq_diff / (2 * ds_const)
        
        v_mid = 0.5 * (v_smooth[:-1] + v_smooth[1:])
        
        # Drag
        F_drag_vec = 0.5 * rho * Cd * v_mid**2
        
        # F
        F_net_req = m * acc_req
        F_long_req = F_net_req + F_drag_vec
        
        # time
        dt_vec = ds_const / np.maximum(v_mid, 1.0)
        T_guess = np.concatenate(([0], np.cumsum(dt_vec))) + init_cond.T
        
        # Inputs splitting
        F_acc_guess = np.maximum(F_long_req, 0)
        F_brake_guess = np.minimum(F_long_req, 0)
        
        # Energy
        P_mech = F_long_req * v_mid
       
        dE = P_mech * dt_vec / 3.6e6 
        E_guess = np.concatenate(([init_cond.E], np.cumsum(dE) + init_cond.E))

        return {
            'v': np.array(v_smooth).flatten(),       
            'T': np.array(T_guess).flatten(),        
            'E': np.array(E_guess).flatten(),        
            'F_acc': np.array(F_acc_guess).flatten(),   
            'F_brake': np.array(F_brake_guess).flatten()
        }