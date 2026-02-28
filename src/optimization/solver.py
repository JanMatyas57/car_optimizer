import casadi as ca
import numpy as np
from src.core.car import Car
from dataclasses import dataclass, field
from matplotlib import pyplot as plt

@dataclass
class SolverResults:
    #mandatory
    success: bool
    lap_time: float
    t: np.ndarray
    v: np.ndarray #total velocity
    s: np.ndarray
    #inputs 
    f_acc: np.ndarray
    f_brake: np.ndarray
    recuperation:np.ndarray
    #model specific
    vx: np.ndarray      = field(default_factory=lambda: np.array([]))
    vy: np.ndarray      = field(default_factory=lambda: np.array([]))
    n: np.ndarray         = field(default_factory=lambda: np.array([]))  # Lateral deviation
    r: np.ndarray         = field(default_factory=lambda: np.array([]))
    xi: np.ndarray       = field(default_factory=lambda: np.array([]))  # Heading error
    kappa_car: np.ndarray = field(default_factory=lambda: np.array([]))  # Actual curvature
    delta: np.ndarray     = field(default_factory=lambda: np.array([]))  # Steering angle
    E: np.ndarray         = field(default_factory=lambda: np.array([]))  
    drs_usage: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_f: np.ndarray = field(default_factory=lambda: np.array([]))  # Slip angle front
    alpha_r: np.ndarray = field(default_factory=lambda: np.array([]))  # Slip angle rear
    SoC: np.ndarray = field(default_factory=lambda: np.array([]))

    # Sensitivities (s/unit)
    sens: dict = field(default_factory=dict)
    #
    mu_x: float = 0.0
    mu_y: float = 0.0
    regen_eff: float = 0.0
    ap_id: int = 0
    

@dataclass
class InitConditions:
    v: float = 0.0  
    T: float = 0.0   
    E: float = 0.0 
    n: float = 0.0
    xi: float = 0.0
    E_target: float = 0.0
    target_time: float = None


class Scales:
    m = 280       # Hmotnost (kg)
    v = 20.0         # Rychlost (m/s)
    F = 6000      # Síla (N) - ideálně m*g
    E = 3.6e6        # Energie (J) - např. 1 kWh, nebo kapacita baterie
    T = 60.0        # Čas (s)
    L = 1000.0       # Délka (m)    

class MassPointSolver:
    def __init__(self,car,track,rk4=False,laps=1):

        self.car = car
        self.track = track
        self.laps = laps
        self.rk4 = rk4
  

    def solveLap(self, init_cond: InitConditions, init_guess=None, ds=0.5):

        opti = ca.Opti()

        # =========================
        # Parameters
        # =========================
        p_mass = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_mu_x = opti.parameter()
        p_mu_y = opti.parameter()

        # Set params values
        opti.set_value(p_mass, self.car.mass)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_mu_x, self.car.mu_x)
        opti.set_value(p_mu_y, self.car.mu_y)

        # =========================
        # Track data
        # =========================
        total_length = self.track.length
        N = int(np.round(total_length / ds))

        s_sim = np.linspace(0, total_length, N + 1)
        ds = s_sim[1:] - s_sim[:-1]

        kappa_grid = self.track.get_curvature(s_sim)

        # =========================
        # States
        # =========================
        v = opti.variable(N + 1)   # speed [m/s]
        T = opti.variable(N + 1)   # time [s]
        E = opti.variable(N + 1)   # energy used [kWh]
        
        # =========================
        # Inputs
        # =========================
        F_acc   = opti.variable(N)   # Gas [N]
        F_brake_mech = opti.variable(N)
        F_brake_regen = opti.variable(N)
        F_brake = F_brake_mech + F_brake_regen
        drs     = opti.variable(N)

        
        # =========================
        # DRS
        # =========================
        if self.car.drs_allowed and self.track.drs is not None:
            drs_flag = self.track.get_drs_flag(s_sim[:-1])
            drs_ef = drs * drs_flag
            cl_eff = p_cl + drs_ef * (self.car.cl_drs - p_cl)
            cd_eff = p_cd + drs_ef * (self.car.cd_drs - p_cd)
        else:
            drs_ef = ca.DM.zeros(N)
            cl_eff = p_cl
            cd_eff = p_cd

        opti.subject_to(drs >= 0)
        opti.subject_to(drs <= 1)

        # Current speed (k)
        v_k = v[:-1]

        # =========================
        # Aero
        # =========================
        F_drag = Car.get_aero_drag(self.car, v_k, cd=cd_eff)

        F_max_x, F_max_y = Car.get_friction_limit(
            self.car, v_k,
            mass=p_mass,
            mu_x=p_mu_x,
            mu_y=p_mu_y,
            cl=cl_eff
        )

        # Lateral Force: F_lat = m * v^2 * curvature
        F_lat = p_mass * v_k**2 * ca.fabs(kappa_grid[:-1])

        # Longitudinal force
        F_long_k = F_acc + F_brake

        # =========================
        # Pedal constraints
        # =========================
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake_regen <= 0)
        opti.subject_to(F_brake_mech <= 0)

        # =========================
        # Friction ellipse
        # =========================
        term_long = (F_long_k / F_max_x)**2
        term_lat  = (F_lat    / F_max_y)**2
        opti.subject_to(term_long + term_lat <= 1)

        #SoC
        E_left = self.car.acp_capacity - E[:-1]
        SoC = E_left/self.car.acp_capacity

        # =========================
        # Dynamics
        # =========================
        F_net = F_long_k - F_drag
        acc = F_net / p_mass

        # Speed integration: v^2 = v0^2 + 2*a*ds
        v_next_sq = v_k**2 + 2 * acc * ds
        opti.subject_to(v[1:]**2 == v_next_sq)

        # Time integration: T = T0 + ds/v
        opti.subject_to(T[1:] == T[:-1] + ds / v_k)

        # Recuperation limit at given SoC
        recuperation_max = self.car.get_recuperation_limit(SoC)

        P_recuperated = -F_brake_regen*v_k

        opti.subject_to(P_recuperated/recuperation_max <= 1)
        opti.subject_to((-F_brake_mech*v_k)/self.car.brake_max <= 1)
        opti.subject_to(F_brake/self.car.brake_max <= 1)
        
        #brakeP = -F_brake * v_k
        #P_recuperated = ca.fmin(brakeP, recuperation_max)

        energy_consumed = (F_acc * ds) / self.car.power_eff
        
        energy_saved = P_recuperated * self.car.regen_eff * (ds / v_k) * 0.94 #0.94 is fixed regen factor extracted from data
       
        dE_step = energy_consumed - energy_saved

        # Integrace do kWh
        opti.subject_to(E[1:] == E[:-1] + dE_step / 3.6e6)

        # =========================
        # Torque limit (30 Nm per wheel)
        # =========================
        opti.subject_to(
            ((self.car.wheel_radius * F_acc)) /
            (self.car.max_torque * self.car.gear_ratio * 4) < 1
        )

        # =========================
        # Power limit
        # =========================

        P_out = (F_acc * v_k) / self.car.power_eff

        # Limit from SoC (dynamic)
        P_max_dynamic = self.car.soc_to_power_interp(SoC)
        opti.subject_to(P_out <= P_max_dynamic)

        # Hard limit
        opti.subject_to(P_out <= self.car.power_max)

        # Bounding SoC
        opti.subject_to(SoC >= 0.0)

        # brake limit
        #opti.subject_to((brakeP * self.car.regen_eff) / self.car.brake_max <= 1)

        # Speed limits
        opti.subject_to(v >= 1.0)  # avoid division by zero

        # =========================
        # Initial conditions
        # =========================
        opti.subject_to(v[0] == init_cond.v)
        opti.subject_to(T[0] == init_cond.T)
        opti.subject_to(E[0] == init_cond.E)
        
        opti.subject_to(drs[0] == 0)  # start with DRS closed

        # =========================
        # Energy limitation
        # =========================
   
        if self.car.energy_limit is not None:
            opti.subject_to(E[-1]-init_cond.E <= self.car.energy_limit)

        # =========================
        # Objective function
        # =========================
        J_Time = T[-1]

        F_norm_factor = (self.car.mass * self.car.g)**2
        J_overlap = 0.004 * ca.sum1((F_acc * -F_brake) / F_norm_factor)

        J_final_v = 0.1 * v[-1]

        J_drs_binary = 0.002 * ca.sum1(drs * (1 - drs))

        opti.minimize(J_Time + J_overlap - J_final_v)

        # =========================
        # Initial guess
        # =========================
        opti.set_initial(v, 15.0)
        opti.set_initial(F_acc, 100.0)

        # =========================
        # Solver setup
        # =========================
        p_opts = {"expand": True}
        s_opts = {"max_iter": 5600, "tol": 1e-2, "print_level": 5}

        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()

            # =========================
            # Sensitivity analysis
            # =========================
            L = opti.f + ca.dot(opti.lam_g, opti.g)

            sens_m   = sol.value(ca.jacobian(L, p_mass))
            sens_cl  = sol.value(ca.jacobian(L, p_cl))
            sens_cd  = sol.value(ca.jacobian(L, p_cd))
            sens_mux = sol.value(ca.jacobian(L, p_mu_x))
            sens_muy = sol.value(ca.jacobian(L, p_mu_y))

            print("sensitivity results")
            print(f"Mass: {sens_m:+.4f} s/kg")
            print(f"CL:   {sens_cl:+.4f} s/unit")
            print(f"CdA:  {sens_cd:+.4f} s/unit")

            return SolverResults(
                success=True,
                lap_time=sol.value(T[-1]),
                t=sol.value(T),
                v=sol.value(v),
                s=s_sim,
                E=sol.value(E),
                SoC=sol.value(SoC)*100,
                f_acc=sol.value(F_acc) / self.car.power_eff,
                f_brake=sol.value(F_brake),
                recuperation=sol.value(P_recuperated*self.car.regen_eff),
                drs_usage=sol.value(drs_ef),
                sens={
                    "dTime_dm": sens_m,
                    "dTime_dCL": sens_cl,
                    "dTime_dCdA": sens_cd,
                    "dTime_dMuX": sens_mux,
                    "dTime_dMuY": sens_muy
                }
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return SolverResults(False, 0, [], [], [], [], [], [], {})
        
    def solveLap_rk4(self, init_cond: InitConditions, init_guess=None, ds=0.5):

        opti = ca.Opti()

        # =========================
        # Parameters
        # =========================
        p_mass = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_mu_x = opti.parameter()
        p_mu_y = opti.parameter()

        # Set params values
        opti.set_value(p_mass, self.car.mass)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_mu_x, self.car.mu_x)
        opti.set_value(p_mu_y, self.car.mu_y)

        # =========================
        # Track data
        # =========================
        total_length = self.track.length
        N = int(np.round(total_length / ds))

        s_sim = np.linspace(0, total_length, N + 1)
        ds = s_sim[1:] - s_sim[:-1]

        kappa_grid = self.track.get_curvature(s_sim)

        # =========================
        # States
        # =========================
        v = opti.variable(N + 1)   # speed [m/s]
        T = opti.variable(N + 1)   # time [s]
        E = opti.variable(N + 1)   # energy used [kWh]
        
        # =========================
        # Inputs
        # =========================
        F_acc   = opti.variable(N)   # Gas [N]
        F_brake_mech = opti.variable(N)
        F_brake_regen = opti.variable(N)
        F_brake = F_brake_mech + F_brake_regen
        drs     = opti.variable(N)

        
        # =========================
        # DRS
        # =========================
        if self.car.drs_allowed and self.track.drs is not None:
            drs_flag = self.track.get_drs_flag(s_sim[:-1])
            drs_ef = drs * drs_flag
            cl_eff = p_cl + drs_ef * (self.car.cl_drs - p_cl)
            cd_eff = p_cd + drs_ef * (self.car.cd_drs - p_cd)
        else:
            drs_ef = ca.DM.zeros(N)
            cl_eff = p_cl
            cd_eff = p_cd

        opti.subject_to(drs >= 0)
        opti.subject_to(drs <= 1)

        # Current speed (k)
        v_k = v[:-1]

        # =========================
        # Aero
        # =========================
        F_drag = Car.get_aero_drag(self.car, v_k, cd=cd_eff)

        F_max_x, F_max_y = Car.get_friction_limit(
            self.car, v_k,
            mass=p_mass,
            mu_x=p_mu_x,
            mu_y=p_mu_y,
            cl=cl_eff
        )

        # Lateral Force: F_lat = m * v^2 * curvature
        F_lat = p_mass * v_k**2 * ca.fabs(kappa_grid[:-1])

        # Longitudinal force
        F_long_k = F_acc + F_brake

        # =========================
        # Pedal constraints
        # =========================
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake_regen <= 0)
        opti.subject_to(F_brake_mech <= 0)

        # =========================
        # Friction ellipse
        # =========================
        term_long = (F_long_k / F_max_x)**2
        term_lat  = (F_lat    / F_max_y)**2
        opti.subject_to(term_long + term_lat <= 1)

        #SoC
        E_left = self.car.acp_capacity - E[:-1]
        SoC = E_left/self.car.acp_capacity

        # Recuperation limit at given SoC
        recuperation_max = self.car.get_recuperation_limit(SoC)

        P_recuperated = -F_brake_regen*v_k

        opti.subject_to(P_recuperated/recuperation_max <= 1)
        opti.subject_to((-F_brake_mech*v_k)/self.car.brake_max <= 1)
        opti.subject_to(F_brake/self.car.brake_max <= 1)

        # =========================
        # Dynamics (RK4 Integration)
        # =========================
        # Pomocná funkce pro získání derivací stavů podle dráhy 's'
        def get_derivatives(v_val):
            # Odpor se průběžně mění s rychlostí uvnitř RK4 sub-kroků
            f_drag_step = Car.get_aero_drag(self.car, v_val, cd=cd_eff)
            f_net_step = F_long_k - f_drag_step
            acc_step = f_net_step / p_mass
            
            dv_ds = acc_step / v_val
            dT_ds = 1.0 / v_val
            
            # Změna energie dE/ds na 1 metr (nezávisí na v, rychlost z výkonu se tu zkrátí)
            dE_ds = (F_acc / self.car.power_eff - (-F_brake_regen * self.car.regen_eff * 0.9)) / 3.6e6
            
            return dv_ds, dT_ds, dE_ds

        # Výpočet koeficientů RK4 pro každý stav naráz (vektorizovaně přes CasADi graf)
        k1_v, k1_T, k1_E = get_derivatives(v_k)
        k2_v, k2_T, k2_E = get_derivatives(v_k + 0.5 * ds * k1_v)
        k3_v, k3_T, k3_E = get_derivatives(v_k + 0.5 * ds * k2_v)
        k4_v, k4_T, k4_E = get_derivatives(v_k + ds * k3_v)

        # Přechod na další krok
        v_next = v_k + (ds / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        T_next = T[:-1] + (ds / 6.0) * (k1_T + 2*k2_T + 2*k3_T + k4_T)
        E_next = E[:-1] + (ds / 6.0) * (k1_E + 2*k2_E + 2*k3_E + k4_E)

        opti.subject_to(v[1:] == v_next)
        opti.subject_to(T[1:] == T_next)
        opti.subject_to(E[1:] == E_next)

        # =========================
        # Torque limit (30 Nm per wheel)
        # =========================
        opti.subject_to(
            ((self.car.wheel_radius * F_acc)) /
            (self.car.max_torque * self.car.gear_ratio * 4) < 1
        )

        # =========================
        # Power limit
        # =========================

        P_out = (F_acc * v_k) / self.car.power_eff

        # Limit from SoC (dynamic)
        P_max_dynamic = self.car.soc_to_power_interp(SoC)
        opti.subject_to(P_out <= P_max_dynamic)

        # Hard limit
        opti.subject_to(P_out <= self.car.power_max)

        # Bounding SoC
        opti.subject_to(SoC >= 0.0)

        # Speed limits
        opti.subject_to(v >= 1.0)  # avoid division by zero

        # =========================
        # Initial conditions
        # =========================
        opti.subject_to(v[0] == init_cond.v)
        opti.subject_to(T[0] == init_cond.T)
        opti.subject_to(E[0] == init_cond.E)
        
        opti.subject_to(drs[0] == 0)  # start with DRS closed

        # =========================
        # Energy limitation
        # =========================
   
        if self.car.energy_limit is not None:
            opti.subject_to(E[-1]-init_cond.E <= self.car.energy_limit)

        # =========================
        # Objective function
        # =========================
        J_Time = T[-1]

        F_norm_factor = (self.car.mass * self.car.g)**2
        J_overlap = 0.004 * ca.sum1((F_acc * -F_brake) / F_norm_factor)

        J_final_v = 0.1 * v[-1]

        J_drs_binary = 0.002 * ca.sum1(drs * (1 - drs))

        opti.minimize(J_Time + J_overlap - J_final_v)

        # =========================
        # Initial guess
        # =========================
        if init_guess is not None:
             # Doporučuju časem využívat předaný init_guess, zatím tu máš hardcoded 15.0
             pass
             
        opti.set_initial(v, 15.0)
        opti.set_initial(F_acc, 100.0)

        # =========================
        # Solver setup
        # =========================
        p_opts = {"expand": True}
        s_opts = {"max_iter": 5600, "tol": 1e-2, "print_level": 5}

        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()

            # =========================
            # Sensitivity analysis
            # =========================
            L = opti.f + ca.dot(opti.lam_g, opti.g)

            sens_m   = sol.value(ca.jacobian(L, p_mass))
            sens_cl  = sol.value(ca.jacobian(L, p_cl))
            sens_cd  = sol.value(ca.jacobian(L, p_cd))
            sens_mux = sol.value(ca.jacobian(L, p_mu_x))
            sens_muy = sol.value(ca.jacobian(L, p_mu_y))

            print("sensitivity results")
            print(f"Mass: {sens_m:+.4f} s/kg")
            print(f"CL:   {sens_cl:+.4f} s/unit")
            print(f"CdA:  {sens_cd:+.4f} s/unit")

            return SolverResults(
                success=True,
                lap_time=sol.value(T[-1]),
                t=sol.value(T),
                v=sol.value(v),
                s=s_sim,
                E=sol.value(E),
                SoC=sol.value(SoC)*100,
                f_acc=sol.value(F_acc) / self.car.power_eff,
                f_brake=sol.value(F_brake),
                recuperation=sol.value(P_recuperated*self.car.regen_eff),
                drs_usage=sol.value(drs_ef),
                sens={
                    "dTime_dm": sens_m,
                    "dTime_dCL": sens_cl,
                    "dTime_dCdA": sens_cd,
                    "dTime_dMuX": sens_mux,
                    "dTime_dMuY": sens_muy
                }
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return SolverResults(False, 0, [], [], [], [], [], [], {})
        

class SingleTrackSolver:
    def __init__(self, car, track, laps=1):
        self.car = car
        self.track = track
        self.laps = laps

    def solveLap(self, init_cond: InitConditions,init_guess=None, ds=1.0):
        opti = ca.Opti()

        # =========================
        # Parameters
        # =========================
        p_mass = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_mu_x = opti.parameter()
        p_mu_y = opti.parameter()

        # Set params values
        opti.set_value(p_mass, self.car.mass)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_mu_x, self.car.mu_x)
        opti.set_value(p_mu_y, self.car.mu_y)

        # =========================
        # Track data & Grid
        # =========================
        total_length = self.track.length
        N = int(np.round(total_length / ds))
        
        s_sim = np.linspace(0, total_length, N + 1)
        ds_val = s_sim[1] - s_sim[0] # Actual uniform step
        
        # Curvature along the path
        kappa_grid = self.track.get_curvature(s_sim[:-1]) # Curvature at steps 0..N-1

        # =========================
        # Variables (States at k=0..N)
        # =========================
        # Note: We use N+1 points for states, N for controls
        vx  = opti.variable(N + 1) # Longitudinal speed
        vy  = opti.variable(N + 1) # Lateral speed
        r   = opti.variable(N + 1) # Yaw rate
        n   = opti.variable(N + 1) # Lateral deviation
        xi  = opti.variable(N + 1) # Heading error (psi - psi_track)
        t   = opti.variable(N + 1) # Time
        E   = opti.variable(N + 1) # Energy

        # =========================
        # Controls (at k=0..N-1)
        # =========================
        delta   = opti.variable(N) # Steering angle
        F_acc   = opti.variable(N) # Drive force (at wheels)
        F_brake = opti.variable(N) # Brake force (at wheels)
        drs     = opti.variable(N) # DRS State

        # =========================
        # Helper / Constants
        # =========================
        # Track limits (Tunnel)
        track_width_limit = 0.5 
        opti.subject_to(opti.bounded(-track_width_limit, n, track_width_limit))

        # Steering limits
        opti.subject_to(opti.bounded(-self.car.steeringAngle_max, delta, self.car.steeringAngle_max))

        # Speed limit (avoid singularity at 0)
        opti.subject_to(vx >= 1.0) 

        # Pedal constraints
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake <= 0)

        # DRS Logic
        use_drs_vec = False # Flag to control indexing inside loop

        if self.car.drs_allowed and self.track.drs is not None:
            drs_flag = self.track.get_drs_flag(s_sim[:-1])
            drs_ef = drs * drs_flag
            
            cl_eff = p_cl + drs_ef * (self.car.cl_drs - p_cl)
            cd_eff = p_cd + drs_ef * (self.car.cd_drs - p_cd)
            use_drs_vec = True
        else:
            drs_ef = ca.DM.zeros(N)
            cl_eff = p_cl # Scalar parameter
            cd_eff = p_cd # Scalar parameter
            opti.subject_to(drs == 0)
            use_drs_vec = False #

        opti.subject_to(drs >= 0)
        opti.subject_to(drs <= 1)

        # =========================
        # DYNAMICS LOOP (Euler Integration)
        # =========================
        
        # Slip limit for linear model validity (5 degrees in rad)
        alpha_limit = 5.0 * (np.pi / 180.0)

        for k in range(N):
            # Current states
            vx_k = vx[k]
            vy_k = vy[k]
            r_k  = r[k]
            n_k  = n[k]
            xi_k = xi[k]
            t_k  = t[k]
            
            # Current inputs
            delta_k   = delta[k]
            F_acc_k   = F_acc[k]
            F_brake_k = F_brake[k]
            
            curv_k = kappa_grid[k]

            # 1. Force distribution
            
            Ff_acc = self.car.power_dist * F_acc_k
            Fr_acc = (1.0 - self.car.power_dist) * F_acc_k
            
            Ff_brk = self.car.brake_bias * F_brake_k
            Fr_brk = (1.0 - self.car.brake_bias) * F_brake_k
            
            Fx_f = Ff_acc + Ff_brk
            Fx_r = Fr_acc + Fr_brk

            # 2. Normal loads (fz)
            
            if use_drs_vec:
                cl_k = cl_eff[k]
                cd_k = cd_eff[k]
            else:
                cl_k = cl_eff 
                cd_k = cd_eff

            # Fz calculations
            fz_aero = 0.5 * self.car.rho * cl_k * vx_k**2
            fz_aero_f = fz_aero * (1 - self.car.cop)
            fz_aero_r = fz_aero * self.car.cop
            
            Fz_f = self.car.fz_static_f + fz_aero_f
            Fz_r = self.car.fz_static_r + fz_aero_r

            # 3. Slip angles - kinematics
            # alpha_f = -atan2(...)
            # alpha_r = -atan2(...)
            
            # Front slip
            vy_f_kin = (vy_k + self.car.lf * r_k)
            alpha_f = -ca.atan2(
                vy_f_kin * ca.cos(delta_k) - vx_k * ca.sin(delta_k),
                vy_f_kin * ca.sin(delta_k) + vx_k * ca.cos(delta_k)
            )

            # Rear slip
            vy_r_kin = (vy_k - self.car.lr * r_k)
            alpha_r = -ca.atan2(vy_r_kin, vx_k)

            # 4. Lateral forces
            # Fy = (alpha_deg / 5.0) * Fz * mu_y
            # Convert alpha to deg: alpha * 180/pi
            # Formula: alpha_rad * (180/pi) / 5.0 * Fz * mu_y
            #        = alpha_rad * (36.0/pi) * Fz * mu_y
            
            Fy_f_lin = alpha_f * (36.0 / np.pi) * Fz_f * p_mu_y
            Fy_r_lin = alpha_r * (36.0 / np.pi) * Fz_r * p_mu_y

            # 5. constraints
            
            # A) Hard limit on slip angle to keep linear model valid
            opti.subject_to(opti.bounded(-alpha_limit, alpha_f, alpha_limit))
            opti.subject_to(opti.bounded(-alpha_limit, alpha_r, alpha_limit))

            # B) Friction ellipse 
            # (Fx / Fx_max)^2 + (Fy / Fy_max)^2 <= 1
            # Fx_max = mu_x * Fz, Fy_max = mu_y * Fz
        
            
            mu_x_fz_f = p_mu_x * Fz_f
            mu_y_fz_f = p_mu_y * Fz_f
            
            mu_x_fz_r = p_mu_x * Fz_r
            mu_y_fz_r = p_mu_y * Fz_r

            ellipse_f = (Fx_f / mu_x_fz_f)**2 + (Fy_f_lin / mu_y_fz_f)**2
            ellipse_r = (Fx_r / mu_x_fz_r)**2 + (Fy_r_lin / mu_y_fz_r)**2
            
            opti.subject_to(ellipse_f <= 1.0)
            opti.subject_to(ellipse_r <= 1.0)

            # 6. Power & Torque Limits
            # Torque: F_acc * radius / (ratio * 4) < max_torque (simplified per wheel check)
            # But F_acc is total force. 
            # Total Torque = F_acc * radius / ratio
            # Check against engine max torque * gear ratio logic or simplier force limit
            # Using your class params:
            
            # Power limit: P = F * v <= P_max
            opti.subject_to( ((F_acc_k * vx_k) / self.car.power_eff)/self.car.power_max <= 1 )
            
            # Regen limit
            opti.subject_to( ((-F_brake_k * vx_k) * self.car.regen_eff)/self.car.brake_max <= 1 )

            # Torque limit (simplified force limit at wheel)
            #max_force_torque = (self.car.max_torque * self.car.gear_ratio * 4) / self.car.wheel_radius
            #opti.subject_to( F_acc_k <= max_force_torque )

            # 7. Dynamics 
            # Based on singletrack_path_ode logic
            
            # Drag
            F_drag = 0.5 * self.car.rho * cd_k * vx_k**2 + self.car.fix_drag

            # Body frame forces
            # Sum forces in x and y direction
            F_x_total = Fx_f * ca.cos(delta_k) - Fy_f_lin * ca.sin(delta_k) + Fx_r - F_drag
            F_y_total = Fx_f * ca.sin(delta_k) + Fy_f_lin * ca.cos(delta_k) + Fy_r_lin
            
            # Moment
            M_z = self.car.lf * (Fx_f * ca.sin(delta_k) + Fy_f_lin * ca.cos(delta_k)) - self.car.lr * Fy_r_lin

            # Time derivatives (dt)
            dvx_dt = F_x_total / p_mass + r_k * vy_k
            dvy_dt = F_y_total / p_mass - r_k * vx_k
            dr_dt  = M_z / self.car.Iz
            
            # Path scaling factor (Sf = dt/ds)
            # d_psi/dt = r
            # d_theta/dt = vx * curvature? No, d_theta/ds = curvature -> d_theta/dt = ds/dt * curvature
            # We need Sf = 1 / (ds/dt)
            # s_dot = (vx * cos(xi) - vy * sin(xi)) / (1 - n * curvature)
            
            # eps (in code) is xi (here). 
            # Note: your code uses eps = psi - th. Here xi = psi - th.
            s_dot_denom = (1.0 - n_k * curv_k)
            s_dot_num   = (vx_k * ca.cos(xi_k) - vy_k * ca.sin(xi_k))
            
            # Avoid division by zero
            ds_dt = s_dot_num / ca.fmax(s_dot_denom, 0.1)
            Sf = 1.0 / ds_dt  # dt/ds

            # Spatial derivatives (d/ds = d/dt * dt/ds)
            dvx_ds = dvx_dt * Sf
            dvy_ds = dvy_dt * Sf
            dr_ds  = dr_dt  * Sf
            dt_ds  = Sf
            
            # Kinematics in path coordinates
            # dn/dt = vx * sin(xi) + vy * cos(xi)
            dn_ds = (vx_k * ca.sin(xi_k) + vy_k * ca.cos(xi_k)) * Sf
            
            # dxi/ds = dpsi/ds - dtheta/ds = r * Sf - curvature
            dxi_ds = r_k * Sf - curv_k

            # Energy
            dE_ds = ((F_acc_k / self.car.power_eff + F_brake_k * self.car.regen_eff) * Sf) / 3.6e6

            # Euler Step
            opti.subject_to(vx[k+1] == vx_k + dvx_ds * ds_val)
            opti.subject_to(vy[k+1] == vy_k + dvy_ds * ds_val)
            opti.subject_to(r[k+1]  == r_k  + dr_ds  * ds_val)
            opti.subject_to(n[k+1]  == n_k  + dn_ds  * ds_val)
            opti.subject_to(xi[k+1] == xi_k + dxi_ds * ds_val)
            opti.subject_to(t[k+1]  == t_k  + dt_ds  * ds_val)
            opti.subject_to(E[k+1] == E[k] + dE_ds * ds_val)


        # =========================
        # Boundary Conditions
        # =========================
        # Start
        opti.subject_to(vx[0] == init_cond.v)
        opti.subject_to(vy[0] == 0) # Assume 0 lateral speed at start
        opti.subject_to(r[0]  == 0) # Assume 0 yaw rate at start (or match curvature)
        opti.subject_to(n[0]  == init_cond.n)
        opti.subject_to(xi[0] == init_cond.xi)
        opti.subject_to(t[0]  == init_cond.T)
        opti.subject_to(E[0]  == init_cond.E)

        # Cyclic condition (if needed, enable for closed track)
        # opti.subject_to(vx[0] == vx[-1])
        
        # Energy limit
        if self.car.energy_limit is not None:
            opti.subject_to(E[-1] <= self.car.energy_limit)

        # =========================
        # Objective function
        # =========================
        J_Time = t[-1]
        
        # Regularization terms
        # 1. Steering Rate (smooth steering)
        # sum (delta_k - delta_k-1)^2
        diff_delta = delta[1:] - delta[:-1]
        J_steer_rate = 7e-4 * ca.sumsqr(diff_delta)
        
        # 2. Acceleration smoothness (throttle smoothing)
        diff_acc = F_acc[1:] - F_acc[:-1]
        J_acc_smooth = 1e-3 * ca.sumsqr(diff_acc) / 1000.0 # Scale down force magnitude

        # 3. Path deviation (stay close to center/opt line)
        J_path = 0.5 * ca.sumsqr(n)
        #0.5 funguje mega dobre

    
        opti.minimize(J_Time+J_path+J_steer_rate)
        #opti.minimize(J_Time)

        # =========================
        # Initial guess 
        # =========================
   
        # Mapping guess to variables
        # Note: guess arrays might be slightly different length due to float math, strict check needed
        g_vx = init_guess['v']
        g_n  = init_guess['n']
        g_xi = init_guess['xi']
        g_r  = init_guess['r']
        g_t  = init_guess['t']
        g_delta = init_guess['delta']
        g_facc  = init_guess['F_acc']
        g_fbrk  = init_guess['F_brake']

        # set initial (handling minimal length mismatch if any)
        L_state = min(len(g_vx), N + 1)
        opti.set_initial(vx[:L_state], g_vx[:L_state])
        opti.set_initial(vy[:L_state], 0.0) # Start with 0 slip
        opti.set_initial(n[:L_state],  g_n[:L_state])
        opti.set_initial(xi[:L_state], g_xi[:L_state])
        opti.set_initial(r[:L_state],  g_r[:L_state])
        opti.set_initial(t[:L_state],  g_t[:L_state])
        
        L_ctrl = min(len(g_delta), N)
        opti.set_initial(delta[:L_ctrl], g_delta[:L_ctrl])
        opti.set_initial(F_acc[:L_ctrl], g_facc[:L_ctrl])
        opti.set_initial(F_brake[:L_ctrl], g_fbrk[:L_ctrl])

        # =========================
        # Solver
        # =========================
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 3000, 
            "tol": 1e-3, 
            "print_level": 5, 
            "mu_strategy": "adaptive" 
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            
            # --- Post processing / Sensitivities ---
            L_lagrange = opti.f + ca.dot(opti.lam_g, opti.g)
            sens_m = sol.value(ca.jacobian(L_lagrange, p_mass))
            sens_cl = sol.value(ca.jacobian(L_lagrange, p_cl))
            
            # Calculate alphas for result (re-evaluate expressions)
            # We can't easily get them from 'sol' directly as they are intermediate expressions
            # so we assume user can recalc them from states if needed, or we compute approximate:
            res_vx = sol.value(vx)
            res_vy = sol.value(vy)
            res_r  = sol.value(r)
            res_delta = sol.value(delta)
            
            # Recompute alphas for output (k=0..N-1)
            # (Simplified re-calculation for visualization)
            res_alpha_f = []
            res_alpha_r = []
            for k in range(N):
                v_x, v_y, yaw_r, delt = res_vx[k], res_vy[k], res_r[k], res_delta[k]
                vy_f = v_y + self.car.lf * yaw_r
                af = -np.arctan2(vy_f * np.cos(delt) - v_x * np.sin(delt), vy_f * np.sin(delt) + v_x * np.cos(delt))
                vy_r = v_y - self.car.lr * yaw_r
                ar = -np.arctan2(vy_r, v_x)
                res_alpha_f.append(af)
                res_alpha_r.append(ar)

            return SolverResults(
                success=True,
                lap_time=sol.value(t[-1]),
                t=sol.value(t),
                v=sol.value(vx), # approx total speed
                s=s_sim,
                f_acc=sol.value(F_acc) / self.car.power_eff,
                f_brake=sol.value(F_brake) * self.car.regen_eff,
                delta=sol.value(delta),
                vx=res_vx,
                vy=res_vy,
                r=res_r,
                n=sol.value(n),
                xi=sol.value(xi),
                alpha_f=np.array(res_alpha_f),
                alpha_r=np.array(res_alpha_r),
                E=sol.value(E),
                drs_usage=sol.value(drs_ef),
                sens={"dTime_dm": sens_m, "dTime_dCL": sens_cl}
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            # Return partial results (debug values) if available
            return SolverResults(False, 0.0, [], [], [], [], [], [], {}, mu_x=self.car.mu_x, mu_y=self.car.mu_y)

class MPCC_Solver:
    def __init__(self,car,track,endurance=False,laps=1):

        self.car = car
        self.track = track
        self.laps = laps
        self.endurance = endurance
        
        if self.endurance:
            self.target_lap = car.acp_capacity/laps

    def solveLap(self, init_cond: InitConditions, init_guess=None, ds=0.5):

        opti = ca.Opti()

        # =========================
        # Parameters
        # =========================
        p_mass = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_mu_x = opti.parameter()
        p_mu_y = opti.parameter()

        # Set params values
        opti.set_value(p_mass, self.car.mass)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_mu_x, self.car.mu_x)
        opti.set_value(p_mu_y, self.car.mu_y)

        # =========================
        # Track data
        # =========================
        total_length = self.track.length
        N = int(np.round(total_length / ds))

        s_sim = np.linspace(0, total_length, N + 1)
        ds = s_sim[1:] - s_sim[:-1]

        kappa_grid = self.track.get_curvature(s_sim)

        # =========================
        # States
        # =========================
        v = opti.variable(N + 1)   # speed [m/s]
        T = opti.variable(N + 1)   # time [s]
        E = opti.variable(N + 1)   # energy used [kWh]
        n = opti.variable(N + 1)   # distance of centerline
        phi = opti.variable(N + 1) # heading error [rad]

        # =========================
        # Inputs
        # =========================
        F_acc   = opti.variable(N)   # Gas [N]
        F_brake = opti.variable(N)   # Brake [N]
        k_car   = opti.variable(N)   # Curvature of the car's path
        drs     = opti.variable(N)   # DRS position [0-1]

        # =========================
        # DRS
        # =========================
        if self.car.drs_allowed and self.track.drs is not None:
            drs_flag = self.track.get_drs_flag(s_sim[:-1])
            drs_ef = drs * drs_flag
            cl_eff = p_cl + drs_ef * (self.car.cl_drs - p_cl)
            cd_eff = p_cd + drs_ef * (self.car.cd_drs - p_cd)
        else:
            drs_ef = ca.DM.zeros(N)
            cl_eff = p_cl
            cd_eff = p_cd

        opti.subject_to(drs >= 0)
        opti.subject_to(drs <= 1)

        # Current speed (k)
        v_k = v[:-1]

        # Race line curvature at current position
        sf = (1 - n[:-1] * kappa_grid[:-1])
        # d(n)/ds = sf * tan(phi)
        dn_ds = sf * ca.tan(phi[:-1])
        # d(phi)/ds = k_car * (ds_car / ds_center) - k_ref
        # ds_car / ds_center = sf / cos(phi)
        dphi_ds = k_car * (sf / ca.cos(phi[:-1])) - kappa_grid[:-1]
        # d(t)/ds = (ds_car / ds_center) / v
        dt_ds = sf / (v_k * ca.cos(phi[:-1])) 

        # Integrate geometry (euler)
        opti.subject_to(n[1:]   == n[:-1]   + dn_ds * ds)
        opti.subject_to(phi[1:] == phi[:-1] + dphi_ds * ds)
        opti.subject_to(T[1:]   == T[:-1]   + dt_ds * ds)

        #limit racing line to 2 m from center
        opti.subject_to(opti.bounded(-1,n,1))
        # limit heading error to ~30 degrees
        opti.subject_to(opti.bounded(-0.5, phi, 0.5)) 
        opti.subject_to(opti.bounded(-0.15, k_car, 0.15)) #limit max curvature

        # =========================
        # Aero
        # =========================
        F_drag = Car.get_aero_drag(self.car, v_k, cd=cd_eff)

        F_max_x, F_max_y = Car.get_friction_limit(
            self.car, v_k,
            mass=p_mass,
            mu_x=p_mu_x,
            mu_y=p_mu_y,
            cl=cl_eff
        )

        #change to car curvature
        # Lateral Force: F_lat = m * v^2 * curvature
        F_lat = p_mass * v_k**2 * ca.fabs(k_car)

        # Longitudinal force
        F_long_k = F_acc + F_brake

        # =========================
        # Pedal constraints
        # =========================
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake <= 0)

        # =========================
        # Friction ellipse
        # =========================
        term_long = (F_long_k / F_max_x)**2
        term_lat  = (F_lat    / F_max_y)**2
        opti.subject_to(term_long + term_lat <= 1)

        # =========================
        # Dynamics
        # =========================
        F_net = F_long_k - F_drag
        acc = F_net / p_mass

        # Speed integration: v^2 = v0^2 + 2*a*ds
        #but we have to use ds car 
        ds_car = ds * sf / ca.cos(phi[:-1])

        v_next_sq = v_k**2 + 2 * acc * ds_car
        opti.subject_to(v[1:]**2 == v_next_sq)

        # Time integration: T = T0 + ds/v
       # opti.subject_to(T[1:] == T[:-1] + ds_car / v_k)

        # Energy integration
        dE_step = (F_acc / self.car.power_eff + F_brake * self.car.regen_eff) * ds_car
        opti.subject_to(E[1:] == E[:-1] + dE_step / 3.6e6)  # J -> kWh

        # =========================
        # Torque limit (30 Nm per wheel)
        # =========================
        opti.subject_to(((self.car.wheel_radius * F_acc)) /
                    (self.car.max_torque * self.car.gear_ratio * 4) < 1)

        # =========================
        # Power limit
        # =========================
        opti.subject_to(((F_acc * v_k) / self.car.power_eff) / self.car.power_max <= 1)

        # Regen limit
        opti.subject_to(((-F_brake * v_k) * self.car.regen_eff) / self.car.brake_max <= 1)

        # Speed limits
        opti.subject_to(v >= 1.0)  # avoid division by zero

        # =========================
        # Initial conditions
        # =========================
        opti.subject_to(v[0] == init_cond.v)
        opti.subject_to(T[0] == init_cond.T)
        opti.subject_to(E[0] == init_cond.E)
        opti.subject_to(n[0] == 0) #start with car on the line
        opti.subject_to(phi[0] == 0) #start with zero heading error
        opti.subject_to(drs[0] == 0)  # start with DRS closed

        # =========================
        # Energy limitation
        # =========================
        if not self.endurance:
            if self.car.energy_limit is not None:
                opti.subject_to(E[-1] <= self.car.energy_limit)
        else:
            target = init_cond.E + init_cond.E_target
            opti.subject_to(E[-1] <= target)

        # =========================
        # Objective function
        # =========================
        J_Time = T[-1]

        F_norm_factor = (self.car.mass * self.car.g)**2
        J_overlap = 0.004 * ca.sum1((F_acc * -F_brake) / F_norm_factor)

        J_reg_steer = 1.0 * ca.sumsqr(k_car[1:] - k_car[:-1])

        W_line = 10
        J_line = W_line * ca.sumsqr(n)

        opti.minimize(J_Time + J_overlap)

        # =========================
        # Initial guess
        # =========================
        if init_guess is not None:
            opti.set_initial(v, 0.9*init_guess['v'])
            opti.set_initial(T, init_guess['t'])
            opti.set_initial(F_acc, 0.9*init_guess['F_acc'])
            opti.set_initial(F_brake,0.9* init_guess['F_brake'])
        else:
            opti.set_initial(v, 15.0)
            opti.set_initial(F_acc, 100.0)

        opti.set_initial(k_car,kappa_grid[:-1])

        # =========================
        # Solver setup
        # =========================
        p_opts = {'expand': True}
        s_opts = {'max_iter': 2000, 'tol': 1e-3, 'print_level': 5}

        opti.solver('ipopt', p_opts, s_opts)

        try:
            sol = opti.solve()

            # =========================
            # Sensitivity analysis
            # =========================
            L = opti.f + ca.dot(opti.lam_g, opti.g)

            sens_m   = sol.value(ca.jacobian(L, p_mass))
            sens_cl  = sol.value(ca.jacobian(L, p_cl))
            sens_cd  = sol.value(ca.jacobian(L, p_cd))
            sens_mux = sol.value(ca.jacobian(L, p_mu_x))
            sens_muy = sol.value(ca.jacobian(L, p_mu_y))

            print("sensitivity results")
            print(f"Mass: {sens_m:+.4f} s/kg")
            print(f"CL:   {sens_cl:+.4f} s/unit")
            print(f"CdA:  {sens_cd:+.4f} s/unit")

            return SolverResults(
                success=True,
                lap_time=sol.value(T[-1]),
                t=sol.value(T),
                v=sol.value(v),
                s=s_sim,
                kappa_car = sol.value(k_car),
                n=sol.value(n),
                xi=sol.value(phi),
                E=sol.value(E),
                f_acc=sol.value(F_acc) / self.car.power_eff,
                f_brake=sol.value(F_brake) * self.car.regen_eff,
                drs_usage=sol.value(drs_ef),
                sens={
                    'dTime_dm': sens_m,
                    'dTime_dCL': sens_cl,
                    'dTime_dCdA': sens_cd,
                    'dTime_dMuX': sens_mux,
                    'dTime_dMuY': sens_muy
                }
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return SolverResults(False, 0, [], [], [], [], [], [], {})


class StrategyMaker:

    def __init__(self,car,laptime_map,laps):
        self.car = car
        self.laptime_map = laptime_map
        self.laps = laps
        pass


    def ef_guess(self,energy_target_endurance):
        #simulate the car with x kWh with max regen to estimate winning EF_score
        N = self.laps
        total_time = 0
        energy_target_lap = energy_target_endurance/self.laps
        SoC_start = 0.8

        #simulate the race
        for i in range(0,N,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.car.acp_capacity)
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,0.94]))
            total_time += laptime
            #print(f"lap {i+1}, laptime {laptime:.2f}, SoC start: {SoC_current:.2f} SoC end: {SoC_start - (i+1)*(energy_target_lap/self.car.acp_capacity):.2f}")
        
        ef_score = total_time**2*energy_target_endurance
        #print(f"Total time: {total_time}")
        #print(f"Ef_score {ef_score}")

        return ef_score
    

    def best_T_guess(self,delta):
        #simulate the car with full energy usage to simulate best time
        N = self.laps
        total_time = 0
        energy_target_lap = (self.car.acp_capacity-0.4)/self.laps
        SoC_start = 1.0

        #simulate the race
        for i in range(0,N,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.car.acp_capacity)
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,0.94]))
            total_time += laptime
            print(f"lap {i+1}, laptime {laptime:.2f}, SoC start: {SoC_current:.2f} SoC end: {SoC_start - (i+1)*(energy_target_lap/self.car.acp_capacity):.2f}")
        
        
        print(f"Total time: {total_time}")
        #print(f"Ef_score {ef_score}")
        return total_time*(1-delta)
    
    def getPoints(self,regen,energy_target,delta,ef_min,t_min):

        total_time = 0
        SoC_start = 1.0
        energy_target_lap = energy_target/self.laps

         #simulate the race
        for i in range(0,self.laps,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.car.acp_capacity)
            if SoC_current < 0.05: 
                return 0.0
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,regen]))
            total_time += laptime


        #calculate total points from endurance
        T_min_adj = t_min*(1-delta)
        T_max = 1.5 * T_min_adj # Podle tabulky 11 v pravidlech            
        P_max_end = 250.0
        P_min_end = 0.1 * P_max_end

        if total_time > T_max:
            nom_end = 0.0
        elif total_time < T_min_adj:
            nom_end = 1.0
        else:
            nom_end = (T_max - total_time) / (T_max - T_min_adj)
            
        Endurance_points = (P_max_end - P_min_end) * (nom_end**2) + P_min_end

        if total_time > T_max: 
            Endurance_points = 0.0

        P_max_eff = 75.0
        EF_max = 2 * ef_min
        Ef_team = (total_time**2) * energy_target

        if Ef_team > EF_max:
            nom_eff = 0.0
        elif Ef_team < ef_min:
            nom_eff = 1.0
        else:
            nom_eff = (EF_max - Ef_team) / (EF_max - ef_min)
            
        Efficiency_points = P_max_eff * (nom_eff**2)
        

        return Endurance_points + Efficiency_points


    def solve(self,ef_min=0,T_min=0,objective="points",enforce_full_soc = False,):
            opti = ca.Opti()
            N = self.laps

            E_lap = opti.variable(N)

            
            SoC_start = opti.variable()

            if enforce_full_soc:
                opti.subject_to(SoC_start == 0.99)
        
            
            # Stav baterie (SoC) na začátku každého kola + na úplném konci závodu (N+1)
            SoC = opti.variable(N + 1)    

            MIN_SOC = 0.11

            # ==========================================
            # 3. Počáteční podmínky
            # ==========================================
            opti.subject_to(SoC[0] == SoC_start) 
        

            total_time = 0
            
            for k in range(N):
            
                opti.subject_to(SoC[k+1] == SoC[k] - (E_lap[k] / self.car.acp_capacity))
               
                state_vec = ca.vertcat(SoC[k], E_lap[k], self.car.regen_eff)
                laptime = self.laptime_map(state_vec)
    
                total_time += laptime


    
            opti.subject_to(SoC >= MIN_SOC)
            opti.subject_to(SoC_start >= 0.5)
            opti.subject_to(SoC_start <= 0.99)
            #opti.subject_to(SoC <= 1.0)
            
        
            opti.subject_to(E_lap >= 0.17) 
            opti.subject_to(E_lap <= 0.4) 


            # === INITIAL GUESS ===
            E_guess = 0.25
            opti.set_initial(E_lap, E_guess)
            opti.set_initial(SoC_start, 1.0)
            soc_guess = [1.0 - k * (E_guess / self.car.acp_capacity) for k in range(N + 1)]
            opti.set_initial(SoC, soc_guess)
           
        
            # ==========================================
            # Konstanty pro výpočty 
            # ==========================================
            
            T_max = 1.5 * T_min # Podle tabulky 11 v pravidlech
            
            P_max_end = 250.0
            P_min_end = 0.1 * P_max_end 

            # ==========================================
            # Endurance Score
            # ==========================================
            if objective == "points":
                opti.subject_to(total_time <= T_max) 
            
            nom_end = (T_max - total_time) / (T_max - T_min)
            Endurance_points = (P_max_end - P_min_end) * (nom_end**2) + P_min_end

            # ==========================================
            # Efficiency Score
            # ==========================================
            P_max_eff = 75.0
            EF_max = 2 * ef_min
            Ef_team = (total_time**2) * ca.sum1(E_lap)

            if objective == "points":
                opti.subject_to(Ef_team <= EF_max)

            nom_eff = (EF_max - Ef_team) / (EF_max - ef_min)
            Efficiency_points = P_max_eff * (nom_eff**2)
            
            # ==========================================
            # Objective Function
            # ==========================================
            Total_points = Endurance_points + Efficiency_points
            
            if objective == "points":
                opti.minimize(-Total_points)

            elif objective == "time":
                opti.minimize(total_time)

            elif objective == "efficiency":
                opti.minimize(Ef_team)
            
            # ==========================================
            # 8. Solver settings
            # ==========================================
            p_opts = {"expand": True}
            s_opts = {
                "max_iter": 5000, 
                "tol": 1e-3,                   
                "acceptable_tol": 1e-2,           
                "print_level": 0,
                "hessian_approximation": "limited-memory"
            }
            opti.solver('ipopt', p_opts, s_opts)

            try:
                
                sol = opti.solve()
                
                print(f"Celkový čas: {sol.value(total_time):.2f} s")
                
                # Extrakce optimálních hodnot
                opt_soc = sol.value(SoC)
                opt_e_targets = sol.value(E_lap)
                
                # Ošetření pro případ, že N=1 (CasADi vrací skalár místo pole)
                if N == 1:
                    opt_e_targets = [opt_e_targets]
                    opt_soc = [opt_soc, sol.value(SoC)[1]]

                opt_lap_times = []
                
               
                for k in range(N):
                    # Extrakce konkrétních hodnot pro toto kolo
                    soc_k = float(opt_soc[k])
                    e_k = float(opt_e_targets[k])
                    regen_k = float(self.car.regen_eff)
                    
                    # Výpočet času na kolo z mapy
                    lap_time_k = float(sol.value(self.laptime_map(ca.vertcat(soc_k, e_k, regen_k))))
                    opt_lap_times.append(lap_time_k)
                    
                    # POKUD JE ČAS PODEZŘELE NÍZKÝ, VYPÍŠEME VSTUPY
                    if lap_time_k < 60.0: # Nastavte podle reálného rekordu trati
                        print(f"!!! ANOMÁLIE V KOLE {k+1} !!!")
                        print(f"   Laptime: {lap_time_k:.2f} s")
                        print(f"   Vstupy -> SoC_start: {soc_k:.4f}, E_lap: {e_k:.4f}, Regen: {regen_k:.2f}")

                opt_total_time = sol.value(total_time)

                if opt_total_time < T_min:
                    print("Závod byl odjet rychleji než je povolené T_min, vracím False.")
                    return {"success": False}

                return {
                    "success": True,
                    "total_time": opt_total_time,
                    "e_targets": opt_e_targets,     
                    "soc_profile": opt_soc,
                    "soc_start": sol.value(SoC_start),
                    "lap_times": opt_lap_times,
                    "points": sol.value(Total_points),
                    "ef_score": sol.value(Ef_team)
                }
                
            except RuntimeError as e:
                print(f"Solver failed: {e}")
                opti.debug.show_infeasibilities()
                return {"success": False}

       
@staticmethod
def calculate_initial_guess(car,track, init_cond: InitConditions, ds=1.0):
        #creates warm start for the solver based on simple forward-backward integration and curve speed limits
       
        total_length = track.length
        N = int(np.round(total_length / ds))
        s_sim = np.linspace(0, total_length, N + 1)
        
        ds_const = total_length / N 
        
        kappa_dm = track.get_curvature(s_sim)
        kappa = np.array(kappa_dm).flatten()
        
        m = car.mass
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

        '''
        plt.figure(figsize=(10, 8))
            
        plt.subplot(4,1,1)
        plt.plot(s_sim, v_smooth * 3.6, label='Speed [km/h]') # s_sim (N+1) vs v (N+1) -> OK
        plt.ylabel('v [km/h]')
        plt.grid(True)
            
        plt.subplot(4,1,2)
            # POZOR: s_sim[:-1] pro síly (protože síly jsou na intervalech)
        plt.plot(s_sim[:-1], F_acc_guess, 'g', label='F Acc [N]') 
        plt.plot(s_sim[:-1], F_brake_guess, 'r', label='F Brake [N]')
        plt.ylabel('Force [N]')
        plt.legend()
        plt.grid(True)
            
        plt.subplot(4,1,3)
        plt.plot(s_sim, E_guess, label='Energy [kWh]') # s_sim (N+1) vs E (N+1) -> OK
        plt.ylabel('E [kWh]')
        plt.grid(True)
            
        plt.subplot(4,1,4)
        plt.plot(s_sim, T_guess, label='Time [s]') # s_sim (N+1) vs T (N+1) -> OK
        plt.ylabel('T [s]')
        plt.xlabel('Distance [m]')
        plt.grid(True)
            
        plt.tight_layout()
        plt.show()
        '''
        return {
            'v': np.array(v_smooth).flatten(),       
            'T': np.array(T_guess).flatten(),        
            'E': np.array(E_guess).flatten(),        
            'F_acc': np.array(F_acc_guess).flatten(),   
            'F_brake': np.array(F_brake_guess).flatten()
        }

@staticmethod
def calculate_initial_guess_single(car, track, init_cond: 'InitConditions', ds=1.0):
        """
    
        1. Longitudinal: Forward-backward integrace
        2. Lateral: Steady-state bicycle model
        """
       
        total_length = track.length
        N = int(np.round(total_length / ds))
        s_sim = np.linspace(0, total_length, N + 1)
        ds_const = total_length / N 
        
        # Track data
        kappa_dm = track.get_curvature(s_sim)
        kappa = np.array(kappa_dm).flatten()
        psi_track = np.array(track.get_heading(s_sim)).flatten()
        
        
        try:
             x_track = np.interp(s_sim, track.s, track.x)
             y_track = np.interp(s_sim, track.s, track.y)
        except:
             x_track = np.zeros_like(s_sim)
             y_track = np.zeros_like(s_sim)

        # Car parametry
        m = car.mass
        g = car.g
        rho = car.rho
     
        Cl = car.cl 
        Cd = car.cd 
        mu_x = car.mu_x * 0.95 # Malá rezerva pro warm start
        mu_y = car.mu_y * 0.95
        
        lf = car.lf
        lr = car.lr
        L = car.wheelbase
        cop = car.cop 

      
        #longitudinal profile
        #   
        v_max_car = 40.0 # [m/s]
        v_curve = np.full(N+1, v_max_car)
        
        # Limit v zatáčkách: m*v^2*k <= mu_y * Fz
        # Fz = mg + 0.5*rho*Cl*v^2
        term_aero = 0.5 * rho * Cl
        denom = m * np.abs(kappa) - mu_y * term_aero
        
        mask = denom > 0
        v_sq_limit = (mu_y * m * g) / denom[mask]
        v_curve[mask] = np.sqrt(v_sq_limit)
        v_curve = np.minimum(v_curve, v_max_car)

        # A) Forward 
        v_fwd = np.zeros(N+1)
        v_fwd[0] = max(init_cond.v, 5.0)
        
        for k in range(N):
            v_curr = v_fwd[k]
            F_drag = 0.5 * rho * Cd * v_curr**2
            F_z = m*g + 0.5 * rho * Cl * v_curr**2
            
            F_grip_long = mu_x * F_z
            F_power = car.power_max / max(v_curr, 1.0) * car.power_eff
            
            F_acc_avail = min(F_grip_long, F_power) - F_drag
            acc = F_acc_avail / m
            
            v_next = np.sqrt(max(v_curr**2 + 2 * acc * ds_const, 1.0))
            v_fwd[k+1] = min(v_next, v_curve[k+1])

        # B) Backward 
        v_bwd = np.zeros(N+1)
        v_bwd[-1] = v_fwd[-1]
        
        for k in range(N, 0, -1):
            v_curr = v_bwd[k]
            
            # 1. Síly od prostředí
            F_drag = 0.5 * rho * Cd * v_curr**2
            F_z = m*g + 0.5 * rho * Cl * v_curr**2
            
            # 2. Limit přilnavosti (Grip)
            F_grip_long = mu_x * F_z
            
            # 3. Limit výkonu (Power) -> F = P / v
            #    Používáme max(v, 1.0) aby nedošlo k dělení nulou
            F_power_limit = abs(car.brake_max) / max(v_curr, 1.0)
            
            # 4. Určení limitní brzdné síly kol
            #    Auto nemůže brzdit víc, než dovolí grip, A ZÁROVEŇ 
            #    nemůže brzdit víc, než dovolí motory (pokud nemá mechanické brzdy).
            F_brake_effective = min(F_grip_long, F_power_limit)
            
            # 5. Výpočet maximální možné dekelerace
            #    (Brzdná síla kol + Odpor vzduchu) / m
            dec_max = (F_brake_effective + F_drag) / m
            
            # 6. Kinematika zpětného chodu
            #    v_prev^2 = v_curr^2 + 2 * a * s
            v_prev_sq = v_curr**2 + 2 * dec_max * ds_const
            v_prev = np.sqrt(v_prev_sq)
            
            # 7. Aplikace limitů (zatáčka, forward pass)
            v_bwd[k-1] = min(v_prev, v_curve[k-1], v_fwd[k-1])

        v_smooth = v_bwd # Finální rychlost

        #lateral dynamics
    
        F_lat_total = m * v_smooth**2 * kappa
        
        # (Static + Aero distribution)
   
        Fz_total = m*g + 0.5 * rho * Cl * v_smooth**2
        Fz_aero = 0.5 * rho * Cl * v_smooth**2
        
        # Rozdělení Fz
        Fz_f = car.fz_static_f + Fz_aero * (1 - cop)
        Fz_r = car.fz_static_r + Fz_aero * cop
        
        # moment equation to center of mass
        # Fy_f * lf = Fy_r * lr => Fy_r = Fy_f * lf/lr
        # Fy_f + Fy_r = F_lat_total
      
        Fy_f_req = F_lat_total * (lr / L)
        Fy_r_req = F_lat_total * (lf / L)
        
        # --- inverse tire model
        #  Fy = (alpha_deg / 5.0) * Fz * mu_y
        # inverse: alpha_deg = (Fy / (Fz * mu_y)) * 5.0
        
        max_Fy_f = Fz_f * mu_y
        max_Fy_r = Fz_r * mu_y
        
        Fy_f_eff = np.clip(Fy_f_req, -max_Fy_f, max_Fy_f)
        Fy_r_eff = np.clip(Fy_r_req, -max_Fy_r, max_Fy_r)

        OPTIMAL_SLIP_ANGLE_DEG = 6.0

        ratio_f = Fy_f_eff / np.maximum(max_Fy_f, 1.0) # prevence /0
        ratio_r = Fy_r_eff / np.maximum(max_Fy_r, 1.0)
        
        alpha_f_deg = ratio_f * OPTIMAL_SLIP_ANGLE_DEG
        alpha_r_deg = ratio_r * OPTIMAL_SLIP_ANGLE_DEG
        
    
        alpha_f_rad = np.radians(alpha_f_deg)
        alpha_r_rad = np.radians(alpha_r_deg)
        
  
        delta_guess = L * kappa + alpha_f_rad - alpha_r_rad
        
  
        delta_guess = np.clip(delta_guess, -car.steeringAngle_max, car.steeringAngle_max)

        # outputs
        
        acc_req = np.diff(v_smooth**2, prepend=v_smooth[0]**2) / (2 * ds_const)
        F_net = m * acc_req
        F_drag_vec = 0.5 * rho * Cd * v_smooth**2
        F_total_req = F_net + F_drag_vec
        
        F_acc_guess = np.maximum(F_total_req, 0)
        F_brake_guess = np.minimum(F_total_req, 0)
        
   
        dt_vec = ds_const / np.maximum(v_smooth, 1.0)
        t_guess = np.cumsum(dt_vec)
        
        # Yaw rate
        r_guess = v_smooth * kappa

        return {
            'v': v_smooth,
            'vy': np.zeros_like(v_smooth), 
            'r': r_guess,
            'psi': psi_track,
            'x': x_track,
            'y': y_track,
            'n': np.zeros_like(v_smooth),
            'xi': np.zeros_like(v_smooth), # Heading error relative to track
            'F_acc': F_acc_guess[:-1],
            'F_brake': F_brake_guess[:-1],
            'delta': delta_guess[:-1],
            't': t_guess,
            's': s_sim,
            'drs_usage':np.zeros(N)
        }