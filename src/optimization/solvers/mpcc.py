import casadi as ca
import numpy as np
from src.core.car import Car
from src.optimization.solvers.base_solver import BaseSolver, SolverResults, InitConditions



#does not work, i leave it here only for reference, it was an attempt but ended up being too slow and not converging well, you might want to use some of this code if you want to implement a working MPCC in the future


class MPCC_Solver(BaseSolver):
    def __init__(self, car, track):

        super().__init__(car, track)





def solve(self, init_cond: InitConditions, init_guess=None, ds=0.5):

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