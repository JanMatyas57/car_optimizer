import casadi as ca
import numpy as np
from src.core.car import Car
from src.optimization.solvers.base_solver import BaseSolver, SolverResults, InitConditions


class MassPointSolver(BaseSolver):
    def __init__(self, car, track):

        super().__init__(car, track)

    def solve(self, init_cond: InitConditions, init_guess=None, ds=0.5):

        opti = ca.Opti()

        # =========================
        # Parameters
        # =========================
        p_weight = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
     

        # Set params values
        opti.set_value(p_weight, self.car.weight)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)


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
       # F_drag = Car.get_aero_drag(self.car, v_k, cd=cd_eff)

        F_max_x, F_max_y = Car.get_friction_limit(
            self.car, v_k,
            weight=p_weight,    
            use_cl=cl_eff
        )

        # Lateral Force: F_lat = m * v^2 * curvature
        F_lat = p_weight * v_k**2 * ca.fabs(kappa_grid[:-1])

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

  
        # rk4 integration
       
        def get_derivatives(v_val):
            
            f_drag_step = Car.get_aero_drag(self.car, v_val, cd=cd_eff)
            f_net_step = F_long_k - f_drag_step
            acc_step = f_net_step / p_weight
            
            dv_ds = acc_step / v_val
            dT_ds = 1.0 / v_val
            
            dE_ds = (F_acc / self.car.power_eff - (-F_brake_regen * self.car.regen_eff * 0.9)) / 3.6e6
            
            return dv_ds, dT_ds, dE_ds

        
        k1_v, k1_T, k1_E = get_derivatives(v_k)
        k2_v, k2_T, k2_E = get_derivatives(v_k + 0.5 * ds * k1_v)
        k3_v, k3_T, k3_E = get_derivatives(v_k + 0.5 * ds * k2_v)
        k4_v, k4_T, k4_E = get_derivatives(v_k + ds * k3_v)

       
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
            (self.car.M_max * self.car.i * 4) < 1
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
        opti.subject_to(v >= 1.0)  

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

        F_norm_factor = (self.car.weight * self.car.g)**2
        J_overlap = 0.004 * ca.sum1((F_acc * -F_brake) / F_norm_factor)

        J_final_v = 0.1 * v[-1]

        J_drs_binary = 0.002 * ca.sum1(drs * (1 - drs))

        opti.minimize(J_Time + J_overlap - J_final_v)

        # =========================
        # Initial guess
        # =========================
        if init_guess is not None:
            opti.set_initial(v, 0.9*init_guess['v'])
            opti.set_initial(F_acc, 0.9*init_guess['F_acc'])
           
            
        else:
            opti.set_initial(v, 15.0)
            opti.set_initial(F_acc, 100.0)

        # =========================
        # Solver setup
        # =========================
        p_opts = {"expand": True}
        s_opts = {"max_iter": 2000, "tol": 1e-2, "print_level": 5}

        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()

            # =========================
            # Sensitivity analysis
            # =========================
            L = opti.f + ca.dot(opti.lam_g, opti.g)

            sens_w   = sol.value(ca.jacobian(L, p_weight))
            sens_cl  = sol.value(ca.jacobian(L, p_cl))
            sens_cd  = sol.value(ca.jacobian(L, p_cd))
         

            print("sensitivity results")
            print(f"Weight: {sens_w:+.4f} s/kg")
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
                    "dTime_dm": sens_w,
                    "dTime_dCL": sens_cl,
                    "dTime_dCdA": sens_cd,
                }
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return SolverResults(False, 0, [], [], [], [], [], [], {})