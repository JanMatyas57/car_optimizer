import casadi as ca
import numpy as np
from src.core.car import Car
from src.optimization.solvers.base_solver import BaseSolver, SolverResults, InitConditions


class MassPointSolver(BaseSolver):
    DISPLAY_NAME = "Mass point (Rk4)"
    def __init__(self, car, track):
        super().__init__(car, track)
        self.simulate_full_battery = False


    def solve(self, init_cond: InitConditions, init_guess=None, end_v= None,ds=0.5):

        if self.simulate_full_battery and self.car.soc_start == 1.0:
            #prevent battery bspline evaluation outside of defined range 
            self.car.soc_start = 0.999

        opti = ca.Opti()

        #parameters
        p_weight = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_rotating_mass = opti.parameter()
     

        # Set params values
        opti.set_value(p_weight, self.car.weight)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_rotating_mass, self.car.rotating_mass)

        # Track data
        total_length = self.track.length
        N = int(np.round(total_length / ds))

        s_sim = np.linspace(0, total_length, N + 1)
        ds = s_sim[1:] - s_sim[:-1]

        kappa_grid = self.track.get_curvature(s_sim)
        
        # States
        v = opti.variable(N + 1)   # speed [m/s]
        T = opti.variable(N + 1)   # time [s]
        E = opti.variable(N + 1)   # energy used [kWh] (mechanical)
        SoC_state = opti.variable(N + 1)   # battery state of charge [-]
        E_pack = opti.variable(N + 1)      # electrical energy from pack [kWh]
        
        
        bat = self.car.battery if getattr(self, "simulate_full_battery", False) else None
        soc_power_mode = getattr(self, "soc_power_mode", "lookup_table")

        if bat is not None:   
            U1        = opti.variable(N + 1)   # RC1 voltage   [V/cell]
            U2        = opti.variable(N + 1)   # RC2 voltage   [V/cell]

        # Inputs
        F_acc   = opti.variable(N)   
        F_brake_mech = opti.variable(N)
        F_brake_regen = opti.variable(N)
        F_brake = F_brake_mech + F_brake_regen
        drs     = opti.variable(N)

        # DRS (we define it everywhere, but the decision variable is only active in allowed zones)
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


        F_max_x, F_max_y = Car.get_friction_limit(
            self.car, v_k,
            weight=p_weight,    
            use_cl=cl_eff
        )

        # Lateral Force: F_lat = m * v^2 * curvature
        F_lat = p_weight * v_k**2 * ca.fabs(kappa_grid[:-1])

        # Longitudinal force
        F_long_k = F_acc + F_brake

        # Pedal constraints
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake_regen <= 0)
        opti.subject_to(F_brake_mech <= 0)

        # Friction ellipse
        term_long = (F_long_k / F_max_x)**2
        term_lat  = (F_lat    / F_max_y)**2
        opti.subject_to(term_long + term_lat <= 1)

        
        P_recuperated = -F_brake_regen*v_k

        # Recuperation limit at given SoC (from battery lookup table)
        if soc_power_mode == "lookup_table":
            recuperation_max = self.car.get_recuperation_limit(SoC_state[:-1])
            opti.subject_to(P_recuperated <= recuperation_max)
    
        
        opti.subject_to((-F_brake_mech*v_k)/self.car.brake_max <= 1)
        opti.subject_to(F_brake/self.car.brake_max <= 1)

  
        # vehicle dynamics integration (rk4) and battery states (euler forward)
        def get_derivatives(v_val):
            
            f_drag_step = Car.get_aero_drag(self.car, v_val, cd=cd_eff)
            f_net_step = F_long_k - f_drag_step
            m_eff = p_weight + p_rotating_mass
            acc_step = f_net_step / m_eff
            
            dv_ds = acc_step / v_val
            dT_ds = 1.0 / v_val
            dE_ds = (F_acc / self.car.power_eff - (-F_brake_regen * self.car.regen_eff * 0.9)) / 3.6e6

            return dv_ds, dT_ds, dE_ds

        # RK4 for vehicle dynamics (same for both battery / no battery)
        k1_v, k1_T, k1_E = get_derivatives(v_k)
        k2_v, k2_T, k2_E = get_derivatives(v_k + 0.5 * ds * k1_v)
        k3_v, k3_T, k3_E = get_derivatives(v_k + 0.5 * ds * k2_v)
        k4_v, k4_T, k4_E = get_derivatives(v_k + ds * k3_v)

        v_next = v_k    + (ds / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        T_next = T[:-1] + (ds / 6.0) * (k1_T + 2*k2_T + 2*k3_T + k4_T)
        E_next = E[:-1] + (ds / 6.0) * (k1_E + 2*k2_E + 2*k3_E + k4_E)

        opti.subject_to(v[1:] == v_next)
        opti.subject_to(T[1:] == T_next)
        opti.subject_to(E[1:] == E_next)

        SoC = SoC_state[:-1]

        # Euler forward for battery states (only when 2RC model is active)
        if bat is not None:
            U1_k = U1[:-1]
            U2_k = U2[:-1]

            # Cell current as optimization variable (solved via power balance)
            I_cell_k = opti.variable(N)

            # Pack power demanded by drivetrain
            P_pack_k = (F_acc * v_k) / self.car.power_eff \
                     - (-F_brake_regen * v_k) * self.car.regen_eff

            # 2RC terminal voltage
            OCV_k  = bat.OCV(SoC)                                         # [V/cell]
            R0_k   = bat.get_R0(SoC)
            U_cell_k = OCV_k - I_cell_k * R0_k - U1_k - U2_k             # [V/cell]

            # Power balance: what drivetrain demands = what pack delivers at terminals
            P_terminal_k = bat.n_series * U_cell_k * bat.n_parallel * I_cell_k  # [W]
            opti.subject_to(P_terminal_k == P_pack_k)

            # SoC integration (Euler forward, path domain)
            dSoC_ds = (-bat.eta * I_cell_k / (bat.Q * 3600.0)) / v_k
            opti.subject_to(SoC_state[1:] == SoC + ds * dSoC_ds)

            # Lookup RC parameters at step start SoC
            R1_k   = bat.get_R1(SoC)
            R2_k   = bat.get_R2(SoC)
            tau1_k = bat.get_tau1(SoC)
            tau2_k = bat.get_tau2(SoC)

            # RC voltage derivatives (path domain)
            dU1_ds = (-U1_k / tau1_k + I_cell_k * R1_k / tau1_k) / v_k
            dU2_ds = (-U2_k / tau2_k + I_cell_k * R2_k / tau2_k) / v_k

            # Euler step
            opti.subject_to(U1[1:]        == U1_k + ds * dU1_ds)
            opti.subject_to(U2[1:]        == U2_k + ds * dU2_ds)

            # Bounds on RC voltages to help solver convergence
            opti.subject_to(opti.bounded(-2, U1, 2))
            opti.subject_to(opti.bounded(-2, U2, 2))

            # E_pack integration: true electrical energy from pack
            dE_pack_ds = P_terminal_k / (v_k * 3.6e6)  # [kWh/m]
            opti.subject_to(E_pack[1:] == E_pack[:-1] + ds * dE_pack_ds)

            # Limit regen current to 55 A per cell
            opti.subject_to(I_cell_k >= -55.0)

            
            if soc_power_mode == "optimum":
                opti.subject_to(U_cell_k >= 3.45)
                opti.subject_to(U_cell_k <= bat.max_cell_voltage)

        else:
            # No battery model: SoC derived algebraically from mechanical E
            E_left = self.car.acp_capacity - E[:-1]
            SoC_algebraic = E_left / self.car.acp_capacity
            opti.subject_to(SoC_state[:-1] == SoC_algebraic)
            opti.subject_to(SoC_state[-1] >= 0.0)
            # E_pack is zero (no battery model to integrate)
            opti.subject_to(E_pack == 0.0)

        # Torque limit (30 Nm per wheel)    
        opti.subject_to(
            ((self.car.wheel_radius * F_acc)) /
            (self.car.M_max * self.car.i * 4) < 1
        )

       
        # Power limit
        P_out = (F_acc * v_k) / self.car.power_eff

        soc_power_mode = getattr(self, "soc_power_mode", "lookup_table")
        if soc_power_mode == "lookup_table":
            # Limit from SoC (dynamic)
            P_max_dynamic = self.car.soc_to_power_interp(SoC)
            opti.subject_to(P_out <= P_max_dynamic)

        # Hard limit (motor/inverter max)
        opti.subject_to(P_out <= self.car.power_max)

        # Bounding SoC
        opti.subject_to(SoC >= 0.0)

        # Speed limits
        opti.subject_to(v >= 1.0)  

       
        # Initial conditions
        opti.subject_to(v[0] == init_cond.v)
        opti.subject_to(T[0] == init_cond.T)
        opti.subject_to(E[0] == init_cond.E)
        opti.subject_to(SoC_state[0] == self.car.soc_start)
        opti.subject_to(E_pack[0] == 0.0)  # no energy consumed at start
        
        opti.subject_to(drs[0] == 0)  # start with DRS closed

        if bat is not None:
            opti.subject_to(U1[0] == 0.0)   # RC voltages relaxed at start
            opti.subject_to(U2[0] == 0.0)

    
        # Energy limitation
        if self.car.energy_limit is not None:
            opti.subject_to(E[-1]-init_cond.E <= self.car.energy_limit)

   
        # Objective function
        J_Time = T[-1]

        F_norm_factor = (self.car.weight * self.car.g)**2
        J_overlap = 0.004 * ca.sum1((F_acc * -F_brake) / F_norm_factor)

        J_final_v = 0.1 * v[-1]

        J_drs_binary = 0.002 * ca.sum1(drs * (1 - drs))

        opti.minimize(J_Time + J_overlap - J_final_v)

    
        # Initial guess
        if init_guess is not None:
            opti.set_initial(v, init_guess['v'])
            opti.set_initial(F_acc, init_guess['F_acc'])
            opti.set_initial(SoC_state, self.car.soc_start)
            opti.set_initial(E_pack, 0.0)

            if bat is not None:
                opti.set_initial(I_cell_k, 10.0)
                opti.set_initial(U1, 0.0)
                opti.set_initial(U2, 0.0)
        else:
            opti.set_initial(v, 15.0)
            opti.set_initial(F_acc, 100.0)
            opti.set_initial(SoC_state, self.car.soc_start)
            opti.set_initial(E_pack, 0.0)

            if bat is not None:
                opti.set_initial(I_cell_k, 10.0)
                opti.set_initial(U1, 0.0)
                opti.set_initial(U2, 0.0)

        if end_v is not None:
            opti.subject_to(v[N] == end_v)

     
        # Solver setup
        p_opts = {"expand": True}
        s_opts = {"max_iter": 2000, "tol": 1e-2, "print_level": 5}

        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()

            
            L = opti.f + ca.dot(opti.lam_g, opti.g)

            sens_w   = sol.value(ca.jacobian(L, p_weight))
            sens_cl  = sol.value(ca.jacobian(L, p_cl))
            sens_cd  = sol.value(ca.jacobian(L, p_cd))
            sens_r_m = sol.value(ca.jacobian(L, p_rotating_mass))
         

            print("sensitivity results")
            print(f"Weight: {sens_w:+.4f} s/kg")
            print(f"CL:   {sens_cl:+.4f} s/unit")
            print(f"CdA:  {sens_cd:+.4f} s/unit")
            print(f"Rotating mass: {sens_r_m:+.4f} s/kg")

            if bat is not None:
                # Extract battery results from solved values 
                SoC_solved = sol.value(SoC)
                U1_solved  = sol.value(U1)
                U2_solved  = sol.value(U2)
                I_solved   = sol.value(I_cell_k)

                # Terminal voltage: Ut = n_s*(OCV - I*R0 - U1 - U2)
                import numpy as np_solve
                OCV_solved = np_solve.array([float(bat.OCV(s)) for s in SoC_solved])
                R0_solved  = np_solve.array([float(bat.get_R0(s)) for s in SoC_solved])
                U_t_solved = bat.n_series * (OCV_solved - I_solved * R0_solved - U1_solved[:-1] - U2_solved[:-1])

                bat_results = dict(
                    U1=U1_solved,
                    U2=U2_solved,
                    U_terminal=U_t_solved,
                    I_cell=I_solved,
                )
            else:
                bat_results = dict(U1=None, U2=None, U_terminal=None, I_cell=None)

            return SolverResults(
                success=True,
                lap_time=sol.value(T[-1]),
                t=sol.value(T),
                v=sol.value(v),
                s=s_sim,
                E=sol.value(E),
                E_pack=sol.value(E_pack),
                SoC=sol.value(SoC)*100,
                f_acc=sol.value(F_acc) / self.car.power_eff,
                f_brake=sol.value(F_brake),
                recuperation=sol.value(P_recuperated*self.car.regen_eff),
                drs_usage=sol.value(drs_ef),
                sens={
                    "dTime_dm": sens_w,
                    "dTime_dCL": sens_cl,
                    "dTime_dCdA": sens_cd,
                    "dTime_dRm": sens_r_m,
                },
                **bat_results,
            )

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return SolverResults(False, 0, [], [], [], [], [], [], {})