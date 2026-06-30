import casadi as ca
import numpy as np
from src.core.car import Car
from src.optimization.solvers.base_solver import BaseSolver, SolverResults, InitConditions

# does not work, kept for reference only (was too slow and didn't converge well)
class SingleTrackSolver(BaseSolver):
    DISPLAY_NAME = "Single Track"
    def __init__(self, car, track, laps=1):
        super().__init__(car, track)

    def solve(self, init_cond: InitConditions, init_guess=None, ds=1.0):
        opti = ca.Opti()

        # parameters
        p_mass = opti.parameter()
        p_cl   = opti.parameter()
        p_cd   = opti.parameter()
        p_mu_x = opti.parameter()
        p_mu_y = opti.parameter()

        # set parameter values
        opti.set_value(p_mass, self.car.mass)
        opti.set_value(p_cl,   self.car.cl)
        opti.set_value(p_cd,   self.car.cd)
        opti.set_value(p_mu_x, self.car.mu_x)
        opti.set_value(p_mu_y, self.car.mu_y)

        # track data and grid
        total_length = self.track.length
        N = int(np.round(total_length / ds))
        
        s_sim = np.linspace(0, total_length, N + 1)
        ds_val = s_sim[1] - s_sim[0] # actual uniform step
        
        # curvature along the path
        kappa_grid = self.track.get_curvature(s_sim[:-1]) # curvature at steps 0..n-1

        # states at k=0..n (n+1 points)
        vx  = opti.variable(N + 1) # longitudinal speed
        vy  = opti.variable(N + 1) # lateral speed
        r   = opti.variable(N + 1) # yaw rate
        n   = opti.variable(N + 1) # lateral deviation
        xi  = opti.variable(N + 1) # heading error
        t   = opti.variable(N + 1) # time
        E   = opti.variable(N + 1) # energy

        # controls at k=0..n-1 (n points)
        delta   = opti.variable(N) # steering angle
        F_acc   = opti.variable(N) # drive force
        F_brake = opti.variable(N) # brake force
        drs     = opti.variable(N) # drs state

        # helper constraints
        track_width_limit = 0.5 
        opti.subject_to(opti.bounded(-track_width_limit, n, track_width_limit))

        # steering limits
        opti.subject_to(opti.bounded(-self.car.steeringAngle_max, delta, self.car.steeringAngle_max))

        # speed limit to avoid singularity at 0
        opti.subject_to(vx >= 1.0) 

        # pedal constraints
        opti.subject_to(F_acc >= 0)
        opti.subject_to(F_brake <= 0)

        # drs logic
        use_drs_vec = False

        if self.car.drs_allowed and self.track.drs is not None:
            drs_flag = self.track.get_drs_flag(s_sim[:-1])
            drs_ef = drs * drs_flag
            
            cl_eff = p_cl + drs_ef * (self.car.cl_drs - p_cl)
            cd_eff = p_cd + drs_ef * (self.car.cd_drs - p_cd)
            use_drs_vec = True
        else:
            drs_ef = ca.DM.zeros(N)
            cl_eff = p_cl
            cd_eff = p_cd
            opti.subject_to(drs == 0)
            use_drs_vec = False

        opti.subject_to(drs >= 0)
        opti.subject_to(drs <= 1)

        # slip limit for linear model validity (5 degrees in rad)
        alpha_limit = 5.0 * (np.pi / 180.0)

        for k in range(N):
            # current states
            vx_k = vx[k]
            vy_k = vy[k]
            r_k  = r[k]
            n_k  = n[k]
            xi_k = xi[k]
            t_k  = t[k]
            
            # current inputs
            delta_k   = delta[k]
            F_acc_k   = F_acc[k]
            F_brake_k = F_brake[k]
            
            curv_k = kappa_grid[k]

            # force distribution
            Ff_acc = self.car.power_dist * F_acc_k
            Fr_acc = (1.0 - self.car.power_dist) * F_acc_k
            
            Ff_brk = self.car.brake_bias * F_brake_k
            Fr_brk = (1.0 - self.car.brake_bias) * F_brake_k
            
            Fx_f = Ff_acc + Ff_brk
            Fx_r = Fr_acc + Fr_brk

            # normal loads (fz)
            if use_drs_vec:
                cl_k = cl_eff[k]
                cd_k = cd_eff[k]
            else:
                cl_k = cl_eff 
                cd_k = cd_eff

            # fz calculations
            fz_aero = 0.5 * self.car.rho * cl_k * vx_k**2
            fz_aero_f = fz_aero * (1 - self.car.cop)
            fz_aero_r = fz_aero * self.car.cop
            
            Fz_f = self.car.fz_static_f + fz_aero_f
            Fz_r = self.car.fz_static_r + fz_aero_r

            # slip angles kinematics
            # front slip
            vy_f_kin = (vy_k + self.car.lf * r_k)
            alpha_f = -ca.atan2(
                vy_f_kin * ca.cos(delta_k) - vx_k * ca.sin(delta_k),
                vy_f_kin * ca.sin(delta_k) + vx_k * ca.cos(delta_k)
            )

            # rear slip
            vy_r_kin = (vy_k - self.car.lr * r_k)
            alpha_r = -ca.atan2(vy_r_kin, vx_k)

            # lateral forces
            Fy_f_lin = alpha_f * (36.0 / np.pi) * Fz_f * p_mu_y
            Fy_r_lin = alpha_r * (36.0 / np.pi) * Fz_r * p_mu_y

            # constraints
            # hard limit on slip angle to keep linear model valid
            opti.subject_to(opti.bounded(-alpha_limit, alpha_f, alpha_limit))
            opti.subject_to(opti.bounded(-alpha_limit, alpha_r, alpha_limit))

            # friction ellipse 
            mu_x_fz_f = p_mu_x * Fz_f
            mu_y_fz_f = p_mu_y * Fz_f
            
            mu_x_fz_r = p_mu_x * Fz_r
            mu_y_fz_r = p_mu_y * Fz_r

            ellipse_f = (Fx_f / mu_x_fz_f)**2 + (Fy_f_lin / mu_y_fz_f)**2
            ellipse_r = (Fx_r / mu_x_fz_r)**2 + (Fy_r_lin / mu_y_fz_r)**2
            
            opti.subject_to(ellipse_f <= 1.0)
            opti.subject_to(ellipse_r <= 1.0)

            # power limit: P = F * v <= P_max
            opti.subject_to( ((F_acc_k * vx_k) / self.car.power_eff)/self.car.power_max <= 1 )
            
            # regen limit
            opti.subject_to( ((-F_brake_k * vx_k) * self.car.regen_eff)/self.car.brake_max <= 1 )

            # drag
            F_drag = 0.5 * self.car.rho * cd_k * vx_k**2 + self.car.fix_drag

            # body frame forces
            F_x_total = Fx_f * ca.cos(delta_k) - Fy_f_lin * ca.sin(delta_k) + Fx_r - F_drag
            F_y_total = Fx_f * ca.sin(delta_k) + Fy_f_lin * ca.cos(delta_k) + Fy_r_lin
            
            # moment
            M_z = self.car.lf * (Fx_f * ca.sin(delta_k) + Fy_f_lin * ca.cos(delta_k)) - self.car.lr * Fy_r_lin

            # time derivatives
            dvx_dt = F_x_total / p_mass + r_k * vy_k
            dvy_dt = F_y_total / p_mass - r_k * vx_k
            dr_dt  = M_z / self.car.Iz
            
            # path scaling factor Sf = dt/ds
            s_dot_denom = (1.0 - n_k * curv_k)
            s_dot_num   = (vx_k * ca.cos(xi_k) - vy_k * ca.sin(xi_k))
            
            # avoid division by zero
            ds_dt = s_dot_num / ca.fmax(s_dot_denom, 0.1)
            Sf = 1.0 / ds_dt  # dt/ds

            # spatial derivatives
            dvx_ds = dvx_dt * Sf
            dvy_ds = dvy_dt * Sf
            dr_ds  = dr_dt  * Sf
            dt_ds  = Sf
            
            # kinematics in path coordinates
            dn_ds = (vx_k * ca.sin(xi_k) + vy_k * ca.cos(xi_k)) * Sf
            dxi_ds = r_k * Sf - curv_k

            # energy
            dE_ds = ((F_acc_k / self.car.power_eff + F_brake_k * self.car.regen_eff) * Sf) / 3.6e6

            # euler step
            opti.subject_to(vx[k+1] == vx_k + dvx_ds * ds_val)
            opti.subject_to(vy[k+1] == vy_k + dvy_ds * ds_val)
            opti.subject_to(r[k+1]  == r_k  + dr_ds  * ds_val)
            opti.subject_to(n[k+1]  == n_k  + dn_ds  * ds_val)
            opti.subject_to(xi[k+1] == xi_k + dxi_ds * ds_val)
            opti.subject_to(t[k+1]  == t_k  + dt_ds  * ds_val)
            opti.subject_to(E[k+1] == E[k] + dE_ds * ds_val)

        # boundary conditions
        opti.subject_to(vx[0] == init_cond.v)
        opti.subject_to(vy[0] == 0) # 0 lateral speed at start
        opti.subject_to(r[0]  == 0) # 0 yaw rate at start
        opti.subject_to(n[0]  == init_cond.n)
        opti.subject_to(xi[0] == init_cond.xi)
        opti.subject_to(t[0]  == init_cond.T)
        opti.subject_to(E[0]  == init_cond.E)

        # energy limit
        if self.car.energy_limit is not None:
            opti.subject_to(E[-1] <= self.car.energy_limit)

        # objective function
        J_Time = t[-1]
        
        # regularization terms
        # steering rate for smooth steering
        diff_delta = delta[1:] - delta[:-1]
        J_steer_rate = 7e-4 * ca.sumsqr(diff_delta)
        
        # acceleration smoothness
        diff_acc = F_acc[1:] - F_acc[:-1]
        J_acc_smooth = 1e-3 * ca.sumsqr(diff_acc) / 1000.0

        # path deviation
        J_path = 0.5 * ca.sumsqr(n)

        opti.minimize(J_Time + J_path + J_steer_rate)

        # initial guess 
        g_vx = init_guess['v']
        g_n  = init_guess['n']
        g_xi = init_guess['xi']
        g_r  = init_guess['r']
        g_t  = init_guess['t']
        g_delta = init_guess['delta']
        g_facc  = init_guess['F_acc']
        g_fbrk  = init_guess['F_brake']

        # set initial values
        L_state = min(len(g_vx), N + 1)
        opti.set_initial(vx[:L_state], g_vx[:L_state])
        opti.set_initial(vy[:L_state], 0.0) # 0 slip initial guess
        opti.set_initial(n[:L_state],  g_n[:L_state])
        opti.set_initial(xi[:L_state], g_xi[:L_state])
        opti.set_initial(r[:L_state],  g_r[:L_state])
        opti.set_initial(t[:L_state],  g_t[:L_state])
        
        L_ctrl = min(len(g_delta), N)
        opti.set_initial(delta[:L_ctrl], g_delta[:L_ctrl])
        opti.set_initial(F_acc[:L_ctrl], g_facc[:L_ctrl])
        opti.set_initial(F_brake[:L_ctrl], g_fbrk[:L_ctrl])

        # solver config
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
            
            # post processing and sensitivities
            L_lagrange = opti.f + ca.dot(opti.lam_g, opti.g)
            sens_m = sol.value(ca.jacobian(L_lagrange, p_mass))
            sens_cl = sol.value(ca.jacobian(L_lagrange, p_cl))
        
            res_vx = sol.value(vx)
            res_vy = sol.value(vy)
            res_r  = sol.value(r)
            res_delta = sol.value(delta)
            
            # recompute slip angles for output visualization
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
                v=sol.value(vx),
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
            return SolverResults(False, 0.0, [], [], [], [], [], [], {}, mu_x=self.car.mu_x, mu_y=self.car.mu_y)