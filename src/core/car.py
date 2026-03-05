from dataclasses import dataclass,field
import numpy as np
import casadi as ca


@dataclass
class Car:
    # --- 1. Parameters from UI ---
    weight: float           # [kg]
    cd: float             # [-] 
    cl: float             # [-]
    cd_drs: float 
    cl_drs: float
    acp_capacity: float
    drs_allowed: bool
    power_max: float 
    regen_eff: float 
    soc_start: float 
    cop: float 
    energy_limit: float = None  # [kWh] 
    performance_delta: float = 0.0

    # Calculated fields 
    fz_static_f: float = field(init=False)
    fz_static_r: float = field(init=False)
    cs_static_f : float = field(init=False)
    cs_static_r : float = field(init=False)

    model_type: str = "mass_point"

    # Constants
    rho = 1.225
    g = 9.81

    # Powertrain
    brake_max = 60000
    power_eff = 0.88
    fix_drag = 230
    wheel_radius = 0.205
    max_torque = 30.0
    gear_ratio = 11.46
    brake_bias = 0.55
    power_dist = 0.5
    
    #grip
    mu_x: float = 1.25           # [-] 
    mu_y: float = 1.25           # [-] 
    

    #soc limitations
    soc_grid_power = [0.0, 0.04, 0.08, 0.12, 0.22, 0.32, 0.42, 0.52, 0.70, 1.00,1.1]
    power_limit_W = [10e3, 14e3, 21e3, 35e3, 62e3, 79e3, 89e3, 95e3, 100e3, 100e3,100e3]
    soc_to_power_interp = ca.interpolant('soc_power', 'bspline', [soc_grid_power], power_limit_W)

    # Suspension
    cogr = 0.5 
    cogz = 0.3
    sus_t = 0.2
    Iz = 110 
    wheelbase = 1.53 
    weight_distrib = 0.46 # to rear
    steeringAngle_max = 25/180*ca.pi;
    
  
    lf = wheelbase * cogr
    lr = wheelbase - lf
    
    tire_model = { 
        "cold_factor": 0.8, 
        "warmup_tau": 1.6, 
        "noise_std": 0.03, 
        "mu_x_nominal": 1.22,
        "mu_y_nominal":1.22
    }

    def __post_init__(self):
        # calculate static weight
        self.fz_static_f = self.weight * self.g * (1 - self.weight_distrib)
        self.fz_static_r = self.weight * self.g * self.weight_distrib
        #self.cs_static_f = self.get_cornering_stiffness(self.fz_static_f)
        #self.cs_static_r = self.get_cornering_stiffness(self.fz_static_r)
        #print(f"cornering stiffness front: {self.cs_static_f}")
        self.cs_static_f = 2 * self.get_cornering_stiffness(self.fz_static_f/2)
        self.cs_static_r = 2 * self.get_cornering_stiffness(self.fz_static_r/2)


    # car calculation functions 
    def get_aero_drag(self, v, cd = None):
        #if we pass cd as optimization parameter
        use_cd = cd if cd is not None else self.cd
        return self.fix_drag + 0.5 * self.rho * use_cd * v**2
    

    def get_recuperation_limit(self, SoC_ca):
        """
        Vrací spojitý (hladký) limit rekuperace ve Wattech pro dané SoC.
        """
        P_max = 40e3   # 40 kW limit
        soc_mid = 0.885 # Kde křivka protíná 20 kW
        k = 120.0       # Strmost poklesu
        
        return P_max / (1.0 + ca.exp(k * (SoC_ca - soc_mid)))
    
    def tire_fy(self, alpha, Fz):
            """
            Linear tire model
            Fy = (alpha/pi*180)/5 * Fz * tire_fyFactor
            """
            return (alpha * 180.0 / ca.pi) / 5.0 * Fz * self.mu_y
    
    def split_long_forces(self, F_acc, F_brake):
        """
        Inputs:
          F_acc   >= 0
          F_brake <= 0
        Returns:
          Ff, Fr  (front/rear longitudinal forces)
        """
        Ff_acc = self.power_dist * F_acc
        Fr_acc = (1.0 - self.power_dist) * F_acc

        Ff_brk = self.brake_bias * F_brake
        Fr_brk = (1.0 - self.brake_bias) * F_brake

        Ff = Ff_acc + Ff_brk
        Fr = Fr_acc + Fr_brk

        return Ff, Fr
    
        
    def get_friction_limit(self, v, weight=None, use_cl=None):
        weight = weight if weight is not None else self.weight
        use_cl = use_cl if use_cl is not None else self.cl

        F_aero = 0.5 * self.rho * use_cl * v**2

        Fz = weight * self.g + F_aero
        
        # OPRAVA: Tady musí být násobení self.mu_x a self.mu_y, nikoliv weight!
        max_x = self.mu_x * Fz
        max_y = self.mu_y * Fz
        
        return (max_x, max_y)
    

    def get_fz(self,v,cla_ef):
        #for single track, returns Fz on front and rear
        fz_aero = 0.5*self.rho*cla_ef*v**2
        fz_aero_front = fz_aero*(1-self.cop)
        fz_aero_rear = fz_aero*self.cop

        fz_f = self.fz_static_f + fz_aero_front
        fz_r = self.fz_static_r + fz_aero_rear

        return fz_f,fz_r
    
    def get_tire_dy(self,fx,fz):

        #save option
        #term = (self.mu_y * fz)**2 - fx**2
        #term_safe = ca.log(1 + ca.exp(term))
        #dy =ca.sqrt(term_safe)
        max_lateral_sq = (self.mu_y * fz)**2 - fx**2
        # Použijeme fmax, aby pod odmocninou bylo minimálně 1e-6
        dy = ca.sqrt(ca.fmax(max_lateral_sq, 1))
        return dy

        #dy = ca.sqrt((self.mu_y*fz)**2 -fx**2) 
        #return dy
    
    def get_cornering_stiffness(self, fz):
       
        PKY1 = 19.7276 
        PKY2 = 3.3304
        PKY4 = 7.5668
        FNOMIN = 2000  # Z tir souboru

        # Pacejka vzorec pro Stiffness (bez Camberu)
        # C_alpha = PKY1 * FNOMIN * sin(PKY4 * arctan(Fz / (PKY2 * FNOMIN)))
        
        term_inner = fz / (PKY2 * FNOMIN)
        c_alpha = PKY1 * FNOMIN * ca.sin(PKY4 * ca.arctan(term_inner))
    
        return c_alpha
    
    
    def update_regen_usage(self, base_drift=-0.0015, sigma=0.02):
    
        noise = np.random.normal(0.0, sigma)
        self.regen_eff += np.round(base_drift + noise,2)
        self.regen_eff = round(np.clip(self.regen_eff, 0.5, 1.0),2)

    def update_grip(self, lap):
        
        #tire warmup
        warmup_progress = 1 - np.exp(-(lap + 1) / self.tire_model["warmup_tau"])

        mu_x_cold = self.tire_model["cold_factor"] * self.tire_model["mu_x_nominal"]
        current_base_mu_x = mu_x_cold + (self.tire_model['mu_x_nominal'] - mu_x_cold) * warmup_progress
        
        # Y složka
        mu_y_cold = self.tire_model["cold_factor"] * self.tire_model["mu_y_nominal"]
        current_base_mu_y = mu_y_cold + (self.tire_model['mu_y_nominal'] - mu_y_cold) * warmup_progress

        noise = self.tire_model["noise_std"] * np.random.randn()
      
        self.mu_x = np.round(current_base_mu_x + noise, 3)
        self.mu_y = np.round(current_base_mu_y + noise, 3)

     
        print(f"Lap {lap+1} Grip: Mu_X={self.mu_x}, Mu_Y={self.mu_y} (Warmup: {warmup_progress:.1%})")
        
       # return self.mu_x, self.mu_y

    def singletrack_time_ode(self, z, u, cl_eff=None, cd_eff=None):
            """
            Time-domain bicycle ODE.

            State:
            z = [vx, vy, psi, dpsi, X, Y]

            Input:
            u = [delta_f, F_acc, F_brake]

            Returns:
            dz/dt (CasADi expression)
            """

            vx   = z[0]
            vy   = z[1]
            psi  = z[2]
            r    = z[3]
            X    = z[4]
            Y    = z[5]

            delta_f = u[0]
            F_acc   = u[1]
            F_brake = u[2]

            # effective aero params 
            if cl_eff is None:
                cl_eff = self.cl
            if cd_eff is None:
                cd_eff = self.cd  

            # distribute long forces
            Ff, Fr = self.split_long_forces(F_acc, F_brake)

            # normal loads 
            Fz_f, Fz_r = self.get_fz(vx, cl_eff)

            # slip angles 
            alpha_f = -ca.atan2(
                (vy + self.lf * r) * ca.cos(delta_f) - vx * ca.sin(delta_f),
                (vy + self.lf * r) * ca.sin(delta_f) + vx * ca.cos(delta_f)
            )

            alpha_r = -ca.atan2(
                (vy - self.lr * r), vx
            )

            # lateral tire forces
            F_fy = self.tire_fy(alpha_f, Fz_f)
            F_ry = self.tire_fy(alpha_r, Fz_r)

            # aero drag (uses vx only, cd is CdA)
            F_drag = 0.5 * self.rho * cd_eff * vx**2 + self.fix_drag

            # dynamics (Matlab style)
            dvx = (Ff * ca.cos(delta_f) - F_fy * ca.sin(delta_f) + Fr - F_drag) / self.mass + r * vy
            dvy = (Ff * ca.sin(delta_f) + F_fy * ca.cos(delta_f) + F_ry) / self.mass - r * vx

            dpsi = r
            dr   = (self.lf * (Ff * ca.sin(delta_f) + F_fy * ca.cos(delta_f)) - self.lr * F_ry) / self.Iz

            dX = vx * ca.cos(psi) - vy * ca.sin(psi)
            dY = vx * ca.sin(psi) + vy * ca.cos(psi)

            return ca.vertcat(dvx, dvy, dpsi, dr, dX, dY)
    
    def singletrack_path_ode(self, s, z, u, track, cl_eff=None, cd_eff=None):
        """
        Path-domain bicycle ODE.

        Independent variable: s

        State:
          z = [vx, vy, psi, dpsi, n, t]

        Input:
          u = [delta_f, F_acc, F_brake]

        Track must provide interpolation:
          th(s) = track heading
          C(s)  = curvature

        Returns:
          dz/ds
        """

        vx   = z[0]
        vy   = z[1]
        psi  = z[2]
        r    = z[3]
        n    = z[4]
        t    = z[5]

        delta_f = u[0]
        F_acc   = u[1]
        F_brake = u[2]

        # effective aero params
        if cl_eff is None:
            cl_eff = self.cl
        if cd_eff is None:
            cd_eff = self.cd  # CdA

        # distribute long forces
        Ff, Fr = self.split_long_forces(F_acc, F_brake)

        # track geometry 
        th = track.heading(s)     # == track.psi(s)
        C  = track.curvature(s)   # == track.kappa(s)

        eps = psi - th

        
        denom = (vx * ca.cos(eps) - vy * ca.sin(eps))
        Sf = (1.0 - n * C) / denom

        # normal loads depend on vx only
        Fz_f, Fz_r = self.get_fz(vx, cl_eff)

        # slip angles 
        alpha_f = -ca.atan2(
            (vy + self.lf * r) * ca.cos(delta_f) - vx * ca.sin(delta_f),
            (vy + self.lf * r) * ca.sin(delta_f) + vx * ca.cos(delta_f)
        )

        alpha_r = -ca.atan2(
            (vy - self.lr * r), vx
        )

        # lateral tire forces
        F_fy = self.tire_fy(alpha_f, Fz_f)
        F_ry = self.tire_fy(alpha_r, Fz_r)

        # aero drag
        F_drag = 0.5 * self.rho * cd_eff * vx**2 + self.fix_drag

        # time-domain derivatives (same as time ODE)
        dvx_dt = (Ff * ca.cos(delta_f) - F_fy * ca.sin(delta_f) + Fr - F_drag) / self.mass + r * vy
        dvy_dt = (Ff * ca.sin(delta_f) + F_fy * ca.cos(delta_f) + F_ry) / self.mass - r * vx
        dpsi_dt = r
        dr_dt   = (self.lf * (Ff * ca.sin(delta_f) + F_fy * ca.cos(delta_f)) - self.lr * F_ry) / self.Iz

        # path coordinates dynamics
        dn_dt = vx * ca.sin(eps) + vy * ca.cos(eps)
        dt_dt = 1.0

        # convert dt derivatives to ds derivatives
        dvx_ds  = Sf * dvx_dt
        dvy_ds  = Sf * dvy_dt
        dpsi_ds = Sf * dpsi_dt
        dr_ds   = Sf * dr_dt
        dn_ds   = Sf * dn_dt
        dt_ds   = Sf * dt_dt

        return ca.vertcat(dvx_ds, dvy_ds, dpsi_ds, dr_ds, dn_ds, dt_ds)




