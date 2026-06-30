from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import casadi as ca


@dataclass
class BatteryModel:
    """
    2RC equivalent circuit battery model loaded from a .npz file.
`
    Cell parameters:
      - Q      : cell capacity [Ah]
      - eta    : coulombic efficiency [-]
      - OCV    : open circuit voltage vs SoC 
      - R0     : ohmic resistance vs SoC 
      - R1,R2  : RC branch resistances vs SoC
      - tau1, tau2 : RC time constants vs SoC

    Pack topology is captured by n_series and n_parallel so that
    cell level quantities can be converted to pack quantities:
        U_pack   = n_series  * U_cell
        I_cell   = I_pack    / n_parallel
        Q_pack   = n_parallel * Q_cell   [Ah]
    """

    # ---- raw data ----
    Q: float                    # cell capacity [Ah]
    eta: float                  # coulombic efficiency [-]
    ocv_soc: np.ndarray         # OCV SoC, ascending
    ocv_u: np.ndarray           # OCV voltage values, [V/cell]
    soc_axis: np.ndarray        # parameter SoC breakpoints, ascending
    R0: np.ndarray              # ohmic resistance, [Ohm]
    R1: np.ndarray              # RC1 resistance, [Ohm]
    R2: np.ndarray              # RC2 resistance, [Ohm]
    tau1: np.ndarray            # RC1 time const,  [s]
    tau2: np.ndarray            # RC2 time const, [s]

    # pack topology (S series per stack, P parallel, N stacks)
    n_series: int = 1           # total cells in series  = S * N
    n_parallel: int = 1         # total cells in parallel = P

    #  voltage limit
    max_cell_voltage: float = 4.45  # [V] upper cell voltage for optimal control mode

   
    interp_OCV:  object = field(default=None, repr=False)
    interp_R0:   object = field(default=None, repr=False)
    interp_R1:   object = field(default=None, repr=False)
    interp_R2:   object = field(default=None, repr=False)
    interp_tau1: object = field(default=None, repr=False)
    interp_tau2: object = field(default=None, repr=False)

 
    @staticmethod
    def from_npz(path: str, n_series: int = None, n_parallel: int = None, max_cell_voltage: float = None) -> 'BatteryModel':
        """
        Load battery model from a .npz file.

        Expected keys: Q, eta, OCV_SOC, OCV_U, R0, R1, R2, tau1, tau2,
                       Param_SoC_axis

        Pack defaults for CTU26: S=15, P=2, N=9  →  n_series=135, n_parallel=2.
        """
        data = np.load(path)

        soc_axis_raw = data['Param_SoC_axis'].flatten().astype(float)
        sort_idx = np.argsort(soc_axis_raw)
        soc_axis = soc_axis_raw[sort_idx]

        def sorted_param(key):
            return data[key].flatten().astype(float)[sort_idx]

        pack_energy_kwh = None
        if 'pack_energy_kwh' in data:
            pack_energy_kwh = float(data['pack_energy_kwh'].flatten()[0])

        ns = n_series if n_series is not None else int(data.get('n_series', 135))
        np_ = n_parallel if n_parallel is not None else int(data.get('n_parallel', 2))
        v_max = max_cell_voltage if max_cell_voltage is not None else float(data.get('max_cell_voltage', 4.46))

        model = BatteryModel(
            Q          = float(data['Q']),
            eta        = float(data['eta']),
            ocv_soc    = data['OCV_SOC'].flatten().astype(float),
            ocv_u      = data['OCV_U'].flatten().astype(float),
            soc_axis   = soc_axis,
            R0         = sorted_param('R0'),
            R1         = sorted_param('R1'),
            R2         = sorted_param('R2'),
            tau1       = sorted_param('tau1'),
            tau2       = sorted_param('tau2'),
            n_series   = ns,
            n_parallel = np_,
            max_cell_voltage = v_max,
        )
        model._pack_energy_kwh = pack_energy_kwh
        return model

   
    def build_interpolants(self):
      
        ocv_soc = self.ocv_soc.tolist()
        soc_ax  = self.soc_axis.tolist()

        opts = {"inline": False}
        self.interp_OCV  = ca.interpolant('OCV',  'bspline', [ocv_soc], self.ocv_u.tolist(), opts)
        self.interp_R0   = ca.interpolant('R0',   'bspline', [soc_ax],  self.R0.tolist(), opts)
        self.interp_R1   = ca.interpolant('R1',   'bspline', [soc_ax],  self.R1.tolist(), opts)
        self.interp_R2   = ca.interpolant('R2',   'bspline', [soc_ax],  self.R2.tolist(), opts)
        self.interp_tau1 = ca.interpolant('tau1', 'bspline', [soc_ax],  self.tau1.tolist(), opts)
        self.interp_tau2 = ca.interpolant('tau2', 'bspline', [soc_ax],  self.tau2.tolist(), opts)


    def OCV(self, soc):
        """helper"""
        return self.interp_OCV(soc)

    def get_R0(self, soc):   return self.interp_R0(soc)
    def get_R1(self, soc):   return self.interp_R1(soc)
    def get_R2(self, soc):   return self.interp_R2(soc)
    def get_tau1(self, soc): return self.interp_tau1(soc)
    def get_tau2(self, soc): return self.interp_tau2(soc)

    @property
    def Q_pack_Ah(self) -> float:
        """Total pack capacity [Ah]."""
        return self.Q * self.n_parallel

    @property
    def energy_kwh(self) -> float:
        """Total energy capacity of the pack in kWh, needs to be computed from simulation"""
        if hasattr(self, '_pack_energy_kwh') and self._pack_energy_kwh is not None:
            return self._pack_energy_kwh
        # fall back to estimation: Q * n_series * n_parallel * mean_OCV / 1000.0
        v_nom = np.mean(self.ocv_u)
        return (self.Q * self.n_series * self.n_parallel * v_nom) / 1000.0



@dataclass
class Car:

    name: str
    weight: float           # [kg]
    cd: float             # drag coefficient (CdA) [m^2]
    cl: float             # lift coefficient (positive for downforce) [-]
    cd_drs: float # drag coefficient with drs closed
    cl_drs: float # drag coefficient with drs open
    drs_allowed: bool
    power_max: float # max power [W]
    regen_eff: float # [-] efficiency of regenerative braking
    soc_start: float # Batterty state of charge at start of simulation (0-1)
    cop: float   # center of pressure as % of wheelbase from front axle (0-1)
    wb: float    # [m] wheelbase
    tw: float    # [m] car track width
    wd: float    # [-] weight distribution as % of load on front axle (0-1)
    cogz: float   # [m] height of center of gravity
    i: float     #car gear ratio
    M_max: float  #max torque of the car [Nm]
    rotating_mass: float = 0.0  # [kg] equivalent rotating inertia (wheels, drivetrain, reduced to translation)
    energy_limit: float = None  # [kWh] 
    performance_delta: float = 0.0 # [-] performance modifier for strategy optimization, dont change

    battery: Optional['BatteryModel'] = field(default=None, repr=False)  # 2RC cell model

    model_type: str = "mass_point"

    # Constants
    rho = 1.225
    g = 9.81

    # Powertrain
    brake_max = 80000
    power_eff = 0.92
    fix_drag = 230
    wheel_radius = 0.205
    brake_bias = 0.55
    power_dist = 0.5
    v_max = 40 #not used for max bounds of GVV calcullation
    
    #grip
    mu_x: float = 1.25            
    mu_y: float = 1.25          
    

    #soc limitations (lookup table)
    soc_grid_power = [0.0, 0.04, 0.08, 0.12, 0.22, 0.32, 0.42, 0.52, 0.70, 1.00,1.1]
    power_limit_W = [10e3, 14e3, 21e3, 35e3, 62e3, 79e3, 89e3, 95e3, 100e3, 100e3,100e3]
    soc_to_power_interp = ca.interpolant('soc_power', 'bspline', [soc_grid_power], power_limit_W)

    

    tire_model = { 
        "cold_factor": 0.8, 
        "warmup_tau": 1.6, 
        "noise_std": 0.03, 
        "mu_x_nominal": 1.22,
        "mu_y_nominal":1.22
    }

    @property
    def acp_capacity(self) -> float:
        """Max usable ACP capacity in kWh, resolved from loaded battery model or defaulting to 6.0."""
        if self.battery is not None:
            return self.battery.energy_kwh
        return 6.0

    def load_battery(self, npz_path: str, battery_type: str = "ep85",
                     n_series: int = None, n_parallel: int = None):
        """
        Load a 2RC battery model from a .npz file and attach it to this Car.

        """
        self.battery = BatteryModel.from_npz(npz_path, n_series=n_series, n_parallel=n_parallel)
        self.battery.build_interpolants()
        print(f"Battery model loaded [{battery_type}]: Q_cell={self.battery.Q:.4f} Ah, "
              f"eta={self.battery.eta:.4f}, "
              f"pack={self.battery.n_series}s{self.battery.n_parallel}p, "
              f"Q_pack={self.battery.Q_pack_Ah:.2f} Ah, vmax={self.battery.max_cell_voltage:.2f} V, "
              f"acp_capacity={self.acp_capacity:.3f} kWh")

    # car calculation functions 
    def get_aero_drag(self, v, cd = None):
        #if we pass cd as optimization parameter
        use_cd = cd if cd is not None else self.cd
        return self.fix_drag + 0.5 * self.rho * use_cd * v**2
    

    def get_recuperation_limit(self, SoC_ca):
       
        P_max = 60e3   
        soc_mid = 0.885 
        k = 50.0       
        
        return P_max / (1.0 + ca.exp(k * (SoC_ca - soc_mid)))
    

    def tire_fy(self, alpha, Fz):
        # tire lateral force calculation for single track model
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
        
        dy = ca.sqrt(ca.fmax(max_lateral_sq, 1))
        return dy

        #dy = ca.sqrt((self.mu_y*fz)**2 -fx**2) 
        #return dy
    
    def get_cornering_stiffness(self, fz):
       
        PKY1 = 19.7276 
        PKY2 = 3.3304
        PKY4 = 7.5668
        FNOMIN = 2000  # Z tir souboru

     
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




