# src/core/track.py
import pandas as pd
import numpy as np
import casadi as ca
import os
from dataclasses import dataclass, field
from src.core.trackGenerator import TrackGenerator

@dataclass
class Track:
    name: str
    
    # Data for visualisation
    s: np.ndarray = field(repr=False)
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    z: np.ndarray = field(repr=False)
    drs: np.ndarray = field(repr=False)
    psi: np.ndarray = field(repr=False)
    kappa: np.ndarray = field(repr=False)
    alpha: np.ndarray = field(repr=False)

    # Telemetry from Real Log (Reference)
    speed: np.ndarray = field(repr=False) # m/s
    brake: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False) # kW
  
    length: float = 0.0
    
    # CasADi interpolant for solver
    kappa_interp: ca.Function = field(repr=False, default=None)
    psi_interp: ca.Function = field(repr=False,default=None)
    drs_interp: ca.Function = field(repr=False, default=None)

    @property
    def ref_data(self):
        """interface for validation module"""
        if self.speed is None:
            return None
            
        return {
            's': self.s,          # x reference
            'v': self.speed,      # speed [m/s]
            'p': self.power       # power [kW]
        }

    @classmethod
    def load(cls, track_name, tracks_dir="data/tracks"):
        """
        loads the .npz track for the simulation
        """
        filepath = os.path.join(tracks_dir, f"{track_name}.npz")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baked track '{track_name}' not found. Run baking first.")
            
        
        data = np.load(filepath)
        
        s, x, y = data['s'], data['x'], data['y']
        kappa = data['kappa']
        drs = data['drs'] if 'drs' in data else None
        
        psi = data['psi'] if 'psi' in data else None
        z = data['z'] if 'z' in data else None
        alpha = data['alpha'] if 'alpha' in data else None
        speed = data['speed'] if 'speed' in data else None
        brake = data['brake'] if 'brake' in data else None
        power = data['power'] if 'power' in data else None
       
       
        kappa_func = ca.interpolant('KappaProfile', 'bspline', [s], kappa)
        if psi is not None:
            psi_func = ca.interpolant('PsiProfile','linear',[s],psi)
        else:
            psi_func = None

        drs_func = None
        if drs is not None:
            
            drs_func = ca.interpolant('DRSProfile', 'linear', [s], drs)
        
        return cls(
            name=track_name,
            s=s, 
            x=x, 
            y=y, 
            z=z,
            drs=drs,
            psi=psi, 
            kappa=kappa,
            alpha=alpha,
            speed = speed,
            brake= brake,
            power = power,
            length=s[-1],
            kappa_interp=kappa_func,
            psi_interp = psi_func,
            drs_interp=drs_func
        )
    
   

    def get_curvature(self, s_var):
        if self.kappa_interp is None:
            raise RuntimeError("Interpolant not initialized.")
        return self.kappa_interp(s_var)
    
    def get_heading(self,s_var):
        if self.psi_interp is None:
            raise RuntimeError("Interpolant not initialized.")
        return self.psi_interp(s_var)
    
    def get_drs_flag(self, s_var):
        
        if self.drs_interp is None:
            
            if isinstance(s_var, (ca.MX, ca.SX)):
                return ca.DM.zeros(s_var.shape)
            return 0.0
            
        return self.drs_interp(s_var)