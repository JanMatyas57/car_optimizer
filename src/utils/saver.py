import numpy as np
import os
import json
from scipy.interpolate import interp1d

def save_results_to_npz(filepath, res, track, ds, car=None):
    
    # results folder has to exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    sens_array = np.array([res.sens]) if res.sens else np.array([])
    
    car_config_str = "{}"
    if car is not None:
        
        target_keys = [
            "weight", "rotating_mass", "cd", "cd_drs", "cl", "cl_drs", 
            "cop", "power_max", "regen_eff",
            "model_type", "energy_limit", "drs_allowed", "soc_start","name"
        ]
        
        car_dict = {}
        for key in target_keys:
           
            car_dict[key] = getattr(car, key, None)
            
        car_config_str = json.dumps(car_dict)

    print(f"Saving results to: {filepath}")

    f_x = interp1d(track.s, track.x, kind='linear', fill_value="extrapolate")
    f_y = interp1d(track.s, track.y, kind='linear', fill_value="extrapolate")

    x_aligned = f_x(res.s)
    y_aligned = f_y(res.s)
    
    _empty = np.array([])

    np.savez_compressed(
        filepath,
        # main data
        t=res.t,
        v=res.v,
        s=res.s,
        x=x_aligned,
        y=y_aligned,
        E=res.E,
        f_acc=res.f_acc,
        f_brake=res.f_brake,
        recuperation=getattr(res, 'recuperation', _empty),
        SoC=getattr(res, 'SoC', _empty),

        # 2RC battery model outputs (empty arrays when no battery model was used)
        U1=res.U1 if res.U1 is not None else _empty,
        U2=res.U2 if res.U2 is not None else _empty,
        U_terminal=res.U_terminal if res.U_terminal is not None else _empty,
        I_cell=res.I_cell if res.I_cell is not None else _empty,

        # scalar values
        lap_time=res.lap_time,

        # metadata
        track_name=str(track.name),
        ds=float(ds),

        # filtered car JSON
        car_config=car_config_str,

        # sensitivity dictionary
        sens=sens_array
    )