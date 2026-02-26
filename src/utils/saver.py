import numpy as np
import os
from scipy.interpolate import interp1d

def save_results_to_npz(filepath, res, track, ds):
    
    
    #results folder has to exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    sens_array = np.array([res.sens]) if res.sens else np.array([])


    print(f"Saving results to: {filepath}")

    f_x = interp1d(track.s, track.x, kind='linear', fill_value="extrapolate")
    f_y = interp1d(track.s, track.y, kind='linear', fill_value="extrapolate")

    x_aligned = f_x(res.s)
    y_aligned = f_y(res.s)
    
    np.savez_compressed(
        filepath,
        #main data
        t=res.t,
        v=res.v,
        s=res.s,
        x= x_aligned,
        y= y_aligned,
        E=res.E,
        f_acc=res.f_acc,
        f_brake=res.f_brake,
        
        # scalar values
        lap_time=res.lap_time,
        
        #metadata
        track_name=str(track.name),
        ds=float(ds),
        
        # sensitivity dictionary
        sens=sens_array
    )