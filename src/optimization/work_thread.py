from PyQt6.QtCore import QThread, pyqtSignal
from src.optimization.solver import InitConditions, SolverResults, calculate_initial_guess, calculate_initial_guess_single
from src.core.track import Track
import numpy as np
from dataclasses import dataclass, field
import casadi as ca
import os
from scipy.interpolate import RegularGridInterpolator, griddata
import time


@dataclass
class SolverPayload:
    kind: str  #type of solver run
    data: object #data
    meta: dict = field(default_factory=dict) #additional meta info


MODE_SINGLE = 0
MODE_ENDURANCE = 1
MODE_ACP_TEST = 2
MODE_STRATEGY_OPT = 3
MODE_DEPLOYMENT_OPT = 4


class SolverWorker(QThread):
    # signals for communication with the main window
    finished = pyqtSignal(object) # returns res
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)     

    def __init__(self, solver, car, track,ds,mode,laps=1,laptime_map = None):
        super().__init__()
        self.solver = solver
        self.car = car
        self.track = track
        self.laps = laps
        self.ds = ds
        self.mode = mode
        self.laptime_map = laptime_map
    
    def run(self):
        try:
            if self.mode == MODE_SINGLE:
                self.run_single_lap()
            elif self.mode == MODE_ENDURANCE:
                self.run_endurance()
            elif self.mode == MODE_ACP_TEST:
                self.run_acp_test()
            elif self.mode == MODE_STRATEGY_OPT:
                self.run_endurance_opt()
            elif self.mode == MODE_DEPLOYMENT_OPT:  
                self.run_deployment_opt()
                
        except Exception as e:
            self.error.emit(str(e))

    def run_single_lap(self):
        
        start = time.time()
        #start from the same speed as in the real log
        if self.track.speed is not None:
            init_cond = InitConditions(v=self.track.speed[0], T=0.0, E=2.0)
        else:
            init_cond = InitConditions(v=5, T=0.0, E=0.0)


        #calculate initial guess for the solver to speed up convergence
        init_guess = calculate_initial_guess_single(self.car,self.track,init_cond, ds=self.ds)
        '''
        res = SolverResults(
            success=True,
            lap_time=init_guess['t'][-1],
            t=init_guess['t'],
            v=init_guess['v'],
            s=init_guess['s'],
            n=init_guess['n'],
            E=np.zeros_like(init_guess['t']),
            f_acc=init_guess['F_acc'],
            f_brake= init_guess['F_brake'],
            delta=init_guess['delta'],
            drs_usage=init_guess['drs_usage']
        )
        '''
        
        # solver
        if self.solver.rk4 == False:
            res = self.solver.solveLap(init_cond, ds=self.ds,init_guess=init_guess)
        else:
            res = self.solver.solveLap_rk4(init_cond, ds=self.ds,init_guess=init_guess)
        
        end = time.time()

        print(f"Time elapsed: {end - start:.6f} s")
        
        if not res.success:
            raise Exception("Optimization failed for single lap.")
        
        payload = SolverPayload(
            kind="single_lap",
            data=res,
            meta={"ds": self.ds}
        )
            
        self.finished.emit(payload)

    def run_endurance_opt(self):
        ef_min_res = self.solver.solve(0,0,objective="efficiency")
        if ef_min_res["success"]:
            ef_min = ef_min_res["ef_score"]
            print(f"opt ef score {ef_min}")
        else:
            ef_min = self.solver.ef_guess(4)
            print(f"fallback ef score {ef_min}")

        #fix time delta
        t_min_res = self.solver.solve(0,0,objective="time")
        if t_min_res["success"]:
            t_min = t_min_res["total_time"]
            print(f"opt min time {t_min}")
        else:
            t_min = self.solver.best_T_guess(0)
            print(f"fallback min time {t_min}")
      

        regen_targets = np.linspace(0.15, 1.0, num=10, endpoint=True)
        delta_targets = np.linspace(0.0, 0.15, num=10, endpoint=True)

        # POZOR NA ZÁVORKY: np.zeros((x, y)) musí mít tuple!
        e_opt_grid = np.zeros((len(regen_targets), len(delta_targets)))
        soc_opt_grid = np.zeros((len(regen_targets), len(delta_targets)))

        current_step =0
        total_steps = len(regen_targets)*len(delta_targets)

        for i, regen_target in enumerate(regen_targets):
            for j, delta_target in enumerate(delta_targets):

                current_step += 1
                self.progress.emit(current_step, total_steps)
                
                self.car.regen_eff = regen_target
                t_min_adj = t_min*(1-delta_target)
              
                # Vyřešíme optimum přes CasADi
                res = self.solver.solve(ef_min, t_min_adj, objective="points")
                
                if res and res.get("success", False):
                    # res["e_targets"] je pole spotřeby na kolo, sečteme ho pro celkovou energii
                    e_opt_grid[i, j] = np.sum(res["e_targets"])
                    soc_opt_grid[i, j] = res["soc_start"]
                else:
                    # Pokud solver pro danou kombinaci nenajde řešení, dáme tam NaN
                    e_opt_grid[i, j] = np.nan
                    soc_opt_grid[i, j] = np.nan

        # Vytvoření DVOU 2D interpolantů (method='linear' je lepší, pokud máme v gridu nějaké NaN hodnoty ze selhání)
        e_interpolant = RegularGridInterpolator(
            (regen_targets, delta_targets), e_opt_grid, method='linear', bounds_error=False, fill_value=np.nan
        )
        soc_interpolant = RegularGridInterpolator(
            (regen_targets, delta_targets), soc_opt_grid, method='linear', bounds_error=False, fill_value=np.nan
        )
     
        payload = SolverPayload(
            kind="endurance_map",
            data=None,
            meta={
                "ef_min": ef_min,
                "t_min": t_min,
                "e_interpolant": e_interpolant,      #
                "soc_interpolant": soc_interpolant,  
                "grid_regen": regen_targets,
                "grid_delta": delta_targets,
                "z_min_e": np.nanmin(e_opt_grid),    
                "z_max_e": np.nanmax(e_opt_grid)
            }
        )

        self.finished.emit(payload)

    def run_deployment_opt(self):
        """
        Vypočítá pouze jedno konkrétní nasazení strategie (jeden bod)
        pro zadaný Regen a Delta a vrátí detailní data pro StrategyResultTab.
        """
        # 1. Získání baseline hodnot (stejně jako u gridu)
        ef_min_res = self.solver.solve(0,0,objective="efficiency")
        if ef_min_res["success"]:
            ef_min = ef_min_res["ef_score"]
            print(f"opt ef score {ef_min}")
        else:
            ef_min = self.solver.ef_guess(4)
            print(f"fallback ef score {ef_min}")


        #fix time delta
        t_min_res = self.solver.solve(0,0,objective="time")
        if t_min_res["success"]:
            t_min = t_min_res["total_time"]
            print(f"opt min time {t_min}")
        else:
            t_min = self.solver.best_T_guess(0)
            print(f"fallback min time {t_min}")
      

        # 2. Úprava času soupeře podle zadané delty
        t_min_adj = t_min * (1 - self.car.performance_delta)

        # 3. Spuštění optimalizace pro tyto konkrétní podmínky
        res_opt = self.solver.solve(ef_min, t_min_adj,objective="points")

        res_full = self.solver.solve(ef_min,t_min_adj,objective="points",enforce_full_soc=True)

        # 4. Vytvoření payloadu
        # Použijeme kind="endurance_deployment", na který vaše UI už čeká
        payload = SolverPayload(
            kind="endurance_deployment",
            data={
                "opt_soc": res_opt,
                "full_soc": res_full
            },
            meta={
                "ef_min": ef_min,
                "t_min": t_min,
                "regen_eff": self.car.regen_eff,
                "delta": self.car.performance_delta
            }
        )

        # 5. Odeslání výsledků do hlavního vlákna
        self.finished.emit(payload)


    def run_acp_test(self):
        
        start = time.time()

        total_race_distance = 22000.0 # [m]
        laps = int(np.round(total_race_distance / self.track.length))

        #battery_scaling_factor = 0.22 # kWh/kg
        original_mass = self.car.mass
        original_regen = self.car.regen_eff

        # grid setup
        acp_targets = np.linspace(3.5,10.0,num=15,endpoint=True)
        e_targets = np.round(acp_targets/self.laps,2)
        
        regen_targets = np.linspace(0.0, 1.1, num=15, endpoint=True)

        soc_targets = np.linspace(0.08,1.05,num=15,endpoint=True)

        total_steps = len(e_targets) * len(regen_targets)*len(soc_targets)
        results_log = []
        
        current_step = 0

        flat_lap_times = []

        last_slowest_lap = 40

        prev_res = None

        for soc in soc_targets:
            for e_target in e_targets:
                for regen in regen_targets:

                    current_step += 1
                    self.progress.emit(current_step, total_steps)
                    
                    # 1. Kontrola, zda je to vůbec fyzikálně možné
                    if soc * self.car.acp_capacity < e_target:
                        flat_lap_times.append(np.nan)
                        continue
                    
                    # 2. Nastavení auta a počátečních podmínek
                    self.car.regen_eff = regen
                    self.car.energy_limit = e_target
                    E_start = np.round(self.car.acp_capacity*(1-soc),2)
                    init_cond = InitConditions(v=self.track.speed[0]-2,T=0.0,E=E_start)
                    print(f"E_start :{E_start}, SoC: {soc} E target: {e_target} Regen: {regen}")
                    res = self.solver.solveLap_rk4(
                        init_cond=init_cond,
                        ds=self.ds,
                        init_guess=prev_res
                    )

                    
                    if res.success:
                        laptime = res.lap_time
                        flat_lap_times.append(laptime)
                        prev_res = res
                        if laptime > last_slowest_lap:
                            last_slowest_lap = laptime

                        # Uložíme i data pro PyQt UI (raw body)
                        res.meta_soc = soc
                        res.meta_energy = e_target
                        res.meta_regen = regen
                        res.meta_mass = self.car.mass
                        results_log.append(res)
                    else:
                        flat_lap_times.append(np.nan) 


       
        flat_lap_times = np.array(flat_lap_times, dtype=float)
        
        if np.isnan(flat_lap_times).any():
       
            
            
            S, E_grid, R = np.meshgrid(soc_targets, e_targets, regen_targets, indexing='ij')
            
       
            points = np.column_stack((S.ravel(), E_grid.ravel(), R.ravel()))
            values = flat_lap_times
            
            # Maska platných hodnot
            valid_mask = ~np.isnan(values)
            
            if valid_mask.any():
                valid_points = points[valid_mask]
                valid_values = values[valid_mask]
                
                # Vyplnění NaN bodů nejbližší platnou hodnotou
                filled_values = griddata(valid_points, valid_values, points, method='nearest')
                
                # Přepíšeme pole s lap časy vyčištěnými daty
                flat_lap_times = filled_values
            else:
                pass

        end = time.time()

        print(f"Time elapsed: {end - start:.6f} s")

        save_dir = os.path.join("data", "tracks", "lapmaps")
        
        # Create the directories if they do not exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct the full file path
        # Added an underscore before the track name for better readability
        file_path = os.path.join(save_dir, f"laptime_map_{self.track.name}.npz")

   
        np.savez_compressed(
            file_path,
            soc_targets=np.array(soc_targets),
            e_targets=np.array(e_targets),
            regen_targets=np.array(regen_targets),
            flat_lap_times=np.array(flat_lap_times), 
            laps=np.array(self.laps),
            acp_capacity=np.array(self.car.acp_capacity)
        )

    
        payload = SolverPayload(
            kind="acp_test",
            data=results_log, # Raw data pro scatter ploty v UI
            meta={
                "laps": laps
            }
        )

        self.finished.emit(payload)
        '''
        for r_idx, regen_val in enumerate(regen_targets):
           
            self.car.regen_eff = regen_val

            prev_res = None 

            for e_idx, e_target in enumerate(e_targets):
                current_step += 1
                self.progress.emit(current_step, total_steps)

                #recalculate battery mass based on energy target
                #capacity_needed = laps * e_target
                #delta_cap = capacity_needed - self.car.acp_capacity # kWh
                #weight_delta = delta_cap / battery_scaling_factor
                
               # self.car.mass = original_mass + weight_delta
                
                # Init conditions for the lap
                current_cond = InitConditions(v=10.0, T=0.0, E=0.0)
                self.car.energy_limit = e_target 
                
                # run the solver
                try:
                    res = self.solver.solveLap(
                        init_cond=current_cond,
                        ds=self.ds,
                        init_guess=prev_res # warm start
                    )
                    
                    if res.success:

                        res.meta_energy = e_target
                        res.meta_regen = regen_val
                        res.meta_mass = self.car.mass
                        
                        results_log.append(res)
                        prev_res = res # use as warm start for next
                    
                except Exception as e:
                    print(f"Solver fail at Regen={regen_val}, E={e_target}: {e}")

        #return car to original state
        self.car.mass = original_mass
        self.car.regen_eff = original_regen

        if not results_log:
            raise Exception("Sensitivity test failed: No feasible solution found.")
        
        payload = SolverPayload(
            kind="acp_test",
            data=results_log,
            meta={"ahoj":1}
        )

        self.finished.emit(payload)
    '''
    def run_drs_test(self):

        #calculate energy targets
        acp_targets = np.linspace(4.2,6.5, num=5, endpoint=True) #acp capacity from 4.2 kwH to 6.5 kwH
        energy_targets = acp_targets/self.laps
        results_drs = []
        results_no_drs = []

        current_step = 0

        print(f"laps: {self.laps}")

        total_steps = len(energy_targets) * 2  # DRS and no-DRS for each target
        for target in energy_targets:
            current_step += 1
            self.progress.emit(current_step, total_steps)

            current_cond = InitConditions(v=self.track.speed[1], T=0.0, E=0.0)
            # run the solver with DRS enabled
            self.car.drs_allowed = True
            self.car.energy_limit = target
            try:
                res_drs = self.solver.solveLap(
                    init_cond=current_cond,
                    ds=self.ds
                )

                laptime = res_drs.lap_time
                energy_used = res_drs.E[-1]
                drs_usage = res_drs.drs_usage
                drs_active_ratio = (np.sum(drs_usage > 0.9) / drs_usage.size) * 100.0
                ef_score = energy_used*self.laps*(laptime*self.laps)**2

                results_drs.append({'target': target*self.laps, 'laptime': laptime, 'energy_used': energy_used, 'drs_active_ratio': drs_active_ratio, 'ef_score': ef_score})
            except Exception as e:
                print(f"DRS test failed for target {target}: {e}")

            # run the solver without DRS
            self.car.drs_allowed = False
            current_step += 1
            self.progress.emit(current_step, total_steps)
            try:
                res_no_drs = self.solver.solveLap(
                    init_cond=current_cond,
                    ds=self.ds
                )
                laptime = res_no_drs.lap_time
                energy_used = res_no_drs.E[-1]
                ef_score = energy_used*self.laps*(laptime*self.laps)**2
                results_no_drs.append({'target': target*self.laps, 'laptime': laptime, 'energy_used': energy_used, 'ef_score': ef_score})

            except Exception as e:
                print(f"No-DRS test failed for target {target}: {e}")
            

        #export results
        payload = SolverPayload(
        kind="drs_test",
        data={
            "drs": results_drs,
            "nodrs": results_no_drs
        },
        meta={"ds": self.ds})

        self.finished.emit(payload)

    def run_endurance(self):
        # we start from 5 ms
        current_cond = InitConditions(v=5.0, T=0.0, E=0.0)
        endurance_log = []
        prev_res = None
        #set the tires to be cold for first lap
        self.car.mu_x = self.car.tire_model['cold_factor']*self.car.tire_model['mu_x_nominal']
        self.car.mu_y = self.car.tire_model['cold_factor']*self.car.tire_model['mu_y_nominal']
        print(f"Starting endurance simulation ({self.solver.laps} laps)...")
        self.car.energy_limit = self.car.acp_capacity/self.laps
        for i in range(self.laps):
            # send info progress
            self.progress.emit(i + 1, self.laps)
            print(f"Lap {i+1} out of {self.laps}")

            #solve current
            res = self.solver.solveLap(
                    init_cond=current_cond, 
                    ds=self.ds,
                    init_guess=prev_res)
                
            if not res.success:
                raise Exception(f"Endurance failed at lap {i+1}")
            
            #save the results
            res.lap_id = i + 1
            res.mu_x = self.car.mu_x       
            res.mu_y = self.car.mu_y
            res.regen_eff = self.car.regen_eff 
            endurance_log.append(res)

            #get the last used mu
            last_mu = self.car.mu_y
            #update the driver error
            self.car.update_grip(i)
            ratio = self.car.mu_y / last_mu
            safe_v = res.v[-1] * np.sqrt(ratio)
            #next lap starting conditions are the conditions we finished this lap
            current_cond.v = safe_v #this is scaled for the new mu change, so it doesnt crash
            current_cond.T = res.t[-1]
            current_cond.E = res.E[-1]
            E_remaining = self.car.acp_capacity - current_cond.E
            #target for next lap:
            laps_remaining = self.laps-(i+1)

            if laps_remaining > 0:
                self.car.energy_limit = E_remaining/laps_remaining
            else:
                self.car.energy_limit = E_remaining
           
            #update the regen factor
            self.car.update_regen_usage()
            print(f"Car regen factor: {self.car.regen_eff}")

            
        payload = SolverPayload(
            kind="endurance",
            data=endurance_log,
            meta={"ds": self.ds, "laps": self.laps, "final_energy": current_cond.E}
        )
        self.finished.emit(payload)





    

