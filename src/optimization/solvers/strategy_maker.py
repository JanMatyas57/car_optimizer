import casadi as ca
from src.optimization.solvers.base_solver import BaseSolver


class StrategyMaker(BaseSolver):

    def __init__(self,car,laptime_map,laps,e_target_min,e_target_max, acp_capacity=None, regen_eff=None):
        self.car = car
        self.laptime_map = laptime_map
        self.laps = laps
        self.e_target_min = e_target_min
        self.e_target_max = e_target_max
        self.acp_capacity = acp_capacity if acp_capacity is not None else car.acp_capacity
        self.regen_eff = regen_eff if regen_eff is not None else car.regen_eff
 

    def ef_guess(self,energy_target_endurance):
        #this function is used only as a backup if solver fails
        #simulate the car with x kWh with max regen to estimate winning EF_score
        N = self.laps
        total_time = 0
        energy_target_lap = energy_target_endurance/self.laps
        SoC_start = 0.8

        #simulate the race
        for i in range(0,N,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.acp_capacity)
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,0.94]))
            total_time += laptime
            #print(f"lap {i+1}, laptime {laptime:.2f}, SoC start: {SoC_current:.2f} SoC end: {SoC_start - (i+1)*(energy_target_lap/self.car.acp_capacity):.2f}")
        
        ef_score = total_time**2*energy_target_endurance
        #print(f"Total time: {total_time}")
        #print(f"Ef_score {ef_score}")

        return ef_score
    

    def best_T_guess(self,delta):
        #this function is used only as a backup if solver fails
        #simulate the car with full energy usage to simulate best time
        N = self.laps
        total_time = 0
        energy_target_lap = (self.acp_capacity-0.4)/self.laps
        SoC_start = 1.0

        #simulate the race
        for i in range(0,N,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.acp_capacity)
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,0.94]))
            total_time += laptime
            print(f"lap {i+1}, laptime {laptime:.2f}, SoC start: {SoC_current:.2f} SoC end: {SoC_start - (i+1)*(energy_target_lap/self.acp_capacity):.2f}")
        
        
        print(f"Total time: {total_time}")
        #print(f"Ef_score {ef_score}")
        return total_time*(1-delta)
    
    def getPoints(self,regen,energy_target,delta,ef_min,t_min):
         #this function is used only as a backup if solver fails
        total_time = 0
        SoC_start = 1.0
        energy_target_lap = energy_target/self.laps

         #simulate the race
        for i in range(0,self.laps,1):

            SoC_current = SoC_start - i*(energy_target_lap/self.acp_capacity)
            if SoC_current < 0.05: 
                return 0.0
            laptime = float(self.laptime_map([SoC_current,energy_target_lap,regen]))
            total_time += laptime


        #calculate total points from endurance
        T_min_adj = t_min*(1-delta)
        T_max = 1.5 * T_min_adj            
        P_max_end = 250.0
        P_min_end = 0.1 * P_max_end

        if total_time > T_max:
            nom_end = 0.0
        elif total_time < T_min_adj:
            nom_end = 1.0
        else:
            nom_end = (T_max - total_time) / (T_max - T_min_adj)
            
        Endurance_points = (P_max_end - P_min_end) * (nom_end**2) + P_min_end

        if total_time > T_max: 
            Endurance_points = 0.0

        P_max_eff = 75.0
        EF_max = 2 * ef_min
        Ef_team = (total_time**2) * energy_target

        if Ef_team > EF_max:
            nom_eff = 0.0
        elif Ef_team < ef_min:
            nom_eff = 1.0
        else:
            nom_eff = (EF_max - Ef_team) / (EF_max - ef_min)
            
        Efficiency_points = P_max_eff * (nom_eff**2)
        

        return Endurance_points + Efficiency_points


    def solve(self,ef_min=0,T_min=0,objective="points",enforce_full_soc = False,):
            opti = ca.Opti()
            N = self.laps

            E_lap = opti.variable(N)

            
            SoC_start = opti.variable()

            if enforce_full_soc:
                opti.subject_to(SoC_start == 1.0)
        
            
            
            SoC = opti.variable(N + 1)    

            MIN_SOC = 0.11

    
            opti.subject_to(SoC[0] == SoC_start) 
        

            total_time = 0
            
            for k in range(N):
            
                opti.subject_to(SoC[k+1] == SoC[k] - (E_lap[k] / self.acp_capacity))
               
                state_vec = ca.vertcat(SoC[k], E_lap[k], self.regen_eff)
                laptime = self.laptime_map(state_vec)
    
                total_time += laptime

            opti.subject_to(SoC >= MIN_SOC)
            opti.subject_to(SoC_start >= 0.5)
            opti.subject_to(SoC_start <= 1.0)
            #opti.subject_to(SoC <= 1.0)
            
        
            opti.subject_to(E_lap >= self.e_target_min) 
            opti.subject_to(E_lap <= self.e_target_max) 


            E_guess = 0.25
            opti.set_initial(E_lap, E_guess)
            opti.set_initial(SoC_start, 1.0)
            soc_guess = [1.0 - k * (E_guess / self.acp_capacity) for k in range(N + 1)]
            opti.set_initial(SoC, soc_guess)
           
     
            
            T_max = 1.5 * T_min # Podle tabulky 11 v pravidlech
            
            P_max_end = 250.0
            P_min_end = 0.1 * P_max_end 

         
            if objective == "points":
                opti.subject_to(total_time <= T_max) 
            
            nom_end = (T_max - total_time) / (T_max - T_min)
            Endurance_points = (P_max_end - P_min_end) * (nom_end**2) + P_min_end

          
            P_max_eff = 75.0
            EF_max = 2 * ef_min
            Ef_team = (total_time**2) * ca.sum1(E_lap)

            if objective == "points":
                opti.subject_to(Ef_team <= EF_max)

            nom_eff = (EF_max - Ef_team) / (EF_max - ef_min)
            Efficiency_points = P_max_eff * (nom_eff**2)
            
          
            Total_points = Endurance_points + Efficiency_points
            
            if objective == "points":
                opti.minimize(-Total_points)

            elif objective == "time":
                opti.minimize(total_time)

            elif objective == "efficiency":
                opti.minimize(Ef_team)
            
        
            p_opts = {"expand": True}
            s_opts = {
                "max_iter": 5000, 
                "tol": 1e-3,                   
                "acceptable_tol": 1e-2,           
                "print_level": 0,
                "hessian_approximation": "limited-memory"
            }
            opti.solver('ipopt', p_opts, s_opts)

            try:
                
                sol = opti.solve()
                
                print(f"Total time: {sol.value(total_time):.2f} s")
                
               
                opt_soc = sol.value(SoC)
                opt_e_targets = sol.value(E_lap)
                
                if N == 1:
                    opt_e_targets = [opt_e_targets]
                    opt_soc = [opt_soc, sol.value(SoC)[1]]

                opt_lap_times = []
                
               
                for k in range(N):
                    
                    soc_k = float(opt_soc[k])
                    e_k = float(opt_e_targets[k])
                    regen_k = float(self.regen_eff)
                    
                   
                    lap_time_k = float(sol.value(self.laptime_map(ca.vertcat(soc_k, e_k, regen_k))))
                    opt_lap_times.append(lap_time_k)
                    
                    #happens when solver tries something outside of the laptime map, which evaluates the laptime to zero
                    if lap_time_k < 1: 
                        print(f"   Laptime: {lap_time_k:.2f} s")
                        print(f"   Inputs -> SoC_start: {soc_k:.4f}, E_lap: {e_k:.4f}, Regen: {regen_k:.2f}")
                        return {"success":False}

                opt_total_time = sol.value(total_time)

                return {
                    "success": True,
                    "total_time": opt_total_time,
                    "e_targets": opt_e_targets,     
                    "soc_profile": opt_soc,
                    "soc_start": sol.value(SoC_start),
                    "lap_times": opt_lap_times,
                    "points": sol.value(Total_points),
                    "ef_score": sol.value(Ef_team)
                }
                
            except RuntimeError as e:
                print(f"Solver failed: {e}")
                opti.debug.show_infeasibilities()
                return {"success": False}