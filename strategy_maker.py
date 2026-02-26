import casadi as ca
import numpy as np

class EnduranceMacroSolver:
    def __init__(self, laps=22):
        self.laps = laps

    def solve_race(self, casadi_laptime_map, fixed_regen=0.8):
        opti = ca.Opti()
        N = self.laps

        # ==========================================
        # 1. Proměnné (Variables)
        # ==========================================
        # Kolik energie auto reálně spotřebuje v každém kole [kWh]
        E_lap = opti.variable(N)
        E_total = opti.variable(N+1)
        
        # Stav baterie (SoC) na začátku každého kola + na úplném konci závodu (N+1)
        SoC = opti.variable(N + 1)    

        # ==========================================
        # 2. Parametry a limity
        # ==========================================
        Total_Capacity = 6
        MIN_SOC = 0.05  # Nenecháme auto úplně chcípnout (5 % rezerva)
        
        # ==========================================
        # 3. Počáteční podmínky
        # ==========================================
        opti.subject_to(SoC[0] == 1.0) # Na startu je baterie nabitá na 100 %
        opti.subject_to(E_total[0] == 0)


        # ==========================================
        # 4. Dynamika závodu (Krok za krokem)
        # ==========================================
        total_time = 0
        
        for k in range(N):
            # A) Výpočet SoC pro další kolo (úbytek energie)
            # Aktuální SoC minus (spotřebovaná energie / celková kapacita)
            opti.subject_to(SoC[k+1] == SoC[k] - (E_lap[k] / Total_Capacity))
            opti.subject_to(E_total[k+1] == E_total[k] + E_lap[k])
            
            state_vec = ca.vertcat(SoC[k], E_lap[k], fixed_regen)
            lap_time = casadi_laptime_map(state_vec)
            
            # C) Přičtení času kola k celkovému času
            total_time += lap_time

        # ==========================================
        # 5. Omezující podmínky (Constraints)
        # ==========================================
        # Baterie nesmí klesnout pod limit a nesmí se magicky nabít nad 100 %
        opti.subject_to(SoC >= MIN_SOC)
        #opti.subject_to(SoC <= 1.0)
        
        # Omezení pro E_lap, aby solver neextrapoloval moc mimo tvůj grid
        # (Dosaď sem reálné hranice tvého e_targets gridu!)
        opti.subject_to(E_lap >= 0.16) 
        opti.subject_to(E_lap <= 0.35) 
        opti.subject_to(E_total[N] <= 5.5)

        # ==========================================
        # 6. Cílová funkce (Objective)
        # ==========================================
        opti.minimize(total_time)

        # ==========================================
        # 7. Initial Guess
        # ==========================================
        allowed_energy = 5.5
        avg_e_per_lap = allowed_energy / N
        opti.set_initial(E_lap, avg_e_per_lap)
        
        # SoC bude lineárně klesat od 1.0 do MIN_SOC
        soc_guess = np.linspace(1.0, MIN_SOC, N + 1)
        opti.set_initial(SoC, soc_guess)

        # ==========================================
        # 8. Nastavení a spuštění IPOPTu
        # ==========================================
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 5000, 
            "tol": 1e-3,                      # Požadovaná přesnost (standardní)
            "acceptable_tol": 1e-2,           # Pokud se zasekne, spokojí se i s tímto
            "acceptable_iter": 5,             # Kolikrát musí být v acceptable zóně, než to ukončí
            "print_level": 5,
            "hessian_approximation": "limited-memory"
        }
        opti.solver('ipopt', p_opts, s_opts)

        try:
            sol = opti.solve()
            
            print(f"Optimalizace ENDURANCE úspěšná!")
            print(f"Celkový čas: {sol.value(total_time):.2f} s")
            
            return {
                "success": True,
                "total_time": sol.value(total_time),
                "e_targets": sol.value(E_lap),     # Pole o délce 22
                "soc_profile": sol.value(SoC),     # Pole o délce 23
                "lap_times": [sol.value(casadi_laptime_map(ca.vertcat(sol.value(SoC)[k], sol.value(E_lap)[k], fixed_regen))) for k in range(N)]
            }
            
        except RuntimeError as e:
            print(f"Makro-solver selhal: {e}")
            # Můžeš si vytisknout, kde se solver zasekl
            print("Poslední známé E_targets:", opti.debug.value(E_lap))
            print("Poslední známé SoC:", opti.debug.value(SoC))
            return {"success": False}
        


laptime_map = ca.Function.load("endurance_laptime_map.casadi")
strategy = EnduranceMacroSolver(laps=22)
strategy.solve_race(laptime_map)