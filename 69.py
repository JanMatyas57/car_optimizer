import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data (jen pro scatter porovnání)
# =========================
soc_grid_recuperation = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                         0.86, 0.88, 0.9, 0.92, 1.0, 1.1]

recuperation_limit_W = [40e3, 40e3, 40e3, 40e3, 40e3, 40e3, 40e3,
                        40e3, 15e3, 5e3, 0, 0, 0]

# =========================
# Sigmoid limit function
# =========================
def get_recuperation_limit(SoC_ca):
    """
    Vrací spojitý (hladký) limit rekuperace ve Wattech pro dané SoC.
    """
    P_max = 40e3      # 40 kW limit
    soc_mid = 0.885   # Kde křivka protíná 20 kW
    k = 120.0         # Strmost poklesu
    
    return P_max / (1.0 + ca.exp(k * (SoC_ca - soc_mid)))

# =========================
# CasADi wrapper (aby šlo evaluovat numericky)
# =========================
soc_sym = ca.MX.sym("soc")
P_expr = get_recuperation_limit(soc_sym)
P_fun = ca.Function("P_fun", [soc_sym], [P_expr])

# =========================
# Evaluation grid
# =========================
soc_plot = np.linspace(min(soc_grid_recuperation),
                       max(soc_grid_recuperation),
                       500)

P_plot = np.array([float(P_fun(s)) for s in soc_plot]) / 1000.0  # kW

# =========================
# Plot
# =========================
plt.figure()
plt.plot(soc_plot, P_plot)
plt.scatter(soc_grid_recuperation,
            np.array(recuperation_limit_W) / 1000.0)

plt.xlabel("State of Charge (-)")
plt.ylabel("Recuperation Limit (kW)")
plt.title("SoC → Recuperation Limit (Sigmoid)")
plt.grid(True)
plt.show()