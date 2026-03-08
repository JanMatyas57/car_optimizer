import numpy as np
import matplotlib.pyplot as plt

# parametry
kappa_k = 150.0
kappa_thr = 0.1
kappa_max = 0.2

# vstupní zakřivení
kappa_vals = np.linspace(0, 0.2, 200)
kappa_vals = kappa_vals/kappa_max

# drs_curve funkce (ekvivalent CasADi)
drs_vals = 1 / (1 + np.exp(kappa_k * (np.abs(kappa_vals) - kappa_thr)))

# plot
plt.figure(figsize=(6,4))
plt.plot(kappa_vals, drs_vals, label='DRS vs Curvature')
plt.axvline(kappa_thr, color='r', linestyle='--', label='Threshold')
plt.xlabel('Curvature κ [1/m]')
plt.ylabel('DRS factor')
plt.title('Soft DRS curve vs curvature')
plt.grid(True)
plt.legend()
plt.show()
