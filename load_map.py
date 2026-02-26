import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 1. Načtení mapy
laptime_map = ca.Function.load("endurance_laptime_map.casadi")

# 2. Vytvoření os X (Energy) a Y (Regen)
# E_target = od 3.5/22 do 7/22 kWh na kolo
energy_vals = np.linspace(3.5/22, 7/22, num=50) 
regen_vals = np.linspace(0.01, 0.94, num=50)

# 3. Vytvoření 2D mřížky (Meshgrid)
E_mesh, R_mesh = np.meshgrid(energy_vals, regen_vals)

# 4. Příprava dat pro CasADi (zploštění mřížek na 1D pole)
E_flat = E_mesh.flatten()
R_flat = R_mesh.flatten()
N_points = len(E_flat)

# Fixní SoC = 1.0 pro tento konkrétní 3D graf
SoC_flat = np.full(N_points, 1)

# 5. Poskládání do matice [SoC, Energy, Regen] a zavolání CasADi
input_matrix = np.vstack((SoC_flat, E_flat, R_flat))
Z_casadi = laptime_map(input_matrix)

# 6. Převedení výsledku zpět do 2D tvaru mřížky
Z_mesh = np.array(Z_casadi).reshape(E_mesh.shape)

# ==========================================
# 7. Vykreslení 3D Grafu
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vykreslení plochy (surface plot)
surf = ax.plot_surface(E_mesh, R_mesh, Z_mesh, 
                       cmap='viridis', edgecolor='none', alpha=0.9)

# Popisky os
ax.set_xlabel('Energy Target per Lap [kWh]')
ax.set_ylabel('Regeneration [%]')
ax.set_zlabel('Lap Time [s]')
ax.set_title('CasADi B-Spline Surface (SoC = 1.0)')

# Přidání colorbaru
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Lap Time [s]')

# Natočení kamery pro lepší pohled
ax.view_init(elev=25, azim=-135)

plt.tight_layout()
plt.show()