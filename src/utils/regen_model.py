import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation parameters
# ---------------------------
N_SIM = 50     # number of Monte Carlo runs
N_LAPS = 22     # laps per run
REGEN_INIT = 0.94

BASE_DRIFT = -0.0015
SIGMA = 0.025


# ---------------------------
# Regen update function
# ---------------------------
def update_regen_usage(regen_eff, base_drift=-0.001, sigma=0.01):
    
    
    noise = np.random.normal(0.0, sigma)
    regen_eff += base_drift + noise
    regen_eff = np.clip(regen_eff, 0.5, 1.0)

    return regen_eff

# ---------------------------
# Run Monte Carlo simulations
# ---------------------------
all_runs = np.zeros((N_SIM, N_LAPS))

for sim in range(N_SIM):
    regen_eff = REGEN_INIT

    for lap in range(N_LAPS):

        

        regen_eff = update_regen_usage(
            regen_eff,
            BASE_DRIFT,
            SIGMA
        )

        all_runs[sim, lap] = regen_eff

# ---------------------------
# Compute statistics
# ---------------------------
mean_curve = np.mean(all_runs, axis=0)
std_curve = np.std(all_runs, axis=0)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))

# individual runs
for i in range(N_SIM):
    plt.plot(all_runs[i], alpha=0.3)

# mean
plt.plot(mean_curve, linewidth=3, label="Mean regen_eff")


plt.title("Monte Carlo: Regen Efficiency Drift Over Laps")
plt.xlabel("Lap")
plt.ylabel("regen_eff")
plt.ylim(0.45, 1.02)
plt.grid(True)
plt.legend()

plt.show()
