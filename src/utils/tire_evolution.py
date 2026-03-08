import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
Total_Laps = 22
Num_Simulations = 50

Base_Mu_X = 1.3  # target optimal grip

Tire_Model = {
    "Cold_Factor": 0.6,   # start grip fraction
    "Warmup_Tau": 1.5,    # warmup time constant
    "Noise_Std": 0.016     # driver inconsistency
}

# ---------------------------
# PREP
# ---------------------------
laps_vector = np.arange(1, Total_Laps + 1)

mu_optimal = Base_Mu_X
mu_cold = Base_Mu_X * Tire_Model["Cold_Factor"]

plt.figure(figsize=(10, 6))
plt.grid(True)

# ---------------------------
# MONTE CARLO SIM
# ---------------------------
for sim in range(Num_Simulations):
    grip_history = np.zeros(Total_Laps)

    for lap in range(Total_Laps):
        # Exponential warmup curve
        warmup_progress = 1 - np.exp(-(lap + 1) / Tire_Model["Warmup_Tau"])

        current_base_mu = mu_cold + (mu_optimal - mu_cold) * warmup_progress

        # Noise
        noise = Tire_Model["Noise_Std"] * np.random.randn()

        mu_lap = current_base_mu + noise

        # Clip grip lower bound
        min_mu = mu_cold * 0.9
        mu_lap = max(mu_lap, min_mu)

        grip_history[lap] = mu_lap

    # Plot each simulation
    plt.plot(laps_vector, grip_history, color=(0.2, 0.6, 1.0, 0.3), linewidth=1.5)


ideal_curve = mu_cold + (mu_optimal - mu_cold) * (1 - np.exp(-laps_vector / Tire_Model["Warmup_Tau"]))
plt.plot(laps_vector, ideal_curve, "r--", linewidth=2, label="Ideal warmup")

plt.axhline(Base_Mu_X, color="black", linestyle="-", label="Optimal Grip")


plt.title(
    f"Tire evolution model (Start: {Tire_Model['Cold_Factor']*100:.0f}%, Tau: {Tire_Model['Warmup_Tau']:.1f})"
)
plt.xlabel("Lap")
plt.ylabel("Longitudinal Friction (μx)")
#plt.ylim([Base_Mu_X * 0.8, Base_Mu_X * 1.05])
plt.legend()

plt.show()
