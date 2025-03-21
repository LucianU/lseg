import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
S0 = 100  # Initial stock price
mu_hat = 0.0005  # Estimated drift (average return per step)
sigma_hat = 0.02  # Estimated volatility (std dev of log returns)

# High-resolution GBM simulation (millisecond steps)
T_continuous = 1  # Simulating for 1 full day
dt_continuous = 1 / (24 * 60 * 60 * 1000)  # Millisecond time steps (1 day = 86,400,000 ms)
num_steps = int(T_continuous / dt_continuous)

# Generate time points
t_vals = np.linspace(0, T_continuous, num_steps)

# Simulate a single GBM path with very fine granularity
S_t_continuous = np.zeros(num_steps)
S_t_continuous[0] = S0  # Start from initial price

# Generate Brownian Motion increments
W_t = np.random.normal(0, np.sqrt(dt_continuous), num_steps - 1)

# Compute GBM values at each millisecond
for i in range(1, num_steps):
    S_t_continuous[i] = S_t_continuous[i-1] * np.exp((mu_hat - 0.5 * sigma_hat**2) * dt_continuous + sigma_hat * W_t[i-1])

# Plot the fine-grained GBM path
plt.figure(figsize=(10, 5))
plt.plot(t_vals[:10000], S_t_continuous[:10000], label="High-Resolution GBM Path", color="blue")
plt.xlabel("Time (fraction of a day)")
plt.ylabel("Price")
plt.title("GBM Path with Millisecond Granularity (First 10,000 Steps)")
plt.legend()
plt.grid()
plt.show()

