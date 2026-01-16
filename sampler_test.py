import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from mc import Sampler  # Importing from your file

# --- 1. Define Test Wavefunction (Amplitude) ---

def psi_gaussian(x, params):
    """
    Target Wavefunction: psi(x) = exp(-x^2 / 2)
    Probability: P(x) = |psi|^2 = exp(-x^2)
    """
    # Sum of squares for N particles
    r2 = jnp.sum(x**2)
    return jnp.exp(-r2 / 2.0)

# --- 2. Configuration ---

N_particles = 3
params = None 
n_samples = 200000   # Total samples to target
thermal = 1000       # Burn-in steps
keep = 20           # Thinning
step_size = 1.2    # Tuned for ~50% acceptance

# Initialize the Sampler with the PSI function
sampler = Sampler(psi_gaussian, (N_particles,))

# --- 3. Test Serial Sampling ---
print("--- Testing Serial Sampler ---")
init_pos = jnp.ones((N_particles,))
start_time = time.time()

samples_serial, _, acc_serial = sampler.sample(
    params, 
    Nsweeps=n_samples, 
    Ntherm=thermal, 
    keep=keep, 
    stepsize=step_size, 
    initial_sample=init_pos
)

ser_time = time.time() - start_time
print(f"Serial Time: {ser_time:.4f}s")
print(f"Acceptance Rate: {acc_serial:.2%}")
print(f"Shape: {samples_serial.shape}")

# --- 4. Test Parallel Sampling ---
print("\n--- Testing Parallel Sampler ---")
# Adjust n_sweeps per chain so total samples are comparable
num_chains = 16  # Use 16 parallel chains
sweeps_per_chain = n_samples // num_chains

start_time = time.time()

samples_parallel, _, acc_parallel = sampler.sample_parallel(
    params, 
    Nsweeps=sweeps_per_chain, 
    Ntherm=thermal, 
    keep=keep, 
    stepsize=step_size, 
    initial_sample=init_pos,
    num_chains=num_chains
)
par_time = time.time() - start_time
print(f"Parallel Time: {par_time:.4f}s")
print(f"Acceptance Rate: {acc_parallel:.2%}")
print(f"Shape: {samples_parallel.shape}")
print("\n")
print(f"Speedup: {ser_time / par_time:.2f}x")
# --- 5. Visualization ---

# Analytical PDF: P(x) ~ e^{-x^2} normalized
x_plot = np.linspace(-3, 3, 200)
pdf_analytical = (1 / np.sqrt(np.pi)) * np.exp(-x_plot**2)

plt.figure(figsize=(12, 5))

# Plot Serial Results
plt.subplot(1, 2, 1)
plt.hist(np.array(samples_serial).flatten(), bins=60, density=True, 
         alpha=0.6, color='dodgerblue', edgecolor='black', label='Serial VMC')
plt.plot(x_plot, pdf_analytical, 'r-', lw=2, label='Analytical')
plt.title(f"Serial Sampling\n(Acc: {acc_serial:.0%})")
plt.legend()

# Plot Parallel Results
plt.subplot(1, 2, 2)
plt.hist(np.array(samples_parallel).flatten(), bins=60, density=True, 
         alpha=0.6, color='forestgreen', edgecolor='black', label='Parallel VMC')
plt.plot(x_plot, pdf_analytical, 'r-', lw=2, label='Analytical')
plt.title(f"Parallel Sampling\n(Acc: {acc_parallel:.0%}, Chains: {num_chains})")
plt.legend()

plt.tight_layout()
plt.show()
