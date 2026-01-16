# from jax import jit, Array, random, vmap, pmap, local_device_count
# from jax.lax import scan, axis_index
# import numpy as np
# from jax import numpy as jnp
# from jax.lax import cond
# import tqdm
# import time 
# from functools import partial


# class Sampler:
#     def __init__(self, psi, shape):
#         self.psi = psi
#         self.shape = shape
#         self.N = shape[0]
#         self.d = shape[1]


#     @partial(jit, static_argnums=(0,))
#     def mcstep_E(self, xis, limit, positions, params):
        
#     #     params = jax.device_put(params, device=jax.devices("cpu")[0])
        
#         newpositions = jnp.array(positions) + xis
        
#         # prob = psi(newpositions, params)**2./psi(positions, params)**2.
#         prob = (self.psi(newpositions, params)**2./self.psi(positions, params)**2.)
        
#         def truefunc(p):
#             return [newpositions, True]

#         def falsefunc(p):
#             return [positions, False]
        
#         return cond(prob >= limit, truefunc, falsefunc, prob)


#     def sample(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample, progress=False):
#         """
#         Monte Carlo sampling function. Samples from the probability distribution psi^2 using Metropolis algorithm.
        
#         :param params: Parameters for the function psi
#         :param Nsweeps: Number of samples kept
#         :param Ntherm: Number of thermalization steps
#         :param keep: Keep every 'keep' samples
#         :param stepsize: Step size for the Metropolis moves
#         :param initial_sample: Initial sample configuration
#         :param progress: Whether to show a progress bar
#         """
#         sq = []
#         spq = []
#         counter = 0
#         num_total = Nsweeps * keep + Ntherm
#         rng = np.random.default_rng(int(time.time()))
#         randoms = rng.uniform(-stepsize, stepsize, size = (num_total, self.N))
#         limits = rng.uniform(0, 1, size = num_total)
        
#         positions_prev = initial_sample
#         if progress:
#             for i in tqdm.tqdm(range(0, num_total), position = 0, leave = True, desc = "MC"):
                
#                 new, moved = self.mcstep_E(randoms[i], limits[i], positions_prev, params)
            
#                 if moved:
#                     counter += 1
            
#                 if i%keep == 0 and i >= Ntherm:
#                     sq.append(new)
#                     spq.append(new.at[-1].set(new[0]))

                    
#                 positions_prev = new
#         else:
#             for i in range(0, num_total):
#                 new, moved = self.mcstep_E(randoms[i], limits[i], positions_prev, params)
#                 if moved:
#                     counter += 1
            
#                 if i%keep == 0 and i >= Ntherm:
#                     sq.append(new)
#                     spq.append(new.at[-1].set(new[0]))
                    
#                 positions_prev = new
#         return jnp.array(sq), jnp.array(spq), counter/num_total

#     # @partial(jit, static_argnums=(0,))
#     # def sample_prime(self, samples):
#     #     sq_prime = jnp.array(samples.copy())
#     #     for i in range(len(samples)):
#     #         a = jnp.array(samples[i])
#     #         # note that this will fuck up if doing fermions with no second species
#     #         a = a.at[-1].set(a[0])
#     #         sq_prime = sq_prime.at[i].set(jnp.array(a))
#     #     return jnp.array(sq_prime)

#     @partial(jit, static_argnums=(0,))
#     def sample_prime(self, samples):
#         x = jnp.asarray(samples)              # (Nsamples, N)
#         return x.at[:, -1].set(x[:, 0])


#     def sample_parallel(
#         self,
#         params,
#         Nsweeps: int,
#         Ntherm: int,
#         keep: int,
#         stepsize: float,
#         initial_sample,
#         key: Array,
#         n_chains: int | None = None,
#         init_jitter: float = 0.0,
#     ):
#         """
#         Run multiple independent Metropolis chains in parallel using pmap.

#         Key distribution: ONLY uses random.split.
#         - split master key -> per-device keys
#         - split per-device key -> per-chain keys (within that device)
#         - each chain evolves by splitting its own key each step

#         Returns:
#         samples:       (n_chains, Nsweeps+1, N)
#         samples_prime: (n_chains, Nsweeps+1, N)
#         accept_rate:   (n_chains,)
#         """
#         ndev = local_device_count()
#         if n_chains is None:
#             n_chains = ndev
#         if n_chains % ndev != 0:
#             raise ValueError(f"n_chains ({n_chains}) must be divisible by local_device_count ({ndev}).")
#         chains_per_dev = n_chains // ndev

#         # total Metropolis steps
#         num_total = Nsweeps * keep + Ntherm + 1

#         # initial positions
#         init = jnp.asarray(initial_sample)
#         if init.ndim == 1:
#             init = jnp.broadcast_to(init, (n_chains, self.N))
#         elif init.shape != (n_chains, self.N):
#             raise ValueError(f"initial_sample must be shape ({self.N},) or ({n_chains},{self.N}); got {init.shape}.")

#         # reshape to (ndev, chains_per_dev, N)
#         init = init.reshape(ndev, chains_per_dev, self.N)

#         # choose kept indices (no boolean masking inside jit/pmap)
#         keep_idx = Ntherm + keep * jnp.arange(Nsweeps + 1)  # (Nsweeps+1,)

#         # split master key -> per-device keys
#         dev_keys = random.split(key, ndev)  # (ndev, 2)

#         def run_on_device(params, init_pos, dev_key):
#             """
#             Runs chains_per_dev chains on one device (vectorized).
#             init_pos: (chains_per_dev, N)
#             dev_key:  (2,)
#             """
#             # split device key -> (chains_per_dev) per-chain keys (+ one spare)
#             chain_keys = random.split(dev_key, chains_per_dev + 1)
#             chain_keys, jitter_key = chain_keys[:-1], chain_keys[-1]

#             # optional: decorrelate identical initial states
#             if init_jitter != 0.0:
#                 # split jitter_key into per-chain jitter keys
#                 jitter_keys = random.split(jitter_key, chains_per_dev)
#                 jitters = vmap(lambda k: random.normal(k, (self.N,)))(jitter_keys) * init_jitter
#                 init_pos = init_pos + jitters

#             def one_chain_scan(pos0, k0):
#                 def step(carry, _):
#                     pos, k = carry
#                     k, k1, k2 = random.split(k, 3)
#                     xis = random.uniform(k1, (self.N,), minval=-stepsize, maxval=stepsize)
#                     limit = random.uniform(k2, (), minval=0.0, maxval=1.0)
#                     new_pos, moved = self.mcstep_E(xis, limit, pos, params)
#                     return (new_pos, k), (new_pos, moved)

#                 (_, _), (positions, moved_flags) = scan(step, (pos0, k0), xs=None, length=num_total)
#                 return positions, moved_flags  # positions: (num_total, N), moved_flags: (num_total,)

#             # vmap over chains on this device
#             positions, moved = vmap(one_chain_scan, in_axes=(0, 0))(init_pos, chain_keys)
#             # positions: (chains_per_dev, num_total, N)

#             kept = jnp.take(positions, keep_idx, axis=1)  # (chains_per_dev, Nsweeps+1, N)
#             acc = jnp.mean(moved.astype(jnp.float32), axis=1)  # (chains_per_dev,)

#             kept_prime = vmap(self.sample_prime, in_axes=0)(kept)
#             return kept, kept_prime, acc

#         pmapped = pmap(run_on_device, in_axes=(None, 0, 0))
#         kept, kept_prime, acc = pmapped(params, init, dev_keys)

#         # merge device and chain axes
#         kept = kept.reshape(n_chains, Nsweeps + 1, self.N)
#         kept_prime = kept_prime.reshape(n_chains, Nsweeps + 1, self.N)
#         acc = acc.reshape(n_chains)
#         return kept, kept_prime, acc



import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap, lax
from jax.lax import scan, cond
from functools import partial
import numpy as np
import time

class Sampler:
    def __init__(self, psi, shape):
        """
        :param psi: Function psi(x, params) returning the wavefunction amplitude.
        :param shape: (N_particles, ...) tuple defining the system shape.
        """
        self.psi = psi
        self.shape = shape
        self.N = shape[0]

    @partial(jit, static_argnums=(0,))
    def mcstep_psi(self, key, positions, params, stepsize):
        """
        Single Metropolis-Hastings step using raw PSI amplitudes.
        """
        k1, k2 = random.split(key)
        
        # 1. Propose Move
        xis = random.uniform(k1, (self.N,), minval=-stepsize, maxval=stepsize)
        newpositions = positions + xis
        
        # 2. Compute Probability Ratio
        # ratio = |psi(new)|^2 / |psi(old)|^2
        psi_old = self.psi(positions, params)
        psi_new = self.psi(newpositions, params)
        
        # Guard against division by zero (underflow protection)
        # We clamp the denominator to a tiny value to avoid NaN.
        denom = jnp.maximum(psi_old**2, 1e-20)
        prob_ratio = (psi_new**2) / denom
        
        # 3. Acceptance Logic
        # Accept if uniform(0,1) < prob_ratio
        rand_val = random.uniform(k2, minval=0.0, maxval=1.0)
        accept = rand_val < prob_ratio

        # 4. Update
        final_pos = cond(accept, lambda: newpositions, lambda: positions)
        return final_pos, accept

    def sample(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample, progress=False):
        """
        Serial sampling using jax.lax.scan for performance.
        """
        total_steps = Nsweeps * keep + Ntherm
        key = random.PRNGKey(int(time.time()))
        
        def step_fn(carry, _):
            pos, k = carry
            k_step, k_next = random.split(k)
            new_pos, moved = self.mcstep_psi(k_step, pos, params, stepsize)
            return (new_pos, k_next), (new_pos, moved)

        # Run Chain
        init_carry = (jnp.array(initial_sample), key)
        _, (history, moved_history) = scan(step_fn, init_carry, length=total_steps)

        # Post-process (Thermalize & Skip)
        kept_samples = history[Ntherm::keep]
        kept_samples_prime = self.sample_prime(kept_samples)
        
        acc_rate = jnp.mean(moved_history)
        
        return kept_samples, kept_samples_prime, acc_rate

    def sample_parallel(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample, 
                        num_chains=None, seed=None):
        """
        Parallel sampling using pmap (multi-device) + vmap (vectorization).
        """
        num_devices = jax.local_device_count()
        if num_chains is None:
            num_chains = num_devices
            
        if num_chains % num_devices != 0:
            raise ValueError(f"num_chains ({num_chains}) must be divisible by devices ({num_devices})")
            
        chains_per_device = num_chains // num_devices
        total_steps = Nsweeps * keep + Ntherm
        
        # --- Prepare Inputs ---
        if seed is None:
            seed = int(time.time())
        key = random.PRNGKey(seed)
        key_devs = random.split(key, num_devices) # Split for pmap

        # Handle Initial Position Broadcasting
        init = jnp.asarray(initial_sample)
        if init.ndim == 1:
            # Broadcast single sample to all chains
            init = jnp.tile(init, (num_devices, chains_per_device, 1))
        elif init.shape != (num_devices, chains_per_device, self.N):
             # Attempt reshape
             init = init.reshape((num_devices, chains_per_device, self.N))

        # --- Define Parallel Logic ---
        def device_run_fn(d_params, d_init_pos, d_key):
            # Split keys for each chain on this device
            d_keys = random.split(d_key, chains_per_device)

            # Single chain logic
            def single_chain_scan(pos, k):
                def step_fn(carry, _):
                    p, current_k = carry
                    k_step, k_next = random.split(current_k)
                    new_p, moved = self.mcstep_psi(k_step, p, d_params, stepsize)
                    return (new_p, k_next), (new_p, moved)
                
                _, (hist, moved_hist) = scan(step_fn, (pos, k), length=total_steps)
                return hist, moved_hist

            # Vectorize over chains on this device
            return vmap(single_chain_scan)(d_init_pos, d_keys)

        # --- Execute ---
        pmapped_sampling = pmap(device_run_fn, in_axes=(None, 0, 0))
        raw_samples, raw_moved = pmapped_sampling(params, init, key_devs)

        # --- Post-Processing ---
        # Reshape to (Total Chains, Steps, N)
        flat_samples = raw_samples.reshape(num_chains, total_steps, self.N)
        flat_moved = raw_moved.reshape(num_chains, total_steps)

        # Slice for kept samples
        kept_samples = flat_samples[:, Ntherm::keep, :]
        
        # Flatten to (Total Samples, N)
        final_samples = kept_samples.reshape(-1, self.N)
        final_samples_prime = self.sample_prime(final_samples)
        
        acc_rate = jnp.mean(flat_moved)

        return final_samples, final_samples_prime, acc_rate

    @partial(jit, static_argnums=(0,))
    def sample_prime(self, samples):
        """Creates the prime coordinates (x') by swapping first and last particle."""
        x = jnp.asarray(samples)
        return x.at[:, -1].set(x[:, 0])