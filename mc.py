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

    def sample(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample):
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



    def sample_batched(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample,
                    num_chains=16, key=random.PRNGKey(int(time.time()))):
        total_steps = Nsweeps * keep + Ntherm

        init = jnp.asarray(initial_sample)
        if init.ndim == 1:
            pos0 = jnp.tile(init[None, :], (num_chains, 1))   # (C,N)
        else:
            pos0 = init  # (C,N)

        psi_batched = jax.vmap(self.psi, in_axes=(0, None))  # psi(pos[c], params)

        def one_step(carry, _):
            pos, k = carry
            k, k1, k2 = random.split(k, 3)

            xis = random.uniform(k1, pos.shape, minval=-stepsize, maxval=stepsize)
            newpos = pos + xis

            psi_old = psi_batched(pos, params)
            psi_new = psi_batched(newpos, params)

            denom = jnp.maximum(psi_old**2, 1e-20)
            prob_ratio = (psi_new**2) / denom

            u = random.uniform(k2, (num_chains,))
            accept = u < prob_ratio

            pos_next = jnp.where(accept[:, None], newpos, pos)
            return (pos_next, k), (pos_next, accept)

        (_, _), (hist, moved) = lax.scan(one_step, (pos0, key), xs=None, length=total_steps)
        # hist: (T, C, N)

        kept = hist[Ntherm::keep]              # (Nsweeps, C, N)
        kept = jnp.swapaxes(kept, 0, 1)        # (C, Nsweeps, N)
        final_samples = kept.reshape(-1, self.N)
        final_samples_prime = self.sample_prime(final_samples)
        acc_rate = jnp.mean(moved)

        return final_samples, final_samples_prime, acc_rate




    def sample_parallel(self, params, Nsweeps, Ntherm, keep, stepsize, initial_sample, 
                        num_chains=None, key=None):
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
        if key is None:
            key = random.PRNGKey(int(time.time()))
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