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
        Serial sampling.
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

    def sample_batched(
        self,
        params,
        Nsweeps,
        Ntherm,
        keep,
        stepsize,
        initial_sample,
        num_chains=16,
        key=random.PRNGKey(int(time.time())),
    ):
        """
        CPU batched sampling across cores.

        :param params: Model parameters for psi.
        :param Nsweeps: Number of samples to keep.
        :param Ntherm: Number of thermalization steps.
        :param keep: Interval between kept samples.
        :param stepsize: Metropolis step size.
        :param initial_sample: Initial sample
        :param num_chains: Number of parallel chains to run.
        :param key: JAX random key.
        """
        total_steps = Nsweeps * keep + Ntherm

        init = jnp.asarray(initial_sample)
        if init.ndim == 1:
            pos0 = jnp.tile(init[None, :], (num_chains, 1))  # (C,N)
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

        (_, _), (hist, moved) = lax.scan(
            one_step, (pos0, key), xs=None, length=total_steps
        )
        # hist: (T, C, N)

        kept = hist[Ntherm::keep]  # (Nsweeps, C, N)
        kept = jnp.swapaxes(kept, 0, 1)  # (C, Nsweeps, N)
        final_samples = kept.reshape(-1, self.N)
        final_samples_prime = self.sample_prime(final_samples)
        acc_rate = jnp.mean(moved)

        return final_samples, final_samples_prime, acc_rate

    @partial(jit, static_argnums=(0,))
    def sample_prime(self, samples):
        """Creates the prime coordinates (x') by swapping first and last particle."""
        x = jnp.asarray(samples)
        return x.at[:, -1].set(x[:, 0])
