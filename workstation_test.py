import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["JAX_ENABLE_X64"]="false"


# os.environ["XLA_FLAGS"]="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16 --xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"]="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=64"

from jax import numpy as jnp
from jax import random, jit, jacfwd, grad, vmap, jvp
from jax.nn import one_hot
from jax.lax import fori_loop
from jax.tree_util import tree_map
from flax import linen as nn
import numpy as np
import time
from tqdm import trange
from functools import partial
import jax.example_libraries.optimizers as jax_opt
import plotly.graph_objects as go

import mc # sampler

# %%
bosonic = True
if bosonic:
    N = 50
else:
    N_up = 2
    N_down = 1
# number of pairwise interactions for the delta function sampling
if bosonic: 
        num_interactions = N * (N - 1) // 2
else:
    num_interactions = N_up * N_down


g = 0 # delta coupling
omega = 1
m = 1
harmonic_omega = 1
sigma = -m*harmonic_omega*g/2 # long range coupling
C = 20 # denominator constant for the symmetrization

G_N_CORES = 64



def astra_energy():
    return (N * harmonic_omega)/2 - m * g**2  * (N*(N**2 - 1))/(24)

true_energy = astra_energy()

print("True energy: ", true_energy)
# compute a 1% bound for the true energy
true_energy_lower = true_energy * 0.99
true_energy_upper = true_energy * 1.01

total_energy = []
total_uncerts = []

fig = go.FigureWidget()

# %%

class MLP(nn.Module):
    features: list[int]  # e.g. [64, 64, 1]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.celu(x)
        x = nn.Dense(self.features[-1])(x)
        return x


@jit
def transform(coords):
    x = coords / C                                  # (N,)
    powers = jnp.cumprod(jnp.broadcast_to(x, (x.shape[0], x.shape[0])), axis=0)
    return jnp.sum(powers, axis=1)

def create_model(rng_key, N, features):
    model = MLP(features=features)
    dummy_x = jnp.ones((N,))          # N inputs of length d → shape (N,d)
    params = model.init(rng_key, dummy_x)["params"]
    return model, params


# %%
model, params = create_model(random.PRNGKey(int(time.time())), N, [50, 100, 200, 200, 100,1])
print(model)

# %%

@jit
def A(x, params):
    """Neural network output, A(x) given a set of parameters"""
    return model.apply({'params': params}, transform(x)).squeeze() + omega * jnp.sum(x**2.)

@jit
def psi(x, params):
    """Wavefunction, psi(x) given a set of parameters"""
    return jnp.exp(-A(x, params))


# %%
mcs = mc.Sampler(psi, (N, 1))

# %%
# s, ar = mcs.sample_batched(params, (10**5)//16, 1000, 5, .3, jnp.zeros((N,)), num_chains=16, key=random.PRNGKey(int(time.time())))
# print("num samples:", s.shape[0])
# print(ar)


# %%
# s, ar = mcs.sample_fast(params, (10**5), 1000, 5, .3, jnp.zeros((N,)), seed=int(time.time()))
# print("num samples:", s.shape[0])
# print(ar)


# %%
pos_initials = jnp.zeros((G_N_CORES, N))
seeds = jnp.arange(G_N_CORES) + int(time.time())
samples, acc = mcs.run_many_chains(params, (10**5)//G_N_CORES, 1000, 5, 0.15, pos_initials, seeds)
print("num samples:", samples.shape[0])
print(acc)

# %%
### Utility functions for pytrees

# multiplies a pytree by a scalar elementwise
def pytree_scalar_mult(scalar, pytree):
    return tree_map(lambda x: scalar * x, pytree)

# adds two pytrees elementwise
def pytree_add(a, b):
    return tree_map(lambda x, y: x + y, a, b)

# returns a pytree whos elements are the mean over the first axis of the input pytree elements
def tree_mean(pytree_batched):
    return tree_map(lambda x: jnp.mean(x, axis=0), pytree_batched)

# generates an empty pytree with the same structure as the input pytree
def tree_zeros_like(pytree):
    return tree_map(jnp.zeros_like, pytree)


# %%
# derivative of the wavefunction with respect to the parameters
dnn_dtheta = jit(grad(psi, 1))
vdnn_dtheta = jit(vmap(dnn_dtheta, in_axes=(0, None), out_axes=0))


grad_psi_x = jit(grad(psi, 0))  # returns (N,)
@partial(jit, static_argnames=("N",))
def laplacian_psi_jvp(x, params, *, N):
    def g(x_):
        return grad_psi_x(x_, params)

    def body(i, acc):
        ei = one_hot(i, N, dtype=x.dtype)
        _, dg = jvp(g, (x,), (ei,))
        return acc + dg[i]

    return fori_loop(0, N, body, jnp.array(0.0, x.dtype))


dA_dtheta = jit(grad(A, 1))
vdA_dtheta = vmap(dA_dtheta, in_axes=(0, None), out_axes=0)

dA_dx = jit(grad(A, 0))
vdA_dx = vmap(dA_dx, in_axes=(0, None), out_axes=0)

A_hessian = jacfwd(jit(grad(A, 0)), 0)
@jit
def d2A_dx2(coords, params):
    return jnp.diag(A_hessian(coords, params))


@jit
def sigma_potential(x):
    # pairwise |xi-xj|
    diffs = jnp.abs(x[:, None] - x[None, :])         # (N,N)
    return sigma * jnp.sum(jnp.triu(diffs, k=1))     # sum_{i<j}


@jit
def Es_nodelta(coords, params):
    return - (1/2) * (1/ psi(coords, params)) * laplacian_psi_jvp(coords, params, N=N) + (1/2) * jnp.sum(coords**2) + sigma_potential(coords)

vEs_nodelta = vmap(Es_nodelta, in_axes=(0,None), out_axes=0)

@jit
def Es_delta(coords, coords_prime, params, alpha, g):
    return num_interactions * g * (psi(coords_prime, params)**2)/(psi(coords, params)**2) * (1/(jnp.sqrt(jnp.pi)*alpha))*jnp.exp(-(coords[-1]/alpha)**2)

vEs_delta = vmap(Es_delta, in_axes=(0,0, None, None, None), out_axes=0)


@jit
def gradient_comp(coords, coords_prime, params, es_nodelta, energy_calc, es_delta):
    return pytree_add(pytree_scalar_mult(2/psi(coords, params) * (es_nodelta - energy_calc), dnn_dtheta(coords, params)), pytree_scalar_mult(2 * es_delta / psi(coords_prime, params), dnn_dtheta(coords_prime, params)))

vgradient_comp = vmap(gradient_comp, in_axes=(0, 0, None, 0, None, 0), out_axes=0)



@partial(jit, static_argnames=("chunk_size","g"))
def compute_stats_and_grad(samples, samples_prime, params, g, *, chunk_size=256):
    n = samples.shape[0]

    # alpha
    ys = samples[:, -1]
    maxabs = jnp.max(jnp.abs(ys))
    alpha = maxabs / jnp.sqrt(-jnp.log(jnp.sqrt(jnp.pi) * 1e-10))

    # energies
    e0 = vEs_nodelta(samples, params)
    ed = vEs_delta(samples, samples_prime, params, alpha, g)
    e = e0 + ed

    mean_e = jnp.mean(e)
    var_e = jnp.mean(e * e) - mean_e * mean_e
    uncert = jnp.sqrt(jnp.maximum(var_e, 0.0)) / jnp.sqrt(n)

    # chunks
    nb = (n + chunk_size - 1) // chunk_size

    grad0 = tree_zeros_like(dnn_dtheta(samples[0], params))

    def body(i, grad_sum):
        start = i * chunk_size
        idx = start + jnp.arange(chunk_size)          # (chunk_size,)
        mask = (idx < n).astype(samples.dtype)        # (chunk_size,)

        idx = jnp.minimum(idx, n - 1)                 # clamp for safe gather

        s_chunk  = samples[idx, :]
        sp_chunk = samples_prime[idx, :]
        e0_chunk = e0[idx]
        ed_chunk = ed[idx]

        grads = vgradient_comp(s_chunk, sp_chunk, params, e0_chunk, mean_e, ed_chunk)

        # multiply each leaf by mask on leading axis, then sum over axis 0
        def masked_sum(x):
            # x has leading dimension chunk_size
            reshape = (chunk_size,) + (1,) * (x.ndim - 1)
            return jnp.sum(x * mask.reshape(reshape), axis=0)

        grads_sum = tree_map(masked_sum, grads)
        return pytree_add(grad_sum, grads_sum)

    grad_sum = fori_loop(0, nb, body, grad0)
    mean_grad = pytree_scalar_mult(1.0 / n, grad_sum)

    return mean_grad, mean_e, uncert

#TODO: Now that the sampling is jitted, we can jit this function as well
def gradient(params, g, num_samples=10**3, thermal=200, skip=5, variation_size=1.0, sampling="serial", chunk_size=256):
    if sampling == "serial":
        samples, _ = mcs.sample_fast(params, num_samples, thermal, skip, variation_size, jnp.zeros((N,)), int(time.time()))
    elif sampling == "parallel":
        seeds = jnp.arange(G_N_CORES) + int(time.time())
        init_positions = jnp.zeros((G_N_CORES, N))
        samples, _ = mcs.run_many_chains(params, num_samples//G_N_CORES, thermal, skip, variation_size, init_positions, seeds )

    samples_prime = mcs.sample_prime(samples)
    return compute_stats_and_grad(samples, samples_prime, params, g, chunk_size=chunk_size)


def step(params, opt_state, step_num, num_samples, thermal, skip, variation_size, g, sampling, chunk_size):
    """
    One optimization step.
    - params: current parameters pytree
    - opt_state: optimizer state (must be carried across steps)
    Returns: (new_params, new_opt_state, energy, uncert)
    """
    grad, energy, uncert = gradient(
        params,
        g,
        num_samples=num_samples,
        thermal=thermal,
        skip=skip,
        variation_size=variation_size,
        sampling=sampling,
        chunk_size=chunk_size,
    )
    new_opt_state = opt_update(step_num, grad, opt_state)
    new_params = get_params(new_opt_state)

    return new_params, new_opt_state, energy, uncert

def update_plot(total_energy, total_uncerts):
    with fig.batch_update():
        fig.data[0].x = np.arange(len(total_energy))
        fig.data[0].y = total_energy
        fig.data[0].error_y.array = total_uncerts

        n = len(total_energy)
        for i in [1, 2, 3]:
            fig.data[i].x = [0, n]
              

def train(params, iterations, num_samples, thermal, skip, variation_size, g, sampling="serial", chunk_size=256, energy_storage=total_energy, uncert_storage=total_uncerts):
    """
    Training loop.
    Returns: (hs, us, ns, final_params)
    """
    hs, us = [], []
    ns = np.arange(iterations)

    # Initialize optimizer state ONCE
    opt_state = opt_init(params)

    pbar = trange(iterations, desc="", leave=True)
    old_params = params

    for step_num in pbar:
        new_params, opt_state, energy, uncert = step(
            old_params,
            opt_state,
            step_num,
            num_samples,
            thermal,
            skip,
            variation_size,
            g,
            sampling=sampling,
            chunk_size=chunk_size,
        )
        

        hs.append(energy)
        us.append(uncert)
        old_params = new_params

        energy_storage.append(energy)
        uncert_storage.append(uncert)
        update_plot(energy_storage, uncert_storage)

        pbar.set_description(f"Energy = {energy}", refresh=True)

        # Use jnp.isnan if energy is a JAX scalar; np.isnan is OK if it's a Python float
        if np.isnan(np.asarray(energy)):
            print("NaN encountered, stopping...")
            #TODO: backtrack to previous params, try again?
            break

    return hs, us, ns, old_params



# %%



energy_trace = fig.add_scatter(
    x=[],
    y=[],
    error_y=dict(type='data', array=[], visible=True),
    mode='lines+markers',
    name='Energy vs Iteration'
)


# add a horizontal line for the true energy
fig.add_trace(go.Scatter(
    x=[0, len(total_energy)],
    y=[true_energy, true_energy],
    mode='lines',
    name='True Energy',
    line=dict(dash='dash', color='red')
))

fig.add_trace(go.Scatter(
    x=[0, len(total_energy)],
    y=[true_energy_lower, true_energy_lower],
    mode='lines',
    name='True Energy Lower Bound',
    line=dict(dash='dot', color='green')
))

fig.add_trace(go.Scatter(
    x=[0, len(total_energy)],
    y=[true_energy_upper, true_energy_upper],
    mode='lines',
    name='True Energy Upper Bound',
    line=dict(dash='dot', color='green')
))

fig.update_layout(
    title='VMC Energy Convergence',
    xaxis_title='Iteration',
    yaxis_title='Energy',
    template='plotly_dark'
)

# display(fig)


# %%
opt_init, opt_update, get_params = jax_opt.adam(10 ** (-3))
# params, N_steps, N_samples, thermalization, skip, variation_size, g, sampling_type, gradient chunk size
resultsa = train(params, 100, 100000, 1000, 2, 0.15, g, sampling="parallel", chunk_size=50000)


# %%
opt_init, opt_update, get_params = jax_opt.adam(10 ** (-4))

resultsb = train(resultsa[3], 100, 100000, 1000, 4, 0.15, g, sampling="parallel", chunk_size=50000)

# %%


# %%



