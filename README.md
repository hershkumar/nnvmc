### Neural Network Variational Monte Carlo

Implements neural network ansatze for bosonic and fermionic systems in 1 dimension. Written using `jax`.

For methodology, see:
- [Neural network solutions of bosonic quantum systems in one dimension](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.109.034004)
- [Machine learning approach to trapped many-fermion systems](https://journals.aps.org/prc/abstract/10.1103/33jq-ks53)


`mc.py` contains Monte Carlo sampling functions, and `nnvmc.ipynb` provides an example training setup for a bosonic system with delta function contact interactions and a long range pairwise potential.
