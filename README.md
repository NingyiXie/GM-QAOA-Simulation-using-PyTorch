# GM-QAOA Simulation with PyTorch

This repository provides a PyTorch implementation of the Grover-Mixer Quantum Alternating Operator Ansatz (GM-QAOA) simulation as used in the paper [Performance Upper Bound of Grover-Mixer Quantum Alternating Operator Ansatz](https://arxiv.org/abs/2405.03173). It also includes the experimental data used in the paper.

## Problem Sets

The problem sets used in the paper are stored in the following files:

- `n9_18_maxKcolorable.json`
- `n16_30_maxcuts_regular3.json`
- `18_30_max_k_vertex_cover.json`
- `7_14_tsps.json`

## Simulation Methods

Our simulation approach requires obtaining the distribution of objective values for a given problem in advance. The methods to retrieve these distributions for problems adopted in the paper are defined in `obj.py`.

When optimizing GM-QAOA, we utilize PyTorch's automatic differentiation to compute gradients. The `GMQAOA` class defined in `gmqaoa.py` can simulate both GM-QAOA and GM-Th-QAOA.

We employ depth progressive methods and quantum annealing initialization (TQA) to obtain multiple sets of initial parameters. Each set is optimized separately, and the best result is selected.

## Experimental Results

The experimental results are stored in the following directories:

- `approx_result`: Contains results for maximizing the expected value.
- `popt_result`: Contains results for maximizing the probability of sampling the optimal solution.

By running the `.ipynb` files, you can plot the Figures from the paper.

## Requirements

We used PyTorch version 1.13.0 for this implementation. Any newer version of PyTorch should also work.

