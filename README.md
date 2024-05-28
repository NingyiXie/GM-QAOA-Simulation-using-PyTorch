# GM-QAOA Simulation with PyTorch

This repository provides a PyTorch implementation of the Grover-Mixer Quantum Alternating Operator Ansatz (GM-QAOA) simulation as used in the paper [Performance Upper Bound of Grover-Mixer Quantum Alternating Operator Ansatz](https://arxiv.org/abs/2405.03173). It also includes the experimental data used in the paper.

## Problem Sets

The problem sets used in the paper are stored in the following files:

- `n9_18_maxKcolorable.json`
- `n16_30_maxcuts_regular3.jsonl`
- `18_30_max_k_vertex_cover.json`
- `7_14_tsps.json`

## Experimental Results

The experimental results are stored in the following directories:

- `approx_results`: Contains results for maximizing the expected value.
- `popt_results`: Contains results for maximizing the probability of observing the optimal solution.

## Requirements

We used PyTorch version 1.13.0 for this implementation. Any newer version of PyTorch should also work.

