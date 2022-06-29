#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:29:14 2022

@author: pasco
"""

from embeddings import *
from sketched_rom import *
from other_operators import *
from mor_solver import *
from sketched_preconditioner import *

import matplotlib.pyplot as plt
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg

from scipy.stats.qmc import LatinHypercube


# =============================================================================
# Quick note
# =============================================================================

# In this file, we use a pyMOR example. In pyMOR, list of vectors are instances 
# of VectorArray, which are related to some VectorSpace, which have an id (strings).
# As a result, every time a source_id or range_id parameter occures in my own code,
# it aims to stay consitent with pyMOR's rules. However, for an external problem,
# (not from pyMOR example), they can be omitted.

# =============================================================================
# Generate the problem
# =============================================================================

# pyMOR gives some basic examples, such as the thermal blocks diffusion problem.
p = thermal_block_problem((2,2))
fom, _ = discretize_stationary_cg(p, diameter=1/(2**6))
lhs = fom.operator
rhs = LincombOperator([fom.rhs], [1.])
Ru = fom.h1_0_product

# Perform a Cholesky decomposition of the stiffness matrix Ru
Qu = CholeskyOperator(Ru.matrix, source_id=Ru.source.id, range_id=Ru.source.id)


# =============================================================================
# A function to generate a parameter set
# =============================================================================

# Let's consider a log-uniform distribution of the parameters.

def generate_mu_set(n_set, seed=None):
    print(f"Generating mu_set with seed {seed}")
    n_set = int(n_set)
    a, b = np.log(0.1), np.log(10)
    parameter_space = fom.parameters.space(a, b)
    n_param = parameter_space.parameters['diffusion']
    sampler = LatinHypercube(d=n_param, seed=seed)
    diff_coefs = np.exp((b - a) * sampler.random(n=n_set) + a)
    
    Mu_set = np.array([
        Mu({'diffusion': diff_coefs[i, :]}) for i in range(n_set)
        ])
    return Mu_set


# =============================================================================
# Generate the embeddings and the sketches
# =============================================================================

# We generate the primal and the online embedding with some arbitrary k. 
# The certification embedding is generated according to the a priori bounds.
# In the primal_sketch, the full basis is stored (Full_basis is True).

N = lhs.source.dim
k_primal = 1000
k_online = 500

primal_embedding = SrhtEmbedding(source_dim=lhs.source.dim, range_dim=k_primal, source_id=lhs.source.id, sqrt_product=Qu)
online_embedding = GaussianEmbedding(source_dim=primal_embedding.range.dim, range_dim=k_online, source_id=primal_embedding.range.id)
certif_embedding = GaussianEmbedding(source_dim=lhs.range.dim, epsilon=0.5, delta=1e-4, oblivious_dim=1, source_id=lhs.range.id, sqrt_product=primal_embedding)

embeddings = {
    'primal': primal_embedding,
    'online': online_embedding,
    'certif': certif_embedding
    }

primal_sketch = SketchedRom(lhs, rhs, embedding=primal_embedding, product=Ru, full_basis=True)
online_sketch = SketchedRom(lhs, rhs, embedding=online_embedding)
certif_sketch = SketchedRom(lhs, rhs, embedding=certif_embedding, product=Ru)


# =============================================================================
# Generate the surrogate model and perform greedy algorithm
# =============================================================================

# Perform the greedy algorithm with a surrogate model based on the galerkin
# projection. It results in a dictionary, with data relate dto the algorithm.
# The error indicator is the residual's norm divided by the rhs's norm.

surrogate = SketchedSurrogate(
    primal_sketch, online_sketch, certif_sketch, 'galerkin'
    )

result = surrogate.weak_greedy(
    r_max=100, tol=1e-4, mu_generator=generate_mu_set, n_train=100
    )

# Plot the results
fig, ax = plt.subplots()
r = len(result['max_errors'])
ax.plot(range(r), np.log10(result['max_errors']), label='online')
ax.plot(range(r), np.log10(result['max_errors_certif']), label='certif')
ax.set_xlabel('greedy iteration')
ax.set_ylabel(r'error indicator : $max_{mu} ||~r(u_r)~|| ~/~ ||~rhs~||$')
ax.set_title('Greedy algorithm with regenerated parameter set and Galerkin projection')
ax.legend()


# We can also perform a greedy algorithm with a fixed parameter set. 
# Let's also consider the minres projection.
mu_train = generate_mu_set(1e3)

primal_sketch = SketchedRom(lhs, rhs, embedding=primal_embedding, product=Ru, full_basis=True)
online_sketch = SketchedRom(lhs, rhs, embedding=online_embedding)
certif_sketch = SketchedRom(lhs, rhs, embedding=certif_embedding, product=Ru)

surrogate = SketchedSurrogate(
    primal_sketch, online_sketch, certif_sketch, 'minres_ls'
    )

result = surrogate.weak_greedy(
    r_max=100, tol=1e-4, mu_train=mu_train
    )


fig, ax = plt.subplots()
r = len(result['max_errors'])
ax.plot(range(r), np.log10(result['max_errors']), label='online')
ax.plot(range(r), np.log10(result['max_errors_certif']), label='certif')
ax.set_xlabel('greedy iteration')
ax.set_ylabel(r'error indicator : $max_{mu} ||~r(u_r)~|| ~/~ ||~rhs~||$')
ax.set_title('Greedy algorithm with fixed parameter set and minres_ls projection')
ax.legend()


# =============================================================================
# Use the pyMOR visualizer
# =============================================================================

# Since we used a pyMOR example, we can use the associated visualization tool.
fom.visualize(primal_sketch.Ur)

