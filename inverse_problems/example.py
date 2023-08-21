#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:47:00 2023

@author: apasco
"""


import numpy as np
import matplotlib.pyplot as plt
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.algorithms.pod import pod
from pymor.algorithms.simplify import expand, contract
from pymor.operators.constructions import InverseOperator, LincombOperator, VectorArrayOperator
from pymor.parameters.base import Mu
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace

from rla.embeddings import GaussianEmbedding
from inverse_problems.recovery_map import PbdwRecoveryMap, DicRecoveryMap
# from inverse_problems.manifold_distance import *
from utilities.factorization import operator_to_cholesky
from inverse_problems.lars import lars_weighted_path
from inverse_problems.manifold_distance import ResidualDistanceAffine

from pymor.core.logger import set_log_levels

# %%
if __name__ == '__main__':
    
    # %% Create the toy problem
    set_log_levels({'pymor':30})
    grid_shape = (3,3)
    problem = thermal_block_problem(grid_shape)
    fom, data = discretize_stationary_cg(problem, diameter=1/(2**5))
    mu_space = fom.parameters.space(0,1)
    param_bounds = ([0]*mu_space.parameters['diffusion'], [1]*mu_space.parameters['diffusion'])
    lhs = fom.operator
    rhs = fom.rhs
    Ru = fom.h1_0_product
    Qu = operator_to_cholesky(Ru)

    # deaffinise
    # u0 = fom.solve(mu_space.parameters.parse(0.5*np.ones(mu_space.parameters['diffusion'])))
    # rhs = contract(expand(rhs + project(lhs, None, -u0)))
    # rhs = rhs.with_(coefficients= [c.factors[0] for c in rhs.coefficients[:-1]] + [1] )
    # fom = fom.deaffinize(u0)

    # %% Create observation space
    W = np.zeros((50, lhs.source.dim))
    W[np.arange(W.shape[0]), np.random.choice(np.arange(lhs.source.dim), size=W.shape[0], replace=False)] = 1
    W = Ru.apply_inverse(lhs.source.from_numpy(W))
    W = gram_schmidt(W, Ru)
    
    # %% Create reduced basis by POD
    mu_train = mu_space.sample_randomly(200)
    u_train = lhs.source.empty()
    for mu in mu_train:
        u_train.append(fom.solve(mu))
    rb, svals = pod(u_train, Ru, modes=20)
    plt.semilogy(svals)
    plt.show()
    
    # %% Create test set
    mu_test = mu_space.sample_randomly(3)
    u_test = lhs.source.empty()
    for mu in mu_test:
        u_test.append(fom.solve(mu))
        
    obs_test = W.inner(u_test, Ru)
    
    # %% PBDW recovery map
    rm_pbdw = PbdwRecoveryMap(rb, W, product=Ru, log_level=30)
    errors_pbdw = np.zeros(len(rb))
    for i in range(len(rb)):
        rmi = rm_pbdw.project_background(np.arange(i+1))
        ui = rmi.solve(obs_test)
        errors_pbdw[i] = (u_test-ui).norm(Ru).mean()
    plt.semilogy(errors_pbdw)
    plt.xlabel("Reduced space dimension")
    plt.ylabel("PBDW mean test error")
    plt.show()
    
    
    # %% Manifold distance
    S = GaussianEmbedding(lhs.source, Qu, {'range_dim':256}, _seed=0)
    V_dic = (1 / u_train.norm(Ru)) * u_train
    X = V_dic.copy()
    X.append(W)
    reduced_lhs = project(S @ InverseOperator(Ru) @ lhs, None, X)
    reduced_rhs = contract(expand(S @ InverseOperator(Ru) @ rhs))
    
    mdist = ResidualDistanceAffine(reduced_lhs, reduced_rhs, param_bounds, log_level=30)
    rm_dic = DicRecoveryMap(V_dic, W, product=Ru, manifold_distance=mdist, log_level=20)
    
    # %% Evolution of the error with growing dic size
    u_dic = rm_dic.solve(obs_test)
    errors_dic = []
    for i in range(1, len(u_train), 10):
        rmi = rm_dic.project_background(np.arange(i+1))
        ui = rmi.solve(obs_test)
        errors_dic.append( (u_test-ui).norm(Ru).mean() )
    errors_dic = np.array(errors_dic)
    plt.semilogy(np.arange(1, len(V_dic), 10), errors_dic)
    plt.xlabel("Dictionary dimension")
    plt.ylabel("Dic Multi Space mean test error")
    plt.show()
    
    # %%
    # rmi = rm_dic.project_background(np.arange(50))
    # u_path, dist = rmi.solve_path(obs_test[:,4])
    # v,_ = rmi.compute_state_path(obs_test[:,4])
    # w = rmi.compute_correction_path(obs_test[:,4], v)
    # c = np.block([[v],[w]])
    # dist, mu_dist = rmi.manifold_distance.evaluate(c)

    
    
    
    
    
    
    
    