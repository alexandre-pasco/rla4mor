#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:00:25 2023

@author: apasco
"""

# %%

import numpy as np
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.algorithms.image import estimate_image
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import InverseOperator, LincombOperator, VectorArrayOperator
from pymor.parameters.base import Mu
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace

from embeddings.embeddings import GaussianEmbedding, EmbeddingVectorized
from embeddings.other_operators import CholeskyOperator
from preconditioners.preconditioned_reductor import PreconditionedReductor

# %% Toy problem

grid_shape = (2,2)
problem = thermal_block_problem(grid_shape)
fom, data = discretize_stationary_cg(problem, diameter=1/(2**6))
mu_space = fom.parameters.space(0,1)
lhs = fom.operator
rhs = fom.rhs
Ru = fom.h1_0_product
Qu = CholeskyOperator(Ru)

# %% Reduced basis

mu_basis = mu_space.sample_randomly(20)
u_basis = lhs.source.empty()
for mu in mu_basis:
    u_basis.append(fom.solve(mu))
u_basis = gram_schmidt(u_basis, Ru)

# %% Preconditioner

n_precond = 5
mu_precond = mu_space.sample_randomly(n_precond)
preconditioner = LincombOperator(
    [InverseOperator(lhs.assemble(mu)) for mu in mu_precond],
    [ProjectionParameterFunctional('precond', size=n_precond, index=i) for i in range(n_precond)]
    )

# %%
 
intermediate_bases = dict()
intermediate_bases['lhs'] = estimate_image((lhs,), (), u_basis, product=Ru, riesz_representatives=True)
intermediate_bases['rhs'] = estimate_image((), (rhs,), (), product=Ru, riesz_representatives=True)

# embedding dimension
k_precond = 10

# for u_u
sigma_u_u = GaussianEmbedding(lhs.source, Qu, {'range_dim':k_precond})
omega_u_u = GaussianEmbedding(lhs.source, Qu, {'range_dim':k_precond})
gamma_u_u = EmbeddingVectorized(omega_u_u.range, sigma_u_u.range.dim, {'range_dim':k_precond})

# for u_ur 
sigma_u_ur = GaussianEmbedding(lhs.source, Qu, {'range_dim':k_precond})
omega_u_ur = GaussianEmbedding(NumpyVectorSpace(len(u_basis)), None, {'range_dim':k_precond})
gamma_u_ur = EmbeddingVectorized(omega_u_ur.range, sigma_u_ur.range.dim, {'range_dim':k_precond})

# for ur_ur
sigma_ur_ur = GaussianEmbedding(NumpyVectorSpace(len(u_basis)), None, {'range_dim':k_precond})
omega_ur_ur = GaussianEmbedding(NumpyVectorSpace(len(u_basis)), None, {'range_dim':k_precond})
gamma_ur_ur = EmbeddingVectorized(omega_ur_ur.range, sigma_ur_ur.range.dim, {'range_dim':k_precond})


# %%

def test_hs_estimators():
    reductor = PreconditionedReductor(
        fom = fom, 
        reduced_basis = u_basis,
        source_bases = {'u_ur': None, 'ur_ur': u_basis, 'u_u': None}, 
        range_bases = {'u_ur': u_basis, 'ur_ur': u_basis, 'u_u': None},
        source_embeddings = {'u_ur': sigma_u_ur, 'ur_ur': sigma_ur_ur, 'u_u': sigma_u_u},
        range_embeddings = {'u_ur': omega_u_ur, 'ur_ur': omega_ur_ur, 'u_u': omega_u_u},
        vec_embeddings = {'u_ur': gamma_u_ur, 'ur_ur': gamma_ur_ur, 'u_u': gamma_u_u},
        intermediate_bases = None,
        product = Ru,
        stable_galerkin=False
        )
    
    for mu, op in zip(mu_precond, preconditioner.operators):
        reductor.add_preconditioner(op, mu=mu)

    test_lst = [test_u_u, test_u_ur, test_ur_ur]

    result = all([t(reductor) for t in test_lst])
    
    return result

def test_galerkin():
    reductor = PreconditionedReductor(
        fom = fom, 
        reduced_basis = u_basis,
        source_bases = {'u_ur': None, 'ur_ur': u_basis, 'u_u': None}, 
        range_bases = {'u_ur': u_basis, 'ur_ur': u_basis, 'u_u': None},
        source_embeddings = {'u_ur': sigma_u_ur, 'ur_ur': sigma_ur_ur, 'u_u': sigma_u_u},
        range_embeddings = {'u_ur': omega_u_ur, 'ur_ur': omega_ur_ur, 'u_u': omega_u_u},
        vec_embeddings = {'u_ur': gamma_u_ur, 'ur_ur': gamma_ur_ur, 'u_u': gamma_u_u},
        intermediate_bases = None,
        product = Ru,
        stable_galerkin=False
        )
    
    for mu, op in zip(mu_precond, preconditioner.operators):
        reductor.add_preconditioner(op, mu=mu)

    mu = mu_space.sample_randomly(1)[0]
    mu_p = Mu({'diffusion':mu['diffusion'], 'precond':np.random.normal(size=n_precond)})
    
    # preconditioned galerkin by hand
    B = Ru @ preconditioner @ lhs
    f = Ru @ preconditioner @ rhs
    Bmu = B.apply2(u_basis, u_basis, mu_p)
    fmu = f.apply_adjoint(u_basis, mu_p).to_numpy().reshape(-1)
    
    # preconditioned galerkin by reductor
    Bp, fp = reductor.assemble_rom_system(mu_p)
    
    # check if equal
    result = np.allclose(Bmu, Bp) and np.allclose(fmu, fp)
    
    return result
    
def test_galerkin_stable():
    reductor = PreconditionedReductor(
        fom = fom, 
        reduced_basis = u_basis,
        source_bases = {'u_ur': None, 'ur_ur': u_basis, 'u_u': None}, 
        range_bases = {'u_ur': u_basis, 'ur_ur': u_basis, 'u_u': None},
        source_embeddings = {'u_ur': sigma_u_ur, 'ur_ur': sigma_ur_ur, 'u_u': sigma_u_u},
        range_embeddings = {'u_ur': omega_u_ur, 'ur_ur': omega_ur_ur, 'u_u': omega_u_u},
        vec_embeddings = {'u_ur': gamma_u_ur, 'ur_ur': gamma_ur_ur, 'u_u': gamma_u_u},
        intermediate_bases = intermediate_bases,
        product = Ru,
        stable_galerkin=True
        )
    
    for mu, op in zip(mu_precond, preconditioner.operators):
        reductor.add_preconditioner(op, mu=mu)

    mu = mu_space.sample_randomly(1)[0]
    mu_p = Mu({'diffusion':mu['diffusion'], 'precond':np.random.normal(size=n_precond)})
    
    # preconditioned galerkin by hand
    B = Ru @ preconditioner @ lhs
    f = Ru @ preconditioner @ rhs
    Bmu = B.apply2(u_basis, u_basis, mu_p)
    fmu = f.apply_adjoint(u_basis, mu_p).to_numpy().reshape(-1)
    
    # preconditioned galerkin by reductor
    Bp, fp = reductor.assemble_rom_system(mu_p)
    
    # check if equal
    result = np.allclose(Bmu, Bp) and np.allclose(fmu, fp)
    
    return result


# %%

def test_u_u(reductor):
    mu = mu_space.sample_randomly(1)[0]
    mu_p = Mu({'diffusion':mu['diffusion'], 'precond':np.random.normal(size=n_precond)})
    
    # sketched HS norm by hands
    E = Ru @ preconditioner @ lhs - Ru
    sEmu = gamma_u_u.apply(
        omega_u_u.apply(
            Ru.apply_inverse(
                E.assemble(mu_p).apply(
                    Ru.apply_inverse(
                        sigma_u_u.as_source_array()
                        )
                    )
                )
            )
        )
    n1 = sEmu.norm()[0]
    
    # sketched HS norm by reductor
    n2 = reductor._estimate_hs(mu_p, 'u_u')
    
    # check if equal
    result = np.isclose(n1, n2)
    
    return result

def test_u_ur(reductor):
    mu = mu_space.sample_randomly(1)[0]
    mu_p = Mu({'diffusion':mu['diffusion'], 'precond':np.random.normal(size=n_precond)})

    # sketched HS norm by hands
    E = Ru @ preconditioner @ lhs - Ru
    sEmu = gamma_u_ur.apply(
        omega_u_ur.apply(
            VectorArrayOperator(u_basis, adjoint=True).apply(
                E.assemble(mu_p).apply(
                    Ru.apply_inverse(
                        sigma_u_ur.as_source_array()
                        )
                    )
                )
            )
        )
    n1 = sEmu.norm()[0]
    
    # sketched HS norm by reductor
    n2 = reductor._estimate_hs(mu_p, 'u_ur')
    
    # check if equal
    result = np.isclose(n1, n2)
    
    return result
    
    
def test_ur_ur(reductor):
    mu = mu_space.sample_randomly(1)[0]
    mu_p = Mu({'diffusion':mu['diffusion'], 'precond':np.random.normal(size=n_precond)})

    # sketched HS norm by hands
    E = Ru @ preconditioner @ lhs - Ru
    sEmu = gamma_ur_ur.apply(
        omega_ur_ur.apply(
            VectorArrayOperator(u_basis, adjoint=True).apply(
                E.assemble(mu_p).apply(
                    VectorArrayOperator(u_basis).apply(
                        sigma_ur_ur.as_source_array()
                        )
                    )
                )
            )
        )
    n1 = sEmu.norm()[0]
    
    # sketched HS norm by reductor
    n2 = reductor._estimate_hs(mu_p, 'ur_ur')
    
    # check if equal
    result = np.isclose(n1, n2)
    
    return result


# %%

if __name__ == '__main__':
    
    print('Testing HS estimators:', end='')
    print(f'{test_hs_estimators()}')

    print('Testing preconditioned galerkin system:', end='')
    print(f'{test_galerkin()}')
    
    print('Testing preconditioned galerkin system stable:', end='')
    print(f'{test_galerkin_stable()}')
