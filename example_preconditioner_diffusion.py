#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:44:29 2022

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

# get ride of the spaces id
# lhs = LincombOperator([NumpyMatrixOperator(op.matrix) for op in lhs.operators], lhs.coefficients)
# rhs = LincombOperator([NumpyMatrixOperator(op.matrix) for op in rhs.operators], rhs.coefficients)

Ru = fom.h1_0_product
# Ru = IdentityOperator(lhs.source)

# Perform a Cholesky decomposition of the stiffness matrix Ru
Qu = CholeskyOperator(Ru.matrix, source_id=Ru.source.id, range_id=Ru.range.id)
# Qu = IdentityOperator(lhs.source)

# =============================================================================
# A function to generate a parameter set
# =============================================================================

# Let's consider a log-uniform distribution of the parameters.
mu_min = 0.1
mu_max = 10

def generate_mu_set(n_set, seed=None):
    print(f"Generating mu_set with seed {seed}")
    n_set = int(n_set)
    a, b = np.log(mu_min), np.log(mu_max)
    parameter_space = fom.parameters.space(a, b)
    n_param = parameter_space.parameters['diffusion']
    sampler = LatinHypercube(d=n_param, seed=seed)
    diff_coefs = np.exp((b - a) * sampler.random(n=n_set) + a)
    
    Mu_set = np.array([
        Mu({'diffusion': diff_coefs[i, :]}) for i in range(n_set)
        ])
    return Mu_set


# generate a test set
mu_test = generate_mu_set(300, seed=300)
u_test = lhs.source.empty()
for i in range(len(mu_test)):
    if i%100 == 0:
        print(f"Generating {i}-th u for test set")
    mu = mu_test[i]
    u = lhs.apply_inverse(rhs.as_range_array(mu), mu)
    u_test.append(u)

# =============================================================================
# Generate the embeddings and the sketches
# =============================================================================

# We generate the primal and the online embedding with some arbitrary k. 
# The certification embedding is generated according to the a priori bounds.
# In the primal_sketch, the full basis is stored (Full_basis is True).

N = lhs.source.dim

primal_embedding = IdentityEmbedding(source_dim=lhs.source.dim, source_id=lhs.source.id, sqrt_product=Qu)
online_embedding = GaussianEmbedding(source_dim=primal_embedding.range.dim, epsilon=0.5, delta=0.1/(50*1000), oblivious_dim=1)
certif_embedding = GaussianEmbedding(source_dim=lhs.source.dim, epsilon=0.5, delta=0.1/(50*1000), oblivious_dim=1, sqrt_product=primal_embedding)


primal_sketch = SketchedRom(lhs, rhs, embedding=primal_embedding, full_basis=True, product=Ru)
online_sketch = SketchedRom(lhs, rhs, embedding=online_embedding)
certif_sketch = SketchedRom(lhs, rhs, embedding=certif_embedding, product=Ru)

# =============================================================================
# Generate the reduced basis
# =============================================================================

# Perform the greedy algorithm with a surrogate model based on the galerkin
# projection. It results in a dictionary, with data relate dto the algorithm.
# The error indicator is the residual's norm divided by the rhs's norm.

r_max = 50
tol = 1e-2
n_train = 500

surrogate = SketchedSurrogate(
    primal_sketch, online_sketch, certif_sketch, 'galerkin'
    )

result = surrogate.weak_greedy(
    r_max=r_max, tol=tol, mu_generator=generate_mu_set, n_train=n_train
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
plt.show()

# =============================================================================
# Check performances on test set
# =============================================================================

# compute the coefficients of the sketched galerkin projection on the test set
ar_gal, _ = surrogate.solve_rom(mu_test)

# compute the orthogonal projection and the galerkin projection
ur = primal_sketch.Ur
u_proj = ur.lincomb(ur.inner(u_test, Ru).T)
u_gal = ur.lincomb(ar_gal.to_numpy())

# compute the relatives errors
norm_test = u_test.norm(Ru)
err_proj = (u_test - u_proj).norm(Ru) / norm_test
err_gal = (u_test - u_gal).norm(Ru) / norm_test

# compute the quantiles on the test set
q = np.linspace(0,1,100)
q_proj = np.quantile(err_proj, q)
q_gal = np.quantile(err_gal, q)
q_ratio = np.quantile(err_gal / err_proj, q)
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].plot(q, q_proj, label="ortho proj")
ax[0].plot(q, q_gal, label="gal proj")
ax[0].set_xlabel("quantile")
ax[0].set_ylabel(r'$ || u_r - u|| ~/~ ||u||$')
ax[0].legend()
ax[1].plot(q, q_ratio)
ax[1].set_xlabel("quantile")
ax[1].set_ylabel(r'$ || u_{gal} - u_{proj}|| ~/~ ||u - u_{proj}||$')
ax[1].legend()
fig.suptitle("quantiles on the test set")
plt.show()

# =============================================================================
# Generate the sketched preconditioner
# =============================================================================

l_max = 2*len(ur)
k_hs = 2*l_max

kwargs = {
    'sigma_u_u':SrhtEmbedding(source_dim=N, range_dim=k_hs, source_id=lhs.source.id),
    'omega_u_u':SrhtEmbedding(source_dim=N, range_dim=k_hs, source_id=lhs.source.id),
    'gamma_u_u':SrhtEmbedding(source_dim=k_hs*k_hs, range_dim=k_hs),
    
    'sigma_u_ur':SrhtEmbedding(source_dim=N, range_dim=k_hs, source_id=lhs.source.id), 
    'omega_u_ur':IdentityEmbedding(source_dim=len(ur)), 
    'gamma_u_ur':SrhtEmbedding(source_dim=k_hs*len(ur), range_dim=k_hs),
    
    'sigma_ur_ur':IdentityEmbedding(source_dim=len(ur)), 
    'omega_ur_ur':IdentityEmbedding(source_dim=len(ur)), 
    'gamma_ur_ur':SrhtEmbedding(source_dim=len(ur)*len(ur), range_dim=k_hs)
    }

sp = SketchedPreconditioner(
    lhs, rhs, ur, primal_embedding, product=Ru, sqrt_product=Qu, **kwargs
    )


# =============================================================================
# Generate the preconditioner by greedy algorithm
# =============================================================================

err_max = np.inf
err2_test_max = []
err_p2_q = []
err_p2_test = []

l=0

while l<l_max and err_max > 0.5:
    # selecting the precond to add
    print("== Greedy for preconditioners : ", l)
    mus = generate_mu_set(300, seed=len(ur)+l)
    coefs, errors = sp.fit(mus, which=2, return_residuals=True)
    err_max = errors.max()
    print("max error", errors.max())
    i_max = errors.argmax()
    mu = mus[i_max]
    P = ImplicitInverseOperator(lhs.assemble(mu).matrix, source_id=lhs.range.id, range_id=lhs.range.id)
    sp.add_preconditioner(P, mu=mu)
    del P
    
    # validation on test sample
    coef2, err2 = sp.fit(mu_test, which=2, solver='minres_ls')
    err2_test_max.append(err2.max())
    
    print("maximal test Delta_U_Ur :  ", err2.max())

    err_p2 = np.zeros(len(mu_test))
    for i in range(len(mu_test)):
        mu = mu_test[i]
        c = coef2[i]
        Ar, br = sp._assemble_galerkin_system(mu, c)
        ar = np.linalg.solve(Ar, br).reshape(-1)
        u_p2 = ur.lincomb(ar)
        err_p2[i] = (u_p2 - u_test[i]).norm(Ru) / norm_test[i]
    
    ratio2 = err_p2 / err_gal
    print("maximal error precond :    ", err_p2.max())
    print("maximal ratio precond/gal: ", ratio2.max())
    
    err_p2_q.append(np.quantile(err_p2, q))
    err_p2_test.append(err_p2)
    
    # quasi opti on test set
    quasi_opti = err_p2 / err_proj
    quasi_opti_gal = err_gal / err_proj
    fig, ax = plt.subplots( figsize=(6,6))
    ax.scatter(quasi_opti_gal, quasi_opti, s=1)
    ax.plot([1,quasi_opti_gal.max()], [1,quasi_opti_gal.max()], linestyle='--', c='r')
    ax.set_xlabel(r'$ 1 + \Delta_2 ~/~ (1 - \Delta_3)  $')
    ax.set_ylabel(r'$ || u_r - u_{proj}|| ~/~ ||u - u_{proj}||$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_title('Quasi optimality on test set : preconditioned vs gal')

    
    
    l=l+1

err2_test_max = np.array(err2_test_max)
err_p2_q = np.array(err_p2_q)
err_p2_test = np.array(err_p2_test)
q_quasi = np.quantile(err_p2_test / err_proj, q, axis=1)

# =============================================================================
# plotting
# =============================================================================


fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].plot(q, q_proj, label="ortho proj")
ax[0].plot(q, q_gal, label="gal")
for j in np.arange(l)[0::5]:
    ax[0].plot(q, err_p2_q[j], label=str(j), c='k', alpha=(1+j)/l)
    ax[1].plot(q, q_quasi[:,j], label=str(j), c='k', alpha=(1+j)/l)
ax[0].set_xlabel("quantile")
ax[0].set_ylabel(r'$ || u_r - u|| ~/~ ||u||$')
ax[0].set_ylim(0, q_gal.max())
ax[0].legend()
ax[1].plot(q, q_ratio, label="gal")
ax[1].set_xlabel("quantile")
ax[1].set_ylabel(r'$ || u_r - u_{proj}|| ~/~ ||u - u_{proj}||$')
ax[1].set_ylim(0.9, q_ratio.max())
ax[1].legend()
ax[0].set_title("Relative error")
ax[1].set_title("Quasi optimality error")


# =============================================================================
# Quasi optimality on test :  observed vs upper bound
# =============================================================================

coefs, errors = sp.fit(mu_test, which=2, return_residuals=True)
err1 = sp._evaluate_ls_errors(mu_test, coefs, which=1)
err3 = sp._evaluate_ls_errors(mu_test, coefs, which=3)


quasi_opti = err_p2_test[-1] / err_proj
quasi_opti_theo = 1 + (err2 / (1-err3))
fig, ax = plt.subplots( figsize=(6,6))
ax.scatter(quasi_opti_theo, quasi_opti, s=1)
ax.plot([1,1.5], [1,1.5], linestyle='--', c='r')
ax.set_xlabel(r'$ 1 + \Delta_2 ~/~ (1 - \Delta_3)  $')
ax.set_ylabel(r'$ || u_r - u_{proj}|| ~/~ ||u - u_{proj}||$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Quasi optimality on test set : observed vs upper bound')




# =============================================================================
# Checking if the sketched indicators are good approximations
# =============================================================================

coef2, err2 = sp.fit(mu_test, which=2, solver='minres_ls')
err1 = sp._evaluate_ls_errors(mu_test, coef2, which=1)
err3 = sp._evaluate_ls_errors(mu_test, coef2, which=3)
i_max = err2.argmax()
mu = mu_test[i_max]
Amu = lhs.assemble(mu)
bmu = rhs.as_range_array(mu)
c = coef2[i_max]

PA_lst = []
AhPh_lst = []
for i in range(len(c)):
    P = ImplicitInverseOperator(lhs.assemble(sp.mus[i]).matrix, source_id=lhs.range.id, range_id=lhs.range.id)
    PA_lst.append(P @ Amu)
Ebis = LincombOperator([IdentityOperator(lhs.source)] + PA_lst, np.concatenate(([1], -c)))

d1 = sp.omega_u_u.apply(
    Qu.apply(
        Ebis.apply(
            Ru.apply_inverse(
                Qu.apply_adjoint(
                    sp.sigma_u_u.as_source_array().conj()
                    )
                )
            )
        )
    ).to_numpy().T
dd1 = sp.gamma_u_u.apply(sp.gamma_u_u.source.from_numpy(np.matrix.flatten(d1.T)))
delta1 = dd1.norm()[0]
print("Delta1 without gamma:", np.linalg.norm(d1))
print("Delta1 with gamma   :", delta1)
print("Delta1 with sp      :", err1[i_max])


d2 = Qu.apply(
    Ru.apply_inverse(
        Ebis.apply_adjoint(
            Qu.apply_adjoint(
                Qu.apply(
                    ur
                    )
                )
            )
        )
    )
dd2 = sp.sigma_u_ur.apply(d2).to_numpy().T
delta2 = sp.gamma_u_ur.apply(
    sp.gamma_u_ur.source.from_numpy(np.matrix.flatten(dd2))
    ).norm()[0]

print("Delta2 no sketch    :", np.linalg.norm(d2.to_numpy()))
print("Delta2 with sigma   :", np.linalg.norm(dd2))
print("Delta2 with gamma   :", delta2)
print("Delta2 with sp      :", err2[i_max])


d3 = sp.SUr.inner(
    Qu.apply(
        Ebis.apply(
            ur
            )
        )
    )
dd3 = sp.gamma_ur_ur.apply(sp.gamma_ur_ur.source.from_numpy(np.matrix.flatten(d3.T)))
delta3 = dd3.norm()[0]

print("Delta3 no sketch    :", np.linalg.norm(d3))
print("Delta3 with gamma   :", delta3)
print("Delta3 with sp      :", err3[i_max])

