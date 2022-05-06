#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:01:11 2022

@author: pasco
"""

import numpy as np
from time import perf_counter
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator, LincombOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from scipy.sparse import csc_matrix

from affine_operations import *
from other_operators import *
from embeddings import *
from sketched_rom import *


class SketchedPreconditioner:
    """
    Class implementing preconditioned (sketched) galerking projection and 
    sketched error indicators for preconditioners.
    
    Attributes
    ----------
    lhs : LincombOperator
        The left-hand side of the full problem.
    rhs : LincombOperator
        The right-hand side of the full problem.
    Ur : NumpyVectorArray
        A array of vectors spanning the reduced space approximating the 
        solution manifold.
    theta : RandomEmbedding
        The U->l2 embedding which is a subspace embedding for
    product : Operator
    
    sqrt_product = Operator
    
    sigma_{x}_{y} : RandomEmbedding
    
    
    """

    def __init__(self, lhs, rhs, Ur, theta, product=None, sqrt_product=None, sigma_u_u=None,
                 omega_u_u=None, gamma_u_u=None, sigma_u_ur=None, omega_u_ur=None,
                 gamma_u_ur=None, sigma_ur_ur=None, omega_ur_ur=None, gamma_ur_ur=None
                 ):
        
        for key, val in locals().items():
            self.__setattr__(key, val)
        
        if product is None or sqrt_product is None:
            self.product = IdentityOperator(lhs.source)
            self.sqrt_product = IdentityOperator(lhs.source)
        
        self.SUr = theta.apply(Ur)
        self.SAr = [] # galerkin lhs
        self.Sbr = [] # galerkin rhs
        self.pa_u_u = []
        self.pa_u_ur = []
        self.pa_ur_ur = []
        self.mus = []


    def add_preconditioner(self, P, mu=None):
        self.mus.append(mu)
        tic = perf_counter()
        self.pa_u_u.append(self._sketch_pa_u_u(P))
        print(f'HS(U->U\') sketched in {perf_counter()-tic:.3f} s')
        tic = perf_counter()
        self.pa_u_ur.append(self._sketch_pa_u_ur(P))
        print(f'HS(U->Ur\') sketched in {perf_counter()-tic:.3f} s')
        tic = perf_counter()
        self.pa_ur_ur.append(self._sketch_pa_ur_ur(P))
        print(f'HS(Ur->Ur\') sketched in {perf_counter()-tic:.3f} s')
        tic = perf_counter()
        SAr, Sbr = self._sub_galerkin_system(P)
        print(f'Galerkin subsystems in {perf_counter()-tic:.3f} s')
        
        self.SAr.append(SAr)
        self.Sbr.append(Sbr)
    
    
    def _assemble_galerkin_system(self, mu, coef=None):

        r = len(self.Ur)
        p = len(self.SAr)
        SAr = np.zeros((r,r), dtype=self.theta.dtype)
        Sbr = np.zeros((r), dtype=self.theta.dtype)
        for i in range(p):
            SAr += coef[i] * self.SAr[i].assemble(mu).matrix
            Sbr += coef[i] * self.Sbr[i].assemble(mu).matrix.reshape(-1)
        return SAr, Sbr
    
    
    def _sub_galerkin_system(self, P):
        Ur = self.Ur
        SUr = self.SUr
        lhs_operators = []
        for Ai in self.lhs.operators:
            SVr = self.theta.apply(P.apply(Ai.apply(Ur)))
            mat = SUr.inner(SVr)
            lhs_operators.append(NumpyMatrixOperator(mat, source_id=Ur.space.id, range_id=Ur.space.id))
        gal_lhs = LincombOperator(lhs_operators, self.lhs.coefficients)
        
        rhs_operators = []
        for bi in self.rhs.operators:
            SF = self.theta.apply(P.apply(bi.as_range_array()))
            mat = SUr.inner(SF)
            rhs_operators.append(NumpyMatrixOperator(mat, range_id=Ur.space.id))
        gal_rhs = LincombOperator(rhs_operators, self.rhs.coefficients)
            
        return gal_lhs, gal_rhs
    
    
    def fit(self, mus, which=1, alpha=0, solver='minres_ls', **kwargs):
        """
        Compute the coefficients and the associated errors of the ls minimization
        problem for some error indicator (which).

        Parameters
        ----------
        mus : list of Mu
            DESCRIPTION.
        which : str, optional
            The error indicator to use. If 1 then HS(U,U), if 2 then HS(U,Ur),
            if 3 then HS(Ur,Ur), if 4 then alpha*HS(U,Ur)+HS(Ur,Ur). 
            The default is 1.
        alpha : float, optional
            DESCRIPTION. The default is 0.
        solver : str, optional
            The solver to use, must be one of minres_ls, stepwise, omp, 
            omp_sklearn, omp_spams. The default is minres_ls.
        kwargs:
            the kwargs for the sparse minres projection.

        Returns
        -------
        coefs : () ndarray
            The coefficients in the preconditioners space.
        errors : TYPE
            The associated errors.

        """
        
        I = IdentityOperator(self.lhs.source)
        if which == 1:
            ls_rhs = self._sketch_u_u(I).to_numpy().T
        elif which == 2:
            ls_rhs = self._sketch_u_ur(I).to_numpy().T
        elif which == 3:
            ls_rhs = self._sketch_ur_ur(I).to_numpy().T
        elif which == 4:
            h1 = self._sketch_u_ur(I).to_numpy().T
            h2 = self._sketch_ur_ur(I).to_numpy().T
            ls_rhs = np.block([[alpha * h1], [h2]])
        else:
            print("Selected error indicator not valid")
        
        coefs = np.zeros((len(mus), len(self.pa_ur_ur)), dtype=ls_rhs.dtype)
        errors = np.zeros(len(mus))
        
        for i in range(len(mus)):
            ls_lhs = self._assemble_ls_matrices(mus[i], which)
            if solver == 'minres_ls':
                coef, err, _, _  = np.linalg.lstsq(ls_lhs, ls_rhs.reshape(-1), rcond=None)
            elif solver in ('omp', 'omp_spams', 'omp_sklearn', 'stepwise'):
                coef, _ = sparse_minres_solver(ls_lhs, ls_rhs, **kwargs)
                residual = np.dot(ls_lhs, coef) - ls_rhs.reshape(-1)
                err = np.linalg.norm(residual)**2
            coefs[i] = coef
            errors[i] = np.sqrt(err)
        
        return coefs, errors
    
    
    def _assemble_ls_matrices(self, mu, which=1, alpha=0):
        coef = np.array(self.lhs.evaluate_coefficients(mu))
        if which == 1:
            vec = self.gamma_u_u.range.empty()
            pa_lst = self.pa_u_u
        elif which == 2:
            vec = self.gamma_u_ur.range.empty()
            pa_lst = self.pa_u_ur
        elif which == 3:
            vec = self.gamma_ur_ur.range.empty()
            pa_lst = self.pa_ur_ur
        elif which != 4:
            print("Selected error indicator not valid")

        if which in (1,2,3):
            for pa in pa_lst:
                vec.append(pa.lincomb(coef))
            mat = vec.to_numpy().T
        else:
            m2 = self._assemble_ls_matrices(mu, which=2)
            m3 = self._assemble_ls_matrices(mu, which=3)
            mat = np.block([[alpha * m2], [m3]])
            
        return mat
    
    
    def _sketch_pa_u_u(self, P):
        result = self.gamma_u_u.range.empty()
        for Ai in self.lhs.operators:
            op = ConcatenationOperator((P, Ai))
            v = self._sketch_u_u(op)
            result.append(v)
        return result
            
    
    def _sketch_pa_u_ur(self, P):
        result = self.gamma_u_ur.range.empty()
        for Ai in self.lhs.operators:
            op = ConcatenationOperator((P, Ai))
            v = self._sketch_u_ur(op)
            result.append(v)
        return result


    def _sketch_pa_ur_ur(self, P):
        result = self.gamma_ur_ur.range.empty()
        for Ai in self.lhs.operators:
            op = ConcatenationOperator((P, Ai))
            v = self._sketch_ur_ur(op)
            result.append(v)
        return result
    

    def _sketch_u_u(self, op):
        sigma_h = self.sigma_u_u.as_source_array().conj()
        omega = self.omega_u_u
        gamma = self.gamma_u_u
        Q = self.sqrt_product

        u = Q.apply_adjoint(sigma_h)
        u = op.apply(u)
        u = Q.apply(u)
        u = omega.apply(u).to_numpy().T
        u_vec = np.concatenate([u[:,j] for j in range(u.shape[1])])
        v = gamma.apply(gamma.source.from_numpy(u_vec))
        return v
    
    
    def _sketch_u_ur(self, op):
        theta = self.theta
        sigma_h = self.sigma_u_ur.as_source_array().conj()
        omega = self.omega_u_ur
        gamma = self.gamma_u_ur
        Ru = self.product
        Q = self.sqrt_product
        SUr = self.SUr
        
        u = Q.apply_adjoint(sigma_h)
        u = Ru.apply_inverse(u)
        u = op.apply(u)
        u = theta.apply(u)
        u = SUr.inner(u)
        u = omega.apply(omega.source.from_numpy(u.T)).to_numpy().T
        u_vec = np.concatenate([u[:,j] for j in range(u.shape[1])])
        v = gamma.apply(gamma.source.from_numpy(u_vec))
        return v
    
    
    def _sketch_ur_ur(self, op):
        theta = self.theta
        sigma_h = self.sigma_ur_ur.as_source_array().conj()
        omega = self.omega_ur_ur
        gamma = self.gamma_ur_ur
        SUr = self.SUr
        Ur = self.Ur
        u = np.dot(Ur.to_numpy().T, sigma_h.to_numpy().T)
        u = op.source.from_numpy(u.T)
        u = op.apply(u)
        u = theta.apply(u)
        u = SUr.inner(u)
        u = omega.apply(omega.source.from_numpy(u.T)).to_numpy().T
        u_vec = np.concatenate([u[:,j] for j in range(u.shape[1])])
        v = gamma.apply(gamma.source.from_numpy(u_vec))
        return v

            



if __name__ == '__main__':
    
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg
    p = thermal_block_problem((2, 2))
    fom, _ = discretize_stationary_cg(p, diameter=1/75)
    param_space = fom.parameters.space(0.0001, 1.)
    lhs = fom.operator
    rhs = LincombOperator([fom.rhs], [1.])
    Ru = fom.h1_0_product
    Qu = CholeskyOperator(Ru.matrix, source_id=Ru.source.id, range_id=Ru.source.id)
    N = lhs.source.dim
    
    mus = param_space.sample_randomly(10)
    mus.append(Mu({"diffusion":[0.001, 1, 0.001, 1]}))
    U = lhs.source.empty()
    for mu in mus:
        U.append(lhs.apply_inverse(rhs.as_range_array(mu), mu))
    U = gram_schmidt(U, product=Ru)
    
    mu_test = param_space.sample_randomly(1)
    U_test = lhs.source.empty()
    for mu in mu_test:
        U_test.append(lhs.apply_inverse(rhs.as_range_array(mu), mu))
    
    
    
    theta = SrhtEmbedding(source_dim=lhs.source.dim, range_dim=1000, sqrt_product=Qu, source_id=lhs.source.id)
    
    kwargs = {
        'sigma_u_u':SrhtEmbedding(source_dim=N, range_dim=20, source_id=lhs.source.id),
        'omega_u_u':SrhtEmbedding(source_dim=N, range_dim=20, source_id=lhs.source.id),
        'gamma_u_u':SrhtEmbedding(source_dim=20*20, range_dim=20),
        
        'sigma_u_ur':SrhtEmbedding(source_dim=N, range_dim=20, source_id=lhs.source.id),
        'omega_u_ur':SrhtEmbedding(source_dim=len(U), range_dim=20, source_id=lhs.source.id),
        'gamma_u_ur':SrhtEmbedding(source_dim=20*20, range_dim=20),
        
        'sigma_ur_ur':SrhtEmbedding(source_dim=len(U), range_dim=20, source_id=lhs.source.id),
        'omega_ur_ur':SrhtEmbedding(source_dim=len(U), range_dim=20, source_id=lhs.source.id),
        'gamma_ur_ur':SrhtEmbedding(source_dim=20*20, range_dim=20)
        }
    
    sp = SketchedPreconditioner(lhs, rhs, U, theta, **kwargs)
    
    Ru_inv = InverseOperator(Ru)
    sp.add_preconditioner(Ru_inv)
    for mu in mus:
        P = ImplicitInverseOperator(lhs.assemble(mu).matrix, source_id=lhs.range.id, range_id=lhs.source.id)
        sp.add_preconditioner(P)

    # Test
    
    # mu = mu_test[0]
    # u = U_test[0]
    mu = Mu({"diffusion":[0.01, 1, 0.01, 1]})
    u = lhs.apply_inverse(rhs.as_range_array(mu), mu)
    u_proj = U.lincomb(U.inner(u, Ru).T)

    # Sketched galerkin

    sr = SketchedRom(lhs, rhs, embedding=theta, product=Ru)
    sr.add_vectors(U, mus)
    coef, _ = sr.solve_rom(mu, 'galerkin')

    u_gal = U.lincomb(coef.to_numpy().reshape(-1))
            
    print("Proj error", (u-u_proj).norm(Ru) / u.norm(Ru))
    print("Gal error", (u-u_gal).norm(Ru) / u.norm(Ru))
    print("Gal Proj error", (u_proj-u_gal).norm(Ru) / u.norm(Ru))
    
    # Preconditioned sketched galerkin
    
    pa1 = sp._assemble_ls_matrices(mu, which=1)
    pa2 = sp._assemble_ls_matrices(mu, which=2)
    pa3 = sp._assemble_ls_matrices(mu, which=3)
    
    h1 = sp._sketch_u_u(IdentityOperator(lhs.source))
    h2 = sp._sketch_u_ur(IdentityOperator(lhs.source))
    h3 = sp._sketch_ur_ur(IdentityOperator(lhs.source))
    
    coef, residual, _, _ = np.linalg.lstsq(pa1.to_numpy().T, h1.to_numpy().reshape(-1), rcond=None)
    
    Ar, br = sp._assemble_galerkin_system(mu, coef)
    u_gal = U.lincomb(np.linalg.solve(Ar, br))
            
    print("Proj error", (u-u_proj).norm(Ru) / u.norm(Ru))
    print("Gal error", (u-u_gal).norm(Ru) / u.norm(Ru))
    print("Gal Proj error", (u_proj-u_gal).norm(Ru) / u.norm(Ru))
    
    