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
from pymor.algorithms.greedy import WeakGreedySurrogate
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
        The l2->l2 embeddings used for the sketched Frobenius 
        error indicators.
    residual_embedding : RandomEmbedding
        The U->l2 embedding used to estimate the preconditioned residual norm.
        
    """

    def __init__(self, lhs, rhs, Ur, theta, residual_embedding, product=None, 
                 sqrt_product=None, sigma_u_u=None,
                 omega_u_u=None, gamma_u_u=None, sigma_u_ur=None, omega_u_ur=None,
                 gamma_u_ur=None, sigma_ur_ur=None, omega_ur_ur=None, gamma_ur_ur=None,
                 ):
        
        for key, val in locals().items():
            self.__setattr__(key, val)
        
        if product is None or sqrt_product is None:
            self.product = IdentityOperator(lhs.source)
            self.sqrt_product = IdentityOperator(lhs.source)
        
        self.SUr = theta.apply(Ur)
        self.galerkin_lhs_lst = [] # galerkin lhs
        self.galerkin_rhs_lst = [] # galerkin rhs
        self.pa_u_u = []
        self.pa_u_ur = []
        self.pa_ur_ur = []
        self.mus = []
        self.SPVr_lst = []
        self.SPF_lst = [] 
        
        # Compute the range orthonormalizers for A
        dtype = Ur.to_numpy().dtype
        self.dtype = dtype
        ####
        
        # for galerkin
        Q_AUr, _ = lincomb_ortho_range(lhs, Ur, self.product, self.sqrt_product)
        self.range_AUr = Q_AUr
        self.projected_AUr = apply2_affine(lhs, Q_AUr, Ur)
        self.projected_PhUr_lst = []
        
        # for now, we compute the indicators based on the naive affine PiAj
        range_based_indicators = False
        if range_based_indicators:
            # for delta U->U'
            source = self.product.apply_inverse(
                self.sqrt_product.apply_adjoint(sigma_u_u.as_source_array().conj())
                )
            Q, R = lincomb_ortho_range(lhs, source, self.product)
            # self.T_u_u = ImplicitInverseOperator(csc_matrix(R, dtype=dtype))
            self.range_AS_u_u = Q
            self.projected_A_u_u = apply2_affine(lhs, Q, source)
            self.projected_Ph_u_u_lst = []
            
            # for delta U->Ur'
            source = self.product.apply_inverse(
                self.sqrt_product.apply_adjoint(sigma_u_ur.as_source_array().conj())
                )
            Q, R = lincomb_ortho_range(lhs, source, self.product)
            # self.T_u_ur = ImplicitInverseOperator(csc_matrix(R, dtype=dtype))
            self.range_AS_u_ur = Q
            self.projected_A_u_ur = apply2_affine(lhs, Q, source)
            self.projected_Ph_u_ur_lst = []
            
            # for delta Ur->Ur
            source = lhs.source.from_numpy((Ur.to_numpy().T @ sigma_ur_ur.get_matrix()).T)
            self.range_AS_ur_ur = Q_AUr
            self.projected_A_ur_ur = apply2_affine(lhs, Q_AUr, source)
            self.projected_Ph_ur_ur_lst = []


    def add_preconditioner(self, P, mu=None):
        self.mus.append(mu)
        
        # for galerkin lhs
        tic = perf_counter()
        projected_PhUr= self._project_PhUr(P)
        print(f'range(Ar)h @ Ph @ Ur in {perf_counter()-tic:.3f} s')
        self.projected_PhUr_lst.append(projected_PhUr)
        
        # for delta U->U'
        tic = perf_counter()
        self.pa_u_u.append(self._sketch_pa_u_u(P))
        print(f'HS(U->U\') sketched in {perf_counter()-tic:.3f} s')
        
        # for delta U->Ur'
        tic = perf_counter()
        self.pa_u_ur.append(self._sketch_pa_u_ur(P))
        print(f'HS(Ur->Ur\') sketched in {perf_counter()-tic:.3f} s')
        
        # for delta Ur->Ur'
        tic = perf_counter()
        self.pa_ur_ur.append(self._sketch_pa_ur_ur(P))
        print(f'HS(Ur->Ur\') sketched in {perf_counter()-tic:.3f} s')
        
        # for galerkin rhs
        rhs_operators = []
        for bi in self.rhs.operators:
            SF = self.theta.apply(P.apply(bi.as_range_array()))
            mat = self.SUr.inner(SF)
            rhs_operators.append(NumpyMatrixOperator(mat, range_id=self.Ur.space.id))
        galerkin_rhs = LincombOperator(rhs_operators, self.rhs.coefficients)
        self.galerkin_rhs_lst.append(galerkin_rhs)

        # for preconditioned residual
        tic = perf_counter()
        self.SPVr_lst.append(self._sketch_spvr(P))
        print(f'PAUr sketched in {perf_counter()-tic:.3f} s')
        
        tic = perf_counter()
        self.SPF_lst.append(self._sketch_spf(P))
        print(f'PF sketched in {perf_counter()-tic:.3f} s')
        

    def add_preconditioner_range_based(self, P, mu=None):
        self.mus.append(mu)
        
        # for galerkin
        tic = perf_counter()
        projected_PhUr= self._project_PhUr(P)
        print(f'range(Ar)h @ Ph @ Ur in {perf_counter()-tic:.3f} s')
        self.projected_PhUr_lst.append(projected_PhUr)
        
        # for delta U->U'
        tic = perf_counter()
        projected_Ph_u_u = self._project_Ph_u_u(P)
        print(f'range(A @ Sigmah)h @ Ph @ Omega in {perf_counter()-tic:.3f} s')
        self.projected_Ph_u_u_lst.append(projected_Ph_u_u)
        
        # for delta U->Ur'
        tic = perf_counter()
        projected_Ph_u_ur = self._project_Ph_u_ur(P)
        print(f'range(A @ Sigmah)h @ Ph @ Ur @ Omega in {perf_counter()-tic:.3f} s')
        self.projected_Ph_u_ur_lst.append(projected_Ph_u_ur)
        
        # for delta Ur->Ur'
        tic = perf_counter()
        projected_Ph_ur_ur = self._project_Ph_ur_ur(P)
        print(f'range(A @ Ur @ Sigmah)h @ Ph @ Ur @ Omega in {perf_counter()-tic:.3f} s')
        self.projected_Ph_ur_ur_lst.append(projected_Ph_ur_ur)
        
        rhs_operators = []
        for bi in self.rhs.operators:
            SF = self.theta.apply(P.apply(bi.as_range_array()))
            mat = self.SUr.inner(SF)
            rhs_operators.append(NumpyMatrixOperator(mat, range_id=self.Ur.space.id))
        galerkin_rhs = LincombOperator(rhs_operators, self.rhs.coefficients)
        self.galerkin_rhs_lst.append(galerkin_rhs)
        
    
    def _project_PhUr(self, P):
        # for galerkin
        V = P.apply(self.product.apply(self.range_AUr))
        V = self.theta.apply(V)
        U = self.theta.apply(self.Ur)
        result = U.inner(V)
        return result
    
    
    def _project_Ph_u_u(self, P):
        # for delta U->U
        omega_h = self.omega_u_u.as_source_array().conj()
        V = self.sqrt_product.apply_adjoint(omega_h)
        V = P.apply_adjoint(V)
        result = V.inner(self.range_AS_u_u)
        return result
    
    
    def _project_Ph_u_ur(self, P):
        # for delta U->Ur'
        V = P.apply(self.range_AS_u_ur)
        V = self.theta.apply(V)
        U = self.theta.apply(self.Ur)
        mat = U.inner(V)
        omega = self.omega_u_ur.get_matrix()
        result = omega @ mat
        return result
    
    
    def _project_Ph_ur_ur(self, P):
        # for delta Ur->Ur'
        V = P.apply(self.range_AS_ur_ur)
        V = self.theta.apply(V)
        U = self.theta.apply(self.Ur)
        mat = U.inner(V)
        omega = self.omega_ur_ur.get_matrix()
        result = omega @ mat
        return result


    def _assemble_galerkin_system(self, mu, coef):
        
        r = len(self.Ur)
        p = len(self.projected_PhUr_lst)
        
        # first compute the range of AUr
        V = self.projected_AUr.assemble(mu).matrix
        
        # Then assemble the projected range of PhUr and the galerkin rhs
        W = np.zeros((r, V.shape[0]), dtype=V.dtype)
        galerkin_rhs = np.zeros((r,1), dtype=self.theta.dtype)
        for i in range(p):
            W = W + coef[i] * self.projected_PhUr_lst[i]
            galerkin_rhs += coef[i] * self.galerkin_rhs_lst[i].assemble(mu).matrix
        
        galerkin_lhs = np.dot(W, V)
        
        return galerkin_lhs, galerkin_rhs
      

    def solve_rom(self, mus, which=1, alpha=0, solver='minres_ls', **kwargs):
        coef_p, error_p = self.fit(mus, which, alpha, solver, **kwargs)
        coef_gal = np.zeros((len(mus), len(self.Ur)), dtype=self.dtype)
        
        for i in range(len(mus)):
            mu = mus[i]
            c = coef_p[i]
            Ar, br = self._assemble_galerkin_system(mu, c)
            ar = np.linalg.solve(Ar, br).reshape(-1)
            coef_gal[i] = ar
        
        return coef_gal, coef_p
    
    
    def _evaluate_residual_norms(self, mus, coef_gal, coef_p):
        r = len(self.Ur)
        p = len(self.projected_PhUr_lst)
        k = self.residual_embedding.range.dim
        
        norms = np.zeros(len(mus))
        
        
        for i in range(len(mus)):
            mu = mus[i]
            c_gal = coef_gal[i]
            c_p = coef_p[i]
            SPV = np.zeros(k, dtype=self.dtype)
            SPF = np.zeros(k, dtype=self.dtype)
            for j in range(p):
                spjv = self.SPVr_lst[j].assemble(mu)
                aux = spjv.apply(spjv.source.from_numpy(c_gal)).to_numpy().reshape(-1)
                SPV = SPV + c_p[j] * aux
                
                spjf = self.SPF_lst[j].as_range_array(mu).to_numpy().reshape(-1)
                SPF = SPF + c_p[j] * spjf
            residual = SPV - SPF
            norms[i] = np.linalg.norm(residual)
        
        return norms

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
        coefs : (len(mus), n_preconditioners) ndarray
            The coefficients in the preconditioners space.
        errors : (len(mus),) ndarray
            The associated errors.

        """
        tic = perf_counter()
        
        ls_rhs = self._compute_ls_rhs(which, alpha)
        
        coefs = np.zeros((len(mus), len(self.mus)), dtype=ls_rhs.dtype)
        errors = np.zeros(len(mus))
        
        for i in range(len(mus)):
            ls_lhs = self._assemble_ls_matrices(mus[i], which, alpha)
            if solver == 'minres_ls':
                coef, _, _, _  = np.linalg.lstsq(ls_lhs, ls_rhs, rcond=None)
            elif solver in ('omp', 'omp_spams', 'omp_sklearn', 'stepwise'):
                coef, _ = sparse_minres_solver(ls_lhs, ls_rhs, **kwargs)
                coef = coef.reshape(-1,1)
            residual = np.dot(ls_lhs, coef) - ls_rhs
            err = np.linalg.norm(residual)
            err = max(np.finfo(err.dtype).resolution, err)
            errors[i] = err
            coefs[i] = coef[:,0]
        print(f"Fitting on param set in {perf_counter() - tic:.3f}s")
        return coefs, errors
    
    
    def _compute_ls_rhs(self, which=1, alpha=0):
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
        return ls_rhs
    
    
    def _assemble_ls_matrices(self, mu, which=1, alpha=0):
        """
        Assemble least-square matrix based on the affine representation P_i @ A_j.

        Parameters
        ----------
        mu : TYPE
            DESCRIPTION.
        which : TYPE, optional
            DESCRIPTION. The default is 1.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
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

    
    def _evaluate_ls_errors(self, mus, coefs, which=1, alpha=0):
        tic = perf_counter()
        ls_rhs = self._compute_ls_rhs(which, alpha).reshape(-1)
        errors = np.zeros(len(mus))
        for i in range(len(mus)):
            ls_lhs = self._assemble_ls_matrices(mus[i], which, alpha)
            residual = np.dot(ls_lhs, coefs[i]) - ls_rhs
            err = np.linalg.norm(residual)
            errors[i] = err
        print(f"Evaluate error {which} on param set in {perf_counter() - tic:.3f}s")
        return errors
    
    
    def _evaluate_galerkin_cond(self, mus, coefs):
        tic = perf_counter()
        cond = np.zeros(len(mus))
        for i in range(len(mus)):
            mu = mus[i]
            Ar, _ = self._assemble_galerkin_system(mu, coefs[i])
            cond[i] = np.linalg.cond(Ar)
        print(f"Evaluate galerkin cond on param set in {perf_counter() - tic:.3f}s")
        return cond
    
    def _assemble_ls_matrices_range_based(self, mu, which=1, alpha=0):
        """
        Assemble least-square matrix based on the affine representation 
        P_i @ range(A) and range(A)h @ A_j.

        Parameters
        ----------
        mu : TYPE
            DESCRIPTION.
        which : TYPE, optional
            DESCRIPTION. The default is 1.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        if which == 1:
            gamma = self.gamma_u_u
            projected_P_lst = self.projected_Ph_u_u_lst
            right_mat = self.projected_A_u_u.assemble(mu).matrix
        elif which == 2:
            gamma = self.gamma_u_ur
            projected_P_lst = self.projected_Ph_u_ur_lst
            right_mat = self.projected_A_u_ur.assemble(mu).matrix
        elif which == 3:
            gamma = self.gamma_ur_ur
            projected_P_lst = self.projected_Ph_ur_ur_lst
            right_mat = self.projected_A_ur_ur.assemble(mu).matrix
        else:
            print("Selected error indicator not valid")

        W = gamma.range.empty()
        for projected_P in projected_P_lst:
            mat = projected_P @ right_mat
            vec_mat = np.concatenate([mat[:,j] for j in range(mat.shape[1])])
            vec = gamma.apply(gamma.source.from_numpy(vec_mat.T))
            W.append(vec)
        result = W.to_numpy().T
    
        return result


    def _sketch_u_u(self, op):
        sigma_h = self.sigma_u_u.as_source_array().conj()
        omega = self.omega_u_u
        gamma = self.gamma_u_u
        Ru = self.product
        Q = self.sqrt_product

        u = Q.apply_adjoint(sigma_h)
        u = Ru.apply_inverse(u)
        u = op.apply(u)
        u = Q.apply(u)
        u = omega.apply(u).to_numpy().T
        u_vec = np.concatenate([u[:,j] for j in range(u.shape[1])])
        v = gamma.apply(gamma.source.from_numpy(u_vec))
        return v
    
    
    def _sketch_u_ur(self, op):
        # theta = self.theta
        # sigma_h = self.sigma_u_ur.as_source_array().conj()
        # omega = self.omega_u_ur
        # gamma = self.gamma_u_ur
        # Ru = self.product
        # Q = self.sqrt_product
        # SUr = self.SUr
        
        # u = Q.apply_adjoint(sigma_h)
        # u = Ru.apply_inverse(u)
        # u = op.apply(u)
        # u = theta.apply(u)
        # u = SUr.inner(u)
        # u = omega.apply(omega.source.from_numpy(u.T)).to_numpy().T
        # u_vec = np.concatenate([u[:,j] for j in range(u.shape[1])])
        # v = gamma.apply(gamma.source.from_numpy(u_vec))
        
        theta = self.theta
        omega_h = self.omega_u_ur.as_source_array().conj()
        sigma = self.sigma_u_ur
        gamma = self.gamma_u_ur
        Ru = self.product
        Q = self.sqrt_product
        SUr = self.SUr
        
        u = sigma.apply(
            Q.apply(
                Ru.apply_inverse(
                    op.apply_adjoint(
                        theta.apply_adjoint(
                            SUr
                            )
                        )
                    )
                )
            ).to_numpy().T.conj()
        u_vec = np.matrix.flatten(u)
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

    def _sketch_spvr(self, P):
        S = self.residual_embedding
        PVr = op_compose_lincomb(P, apply_affine(self.lhs, self.Ur))
        PF = op_compose_lincomb(P, self.rhs)
        SPVr = op_compose_lincomb(S, PVr)
        return SPVr
        
    def _sketch_spf(self, P):
        S = self.residual_embedding
        PF = op_compose_lincomb(P, self.rhs)
        SPF = op_compose_lincomb(S, PF)
        return SPF


class SketchedPreconditionerSurrogate(WeakGreedySurrogate):
    
    def __init__(self, sketched_preconditioner, preconditioner_generator, which_fit, which_err):
        self.__auto_init(locals())
    
    def evaluate(self, mus, return_all_values=False):
        coefs, _ = self.sketched_preconditioner.fit(mus, self.which_fit, return_residuals=True)
        if self.which_err in (1,2,3,4):
            errors = self.sketched_preconditioner._evaluate_ls_errors(mus, coefs, self.which_err)
        else: #conditionning based
            errors = self.sketched_preconditioner._evaluate_galerkin_cond(mus, coefs)
        
        if return_all_values:
            result = errors
        else:
            i_max = errors.argmax()
            result = (errors.max(), mus[i_max])
        return result
    
    def extend(self, mu):
        P = self.preconditioner_generator(mu)
        self.sketched_preconditioner.add_preconditioner(P, mu)
        


    
    