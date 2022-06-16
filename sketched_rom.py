#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:29:28 2022

@author: pasco
"""

import sys
import numpy as np
from time import perf_counter
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.parameters.base import Mu
from scipy.sparse import csc_matrix

from affine_operations import *
from other_operators import *
from embeddings import *
from sparse_projection import *

class SketchedRom():
    """
    A sketched reduced model
    
    Attributes
    ----------
    lhs : LincombOperator
        The left-hand side of the full problem
    rhs : LincombOperator
        The right-hand side of the full problem
    embedding : RandomEmbedding
        The embedding used.
    output_functional : LincombOperator
        An LincombOperator mapping a given solution to the model output quantity 
        of interest.
    product : NumpyMatrixOperator
        The metric operator such as U.inner(V, product) gives the inner 
        product between U and V.
    mus : list of Mu
        The parameters used to construct the basis
    SUr : NumpyVectorArray
        The sketched reduced basis.
    SVr : LincombOperator
        The sketched image of the reduced basis by the operator.
        SVr = embedding @ product^-1 @ A @ Ur
    SF : LincombOperator
        The sketched affine rhs.
        SF = embedding @ product^-1 @ rhs.
    Ur : NumpyVectorArray
        If full_basis is True, it is the full reduced basis. Else, it is None.
    full_basis : bool
        If True, the full basis will be stored.
    """
    
    def __init__(self, lhs, rhs, embedding=None, output_functional=None, product=None, 
                 full_basis=False):
                
        for key, val in locals().items():
            self.__setattr__(key, val)
        self.mus = []
        self.SUr = embedding.range.empty()
        self.SVr = None
        self.SF = None
        self.Ur = lhs.source.empty()
        if product is None:
            self.product = IdentityOperator(lhs.source)
        
    def _sketch_rhs(self):
        if self.embedding is None:
            SF = None
            print("No embedding to sketch the rhs.")
        else:
            prod_inv = InverseOperator(self.product)
            op = ConcatenationOperator((self.embedding, prod_inv))
            SF = op_compose_lincomb(op, self.rhs)
        return SF


    def _sketch_u(self, U):
        SU = self.embedding.apply(U)
        return SU


    def _sketch_sv(self, U):
        prod_inv = InverseOperator(self.product)
        op = ConcatenationOperator((self.embedding, prod_inv))
        AUr = apply_affine(self.lhs, U)
        SVr = op_compose_lincomb(op, AUr)
        return SVr
    
    
    def _sketch_output_linear(self, U):
        if self.output_functional is not None:
            lU = None
        else:
            assert self.output_functional.range.dim == 1
            lU = apply_affine(self.output_functional, U)
        return lU
    
    
    def add_vectors(self, U, mus, check_mu=True):
        if isinstance(mus, Mu):
            mus = [mus]
        if self.SF is None :
            self.SF = self._sketch_rhs()
        if self.full_basis: 
            self.Ur.append(U)
        for mu in mus:
            # Check if this value has already been added
            for mu_added in self.mus:
                assert not(check_mu) or not(mu.allclose(mu_added))
            self.mus.append(mu)
        SU = self._sketch_u(U)
        SV = self._sketch_sv(U)
        SVr_next = lincomb_join(self.SVr, SV, axis=1)   
        self.SUr.append(SU)
        self.SVr = SVr_next
        # TO DO: extending the reduced output functional
    
    
    def orthonormalize_basis(self, T=None, offset=0):
        """
        

        Parameters
        ----------
        T : Operator, optional
            The operator such as SUr @ T is orthonormal. If None, T is computed. 
            The default is None.
        offset : int, optional
            If T is None, the offset for the gram-schmidt orthonormalization. 
            The default is 0.

        Returns
        -------
        None.

        """
        if T is None:
            dtype = self.SUr.to_numpy().dtype
            Q, R = gram_schmidt(self.SUr, offset=offset, return_R=True, reiterate=True, rtol=np.finfo(dtype).resolution)
            T = ImplicitInverseOperator(
                csc_matrix(R, dtype=dtype), source_id=self.SVr.source.id, 
                range_id=self.lhs.source.id
                )
        else:
            SUr_H = T.apply_adjoint(T.source.from_numpy(self.SUr.to_numpy().conj().T))
            self.SUr = self.SUr.space.from_numpy(SUr_H.to_numpy().conj().T)
        
        if self.full_basis:
            Ur_H = T.apply_adjoint(T.source.from_numpy(self.Ur.to_numpy().conj().T))
            self.Ur = self.Ur.space.from_numpy(Ur_H.to_numpy().conj().T)

        self.SVr = lincomb_compose_op_implicit(self.SVr,T)


    def from_sketch(self, sketch, seed=None):
        """
        Generate the sketch from another SketchedRom by embedding the already 
        embedded quantities of sketch

        Parameters
        ----------
        sketch : SketchedRom
            The SketchedRom to sketch again.
        see : int
            The seed 
        Returns
        -------
        None.

        """
        self.embedding.set_seed(seed)
        embedding = self.embedding
        self.SUr = embedding.apply(sketch.SUr)
        self.SVr = op_compose_lincomb(embedding, sketch.SVr)
        self.SF = op_compose_lincomb(embedding, sketch.SF)

    
    def solve_rom(self, mus, projection, **kwargs):
        if not isinstance(mus, list) and not isinstance(mus, np.ndarray):
            mus = [mus]
        if projection in ('galerkin', 'minres_ls', 'minres_normal'):
            coefs, times = self.rb_projection(mus, projection)
        elif projection in ('sparse_minres'):
            coefs, times = self.sparse_minres_projection(mus, **kwargs)
        else:
            print("Projection not valid")
        return coefs, times
    
    
    def rb_projection(self, mus, projection):
        if self.SVr is None:
            times = {'offline_assembling': 0., 'rom_solving': 0., 'offline_assembling': 0.}
            coefs = None
        else:
            coefs, times = self._rb_projection(mus, projection)
        return coefs, times
    
    
    def _rb_projection(self, mus, projection):
        print(f"***RB {projection}")
        coefs = self.SVr.source.empty()
        ls = False
        times = {'offline_assembling': 0., 'rom_solving': 0.}
        
        tic = perf_counter()
        if projection == 'galerkin': 
            red_lhs, red_rhs = self._galerkin_system()
        elif projection == 'minres_ls':
            red_lhs, red_rhs = self._minres_ls_system()
            ls = True
        elif projection == 'minres_normal':
            red_lhs, red_rhs = self._minres_normal_system()
        else:
            print("WARNING : Projection not valid.")
        times['offline_assembling'] = perf_counter() - tic
        
        tic = perf_counter()
        for mu in mus:
            # compute the reduced solutions
            coef = red_lhs.apply_inverse(red_rhs.as_range_array(mu), mu, least_squares=ls)
            coefs.append(coef)
        times['rom_solving'] = perf_counter() - tic

        return coefs, times


    def _galerkin_system(self):
        SUr = self.SUr
        SVr = self.SVr
        SF = self.SF
        op = NumpyMatrixOperator(SUr.to_numpy().conj(), source_id=SVr.range.id, range_id=SVr.source.id)
        reduced_lhs = op_compose_lincomb(op, SVr)
        reduced_rhs = op_compose_lincomb(op, SF)
        return reduced_lhs, reduced_rhs


    def _minres_ls_system(self):
        reduced_lhs = self.SVr
        reduced_rhs = self.SF
        return reduced_lhs, reduced_rhs
        
        
    def _minres_normal_system(self):
        SVr = self.SVr
        SF = self.SF
        reduced_lhs = lincomb_adjoint_compose_lincomb(SVr, SVr)
        reduced_rhs = lincomb_adjoint_compose_lincomb(SVr, SF)
        return reduced_lhs, reduced_rhs


    def residual_norms(self, coefs, mus, return_residuals=False):
        if self.SVr is None:
            residuals = self.embedding.range.empty()
            errors = np.ones(len(mus))
        else:
            errors, residuals = self._residual_norms(coefs, mus, return_residuals)
        return errors, residuals


    def _residual_norms(self, coefs, mus, return_residuals=False):
        """
        Compute the residual norms over mus by first computing the sketched 
        residual vectors.
        
        Parameters
        ----------
        coefs : NumpyVectorArray
            The reduced coefficients over mus.
        mus : list of Mu
            The parameters on which the residual norm is to be calculated.
        return_residuals : bool
            If True, the sketched residuals are returned. The default is False.

        Returns
        -------
        errors : np.ndarray
            The relative errors, defined by the residual norm over the rhs norm.
        residuals : NumpyVectorArray
            If return_residuals is True, contains the residuals. Else, empty.

        """
        
        SVr = self.SVr
        SF = self.SF
        residuals = SVr.range.empty()
        errors = np.ones(len(mus))
        
        for i in range(len(mus)):
            SF_mu = SF.as_range_array(mus[i])
            residual = SVr.apply(coefs[i], mus[i]) - SF_mu
            error = residual.norm()[0] / SF_mu.norm()[0]
            errors[i] = error
            if return_residuals:
                residuals.append(residual)
        return errors, residuals

    
    def residual_norms_reduced(self, coefs, mus):
        """
        Compute the residual norms over mus using the online efficient 
        representation of the squared residual norm.

        Parameters
        ----------
        coefs : NumpyVectorArray
            The reduced coefficients over mus.
        mus : list of Mu
            The parameters on which the residual norm is to be calculated.

        Returns
        -------
        errors : np.ndarray
            the relative errors, defined by the residual norm over the rhs norm.

        """
        SVr = self.SVr
        SF = self.SF
        M1 = lincomb_adjoint_compose_lincomb(SVr, SVr)
        M2 = lincomb_adjoint_compose_lincomb(SVr, SF)
        M3 = lincomb_adjoint_compose_lincomb(SF, SF)
        errors = np.ones(len(mus))
        
        for i in range(len(mus)):
            m3 = M3.as_range_array(mus[i]).real.to_numpy()[0,0]
            error = M1.apply2(coefs[i], coefs[i], mu=mus[i]).real[0, 0] / m3
            error = error + 1 - 2 * M2.apply_adjoint(coefs[i].conj(), mu=mus[i]).real.to_numpy()[0, 0] / m3
            errors[i] = np.sqrt(max(0,error))
        
        return errors

    
    def sparse_minres_projection(self, mus, r=None, stol=None, solver='stepwise'):
        """
        Use a particular solver to approximate the solution of the least 
        square problem 
            \min_x \|x\|_0 such that \| Vx - F \| < stol
        with the constraint \|x\|_0 <=r, for all parameter values.

        Parameters
        ----------
        mus : list of Mu
            the parameter
        r : int, optional
            The maximul number of vectors to use. If None, all vectors can be used. 
        stol : float, optional
            The residual norm target. If None, 0 is taken as tolerance.
        solver : str, optional
            The algorithm used, among stepwise, omp_spams, omp_sklearn. The default is 'stepwise'.

        Returns
        -------
        coefs : NumpyVectorArray
            DESCRIPTION.
        times : dict
            DESCRIPTION.

        """
        if self.SVr is None:
            times = {'offline_assembling': 0., 'rom_solving': 0., 'offline_assembling': 0.}
            coefs = None
        elif r >= self.SVr.source.dim:
            coefs, times = self.rb_projection(mus, 'minres_ls')
        else:
            coefs, times = self._sparse_minres_projection(mus, r, stol, solver)
        return coefs, times



    def _sparse_minres_projection(self, mus, r, stol, solver):
        
        # If omp from sklearn or spams is used, the matrix must be real
        dkind = self.SUr.to_numpy().dtype.kind
        handle_cplx = (dkind=='c') and (solver in ('omp_sklearn', 'omp_spams'))
        print(f"***sparse minres r={r}, stol={stol}, solver={solver}, cplx={handle_cplx}")
        times = {'offline_assembling': 0., 'rom_solving': 0.}
        
        tic = perf_counter()
        if handle_cplx:
            r = 2*r
            SVr = lincomb_complex_to_real(self.SVr)
            SF = lincomb_vector_complex_to_real(self.SF)
        else :
            SVr = self.SVr
            SF = self.SF
        times['offline_assembling'] = perf_counter() - tic
        
        coefs = SVr.source.empty()
        
        tic = perf_counter()
        for mu in mus:
            # compute the reduced solutions
            V = SVr.assemble(mu).matrix
            F = SF.assemble(mu).matrix
            coef, _ = sparse_minres_solver(V, F, r, stol, False, solver)
            coefs.append(coefs.space.from_numpy(coef))
            
        if handle_cplx:
            coefs_np = coefs.to_numpy()
            space = self.SVr.source
            coefs = space.from_numpy(coefs_np[:,:space.dim] + 1.j*coefs_np[:,space.dim:])
        
        times['rom_solving'] = perf_counter() - tic

        return coefs, times
        







