#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:29:28 2022

@author: pasco
"""

import numpy as np
from time import perf_counter
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from scipy.sparse import csc_matrix

from affine_operations import *
from other_operators import *
from embeddings import *


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
    cholesky_product : CholeskyOperator
        The cholesky operator corresponding to the product operator. If noted Q,
        then Q @ Q^T = product.
    cholesky_ordering : str
        The ordering method for the sparse Cholesky factorization. 
        Default is 'default'.
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
                 cholesky_product=None, cholesky_ordering='default', full_basis=False):
                
        for key, val in locals().items():
            self.__setattr__(key, val)
        self.SUr = embedding.range.empty()
        self.SVr = None
        self.SF = None
        self.Ur = lhs.source.empty()
        
        if cholesky_product is None: 
            if product is None:
                self.product = IdentityOperator(lhs.source)
                self.cholesky_product = IdentityOperator(lhs.source)
            else:
                print("No cholesky_product given, performing the Cholesky factorization.")
                self.cholesky_product = CholeskyOperator(
                    product.matrix, mode="auto", ordering_method=cholesky_ordering, 
                    source_id=product.source.id, range_id=product.range.id
                    )
                print("Done")
        
        
    def _sketch_rhs(self):
        if self.embedding is None:
            SF = None
            print("No embedding to sketch the rhs.")
        else:
            chol_inv = InverseOperator(self.cholesky_product)
            theta = ConcatenationOperator((self.embedding, chol_inv))
            SF = op_compose_lincomb(theta, self.rhs)
        return SF


    def _sketch_u(self, U):
        chol_H = AdjointOperator(self.cholesky_product)
        theta = ConcatenationOperator((self.embedding, chol_H))
        SU = theta.apply(U)
        return SU


    def _sketch_sv(self, U):
        chol_inv = InverseOperator(self.cholesky_product)
        theta = ConcatenationOperator((self.embedding, chol_inv))
        AUr = apply_affine(self.lhs, U)
        SVr = op_compose_lincomb(theta, AUr)
        return SVr
    
    
    def _sketch_output_linear(self, U):
        if self.output_functional is not None:
            lU = None
        else:
            assert self.output_functional.range.dim == 1
            lU = apply_affine(self.output_functional, U)
        return lU
    
    
    def add_vectors(self, U, projection=None):
        if self.SF is None :
            self.SF = self._sketch_rhs()
        if self.full_basis: 
            self.Ur.append(U)

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
            Q, R = gram_schmidt(self.SUr, offset=offset, return_R=True)
            T = ImplicitInverseOperator(
                csc_matrix(R), source_id=self.primal_sketch.SVr.source.id, 
                range_id=self.primal_sketch.lhs.source.id
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

    
    def solve_rom(self, mus, projection):
        if projection in ('galerkin', 'minres_ls', 'minres_normal'):
            coefs, times = self.rb_projection(mus, projection)
        return coefs, times
    
    
    def rb_projection(self, mus, projection):
        if self.SVr is None:
            times = {'offline_assembling': 0., 'rom_solving': 0., 'offline_assembling': 0.}
            coefs = None
        else:
            coefs, times = self._rb_projection(mus, projection)
        return coefs, times
    
    
    def _rb_projection(self, mus, projection):
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
            coef = red_lhs.apply_inverse(red_rhs.as_range_array(mu), mu, least_squares=(ls))
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
            errors, residuals = self._residual_norms(coefs, mus)
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
    

        

    # def _extended_galerkin_system(self, SAr, Sbr, SU, SV):
    #     SUr = self.SUr
    #     SVr = self.SVr
    #     SVr_next = lincomb_join(self.SVr, SV, axis=1)
        
    #     op_col = NumpyMatrixOperator(SUr.to_numpy().conj(), source_id=SV.range.id, range_id=SV.source.id)
    #     op_row = NumpyMatrixOperator(SU.to_numpy().conj(), source_id=SV.range.id, range_id=SV.source.id)
        
    #     new_cols = op_compose_lincomb(op_col, SVr)
    #     new_rows = op_compose_lincomb(op_row, SVr_next)
        
    #     SAr_bis = lincomb_join(SAr, new_cols, axis=1)
    #     SAr_next = lincomb_join(SAr_bis, new_rows, axis=0)
        
    #     Sbr_next = lincomb_join(Sbr, op_compose_lincomb(op_row, self.SF), axis=0)
        
    #     return SAr_next, Sbr_next
        
    
    def sparse_projection(self, mus, projection):
        if self.SVr is None:
            times = {'offline_assembling': 0., 'rom_solving': 0., 'offline_assembling': 0.}
            coefs = None
        else:
            coefs, times = self._sparse_projection(mus, projection)
        return coefs, times

    def _sparse_projection(self, mus, projection):
        print("Not implemented yet")
        pass
        
        