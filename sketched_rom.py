#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:29:28 2022

@author: pasco
"""

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import inv

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
            T = InverseOperator(ImplicitLuOperator(csc_matrix(R)))
            self.SUr = Q
        else:
            SUr_H = T.apply_adjoint(T.source.from_numpy(self.SUr.to_numpy().conj().T))
            self.SUr = self.SUr.space.from_numpy(SUr_H.to_numpy().conj().T)
        
        if self.full_basis:
            Ur_H = T.apply_adjoint(T.source.from_numpy(self.Ur.to_numpy().conj().T))
            self.Ur = self.Ur.space.from_numpy(Ur_H.to_numpy().conj().T)

        self.SVr = lincomb_compose_op_implicit(self.SVr,T)


    def from_sketch(self, sketch):
        """
        Generate the sketch from another SketchedRom by embedding the already 
        embedded quantities of sketch

        Parameters
        ----------
        sketch : SketchedRom
            The SketchedRom to sketch again.

        Returns
        -------
        None.

        """
        embedding = self.embedding
        self.SUr = embedding.apply(sketch.SUr)
        self.SVr = op_compose_lincomb(embedding, sketch.SVr)
        self.SF = op_compose_lincomb(embedding, sketch.SF)


    def evaluate_residual(self, coef, mu, SVr=None, SF=None):
        """
        

        Parameters
        ----------
        coef : NumpyVectorArray
            The coefficient of the reduced solution.
        mu : Mu
            
        SVr : Operator, optional
            Can be passed as argument if already assembled. The default is None.
        SF : Operator, optional
            Can be passed in agrument if already assembled. The default is None.

        Returns
        -------
        residual : NumpyVectorArray
            The sketched residual corresponding to the reduced solution.

        """
        if SVr is None or SF is None:
            SVr = self.SVr
            SF = self.SF
        residual = SVr.apply(coeff, mu) - SF.as_range_array(mu)
        return residual
    
    
    def rb_projection(self, mus, projection):
        coefs = self.SVr.source.empty()
        ls = False
        
        if projection == 'galerkin': 
            red_lhs, red_rhs = self._galerkin_system()
        elif projection == 'minres_ls':
            red_lhs, red_rhs = self._minres_ls_system()
            ls = True
        elif projection == 'minres_normal':
            red_lhs, red_rhs = self._minres_normal_system()
        else:
            print("Projection not valid. Galerkin projection performed")

        for mu in mus:
            # compute the reduced solutions
            coef = red_lhs.apply_inverse(red_rhs.as_range_array(mu), mu, least_squares=(ls))
            coefs.append(coef)
        return coefs


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

    def error_estimator(self, coefs, mus, return_residuals=False):
        assert len(mus) == len(coefs)
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
        

        
        
        