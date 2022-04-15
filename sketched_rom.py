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
from scipy.sparse import csc_matrix
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
        SF = embedding @ product^-1 @ rhs
    full_basis : bool
        If True, the full basis will be stored.
    Ur : NumpyVectorArray
        If full_basis is True, it is the full reduced basis. Else, it is None.
    
    """
    
    def __init__(self, lhs, rhs, embedding=None, output_functional=None, product=None, 
                 cholesky_product=None, cholesky_ordering='default', full_basis=False):
        
        for key, val in locals().items():
            self.__setattr__(key, val)
        self.SUr = embedding.range.empty()
        self.SVr = None
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
        self.SF = self._sketch_rhs()
        
        
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
        chol_inv = InverseOperator(self.cholesky_product)
        theta = ConcatenationOperator((self.embedding, chol_inv))
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
    
    
    def add_vectors(self, U):
        if self.full_basis: 
            self.Ur.append(U)
        self.SUr.append(self._sketch_u(U))
        self.SVr = lincomb_join(self.SVr, self._sketch_sv(U))
        # TO DO: extending the reduced output functional
    
    
    def orthonormalize_basis(self):
        # TO DO
        pass
    
    
    def from_sketch(self, sketch):
        # TO DO
        pass
        