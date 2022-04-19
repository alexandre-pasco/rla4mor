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
    
    
    def add_vectors(self, U):
        if self.SF is None :
            self.SF = self._sketch_rhs()
        if self.full_basis: 
            self.Ur.append(U)
        self.SUr.append(self._sketch_u(U))
        self.SVr = lincomb_join(self.SVr, self._sketch_sv(U))
        # TO DO: extending the reduced output functional
    
    
    def orthonormalize_basis(self, T=None, offset=0):
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
        embedding = self.embedding
        self.SUr = embedding.apply(sketch.SUr)
        self.SVr = op_compose_lincomb(embedding, sketch.SVr)
        self.SF = op_compose_lincomb(embedding, sketch.SF)
    
    


from pymor.algorithms.greedy import WeakGreedySurrogate

class SketchedSurrogate(WeakGreedySurrogate):
    
    def __init__(self, lhs, rhs, embedding=None, output_functional=None, product=None, 
                 cholesky_product=None, cholesky_ordering='default', full_basis=False):
        
        self.sketched_rom = SketchedRom(
            lhs, rhs, embedding, output_functional, product, 
            cholesky_product, cholesky_ordering, full_basis
            )
    
    def evaluate(self, mus, return_all_values=False):
        # Provisional. Only Galerkin for now
        # sketch = self.sketched_rom
        # SUr = sketch.SUr
        # SVr = sketch.SVr
        # SF = sketch.SF
        # op = AdjointOperator(NumpyMatrixOperator(SUr.to_numpy().T, source_id=SVr.range.id))
        # Ar = op_compose_lincomb(op, SVr)
        # br = op_compose_lincomb(op, SF) 
        
        # errors = []
        
        # for mu in mus:
        #     A = Ar.assemble(mu).matrix
        #     b = br.assemble(mu).matrix
        #     coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        #     ar = SVr.source.from_numpy(coef.T)
        #     residual = SVr.apply(ar, mu) - SF.as_range_array(mu)
        #     err = residual.norm()[0]
        #     errors.append(err)
        
        # if return_all_values:
        #     result = np.array(errors)
        # else:
        #     ind = np.argmax(errors)
        #     result = (errors[ind], mus[ind])
        
        # return result
        pass
    
    def extend(self, mu, U=None):
        if isinstance(mu, list):
            mus = mu
        else:
            mus = [mu]

        if U is None:
            lhs = self.sketched_rom.lhs
            rhs = self.sketched_rom.rhs
            U = lhs.source.empty()
            for i in range(len(mus)):
                b = rhs.as_range_array(mus[i])
                U.append(lhs.apply_inverse(b, mus[i]))

        r = len(self.sketched_rom.SUr)
        self.sketched_rom.add_vectors(U)
        self.sketched_rom.orthonormalize_basis(offset=r)
        
        
        
        