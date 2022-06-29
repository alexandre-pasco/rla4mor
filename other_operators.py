#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 09:55:02 2022

@author: pasco
"""

import numpy as np

from pymor.operators.constructions import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator

import ffht
from sksparse.cholmod import cholesky
from scipy.sparse.linalg import splu, LinearOperator


class CholeskyOperator(Operator):
    """
    
    An operator Q obtained by performing a cholesky factorization of a sparse
    symetric positive--definite matrix. The resulting operator is such as 
    Q^H @ Q = R
    
    Attibutes
    ---------
    product_matrix : matrix format from scipy.sparse 
        The matrix to factorize. The most efficient format is csc_matrix.
    mode : str
        The algorithm to compute the Cholesky decomposition. See CHOLMOD 
        documentation. Default is 'auto'.
    ordering_method : str
        The ordering algorithm used to order the matrix. See CHOLMOD 
        documentation. Default is 'natural', which means no permutation 
        is performed.
    
    """
    
    def __init__(self, product_matrix, mode="auto", ordering_method="default", 
                 source_id=None, range_id=None, name=None):
        self.__auto_init(locals())
        self.linear = True
        self.source = NumpyVectorSpace(product_matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(product_matrix.shape[0], range_id)
        self.factor = cholesky(product_matrix, mode=mode, ordering_method=ordering_method)
        
        
    def apply(self, U, mu=None):
        factor = self.factor
        Lt = factor.L().T
        result = Lt.dot(factor.apply_P(U.to_numpy().T))
        return self.source.from_numpy(result.T)
    
    
    def apply_inverse(self, U, mu=None, **kwargs):
        factor = self.factor
        result = factor.apply_Pt(factor.solve_Lt(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    
    def apply_adjoint(self, U, mu=None):
        factor = self.factor
        L = factor.L()
        result = factor.apply_Pt(L.dot(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    def apply_inverse_adjoint(self, U, mu=None, **kwargs):
        assert U in self.source
        factor = self.factor
        result = factor.solve_L(factor.apply_P(U.to_numpy().T), False)
        return self.source.from_numpy(result.T)

    
    def get_matrix(self):
        factor = self.factor
        return factor.apply_Pt(factor.L()).conj().T


class ImplicitInverseOperator(Operator):
    """
    
    An operator obtained by performing a LU factorization of a sparse 
    invertible matrix. It is an implicite representation of the inverse of 
    this matrix based on the scipy.sparse.linalg.SuperLU class.
    
    Attibutes
    ---------
    matrix : scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        The matrix to factorize.
    factorization : scipy.sparse.linalg.SuperLU
        
    """
    def __init__(self, matrix, factorization=None, permc_spec="COLAMD", source_id=None, 
                 range_id=None, name=None):
        self.__auto_init(locals())
        self.linear = True
        self.source = NumpyVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(matrix.shape[0], range_id)
        if factorization is None:
            self.factorization = splu(matrix, permc_spec=permc_spec)


    def apply(self, U, mu=None):
        assert U in self.range
        slu = self.factorization
        result = slu.solve(U.to_numpy().T)
        return self.source.from_numpy(result.T)
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        slu = self.factorization
        result = slu.solve(U.to_numpy().T, trans='H')
        return self.source.from_numpy(result.T)


    def apply_inverse(self, U, mu=None):
        operator = NumpyMatrixOperator(self.matrix, self.source_id, self.range_id)
        return operator.apply(U)


    def apply_inverse_adjoint(self, U, mu=None):
        operator = NumpyMatrixOperator(self.matrix, self.source_id, self.range_id)
        return operator.apply_adjoint(U)

        
        
class ScipyLinearOperator(LinearOperator):
    """
    Class used to wrap a pymor Operator to a scipy LinearOperator, which can
    be used as a preconditioner for iterative solving method like GMRES.
    """
    def __init__(self, operator):
        self.operator = operator
        self.shape = (operator.range.dim, operator.source.dim)
    
    def _matvec(self, x):
        u = self.operator.source.from_numpy(x)
        return self.operator.apply(u).to_numpy().reshape(-1)
    
    def _rmatvec(self, x):
        u = self.operator.range.from_numpy(x)
        return self.operator.apply_adjoint(u).to_numpy().reshape(-1)
    
    
