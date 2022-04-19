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
from scipy.sparse.linalg import splu


class CholeskyOperator(Operator):
    """
    
    An operator Q obtained by performing a cholesky factorization of a sparse
    symetric positive--definite matrix. The resulting operator is such as 
    Q @ Q.T = R
    
    Attibutes
    ---------
    matrix : matrix format from scipy.sparse 
        The matrix to factorize. The most efficient format is csc_matrix.
    mode : str
        The algorithm to compute the Cholesky decomposition. See CHOLMOD 
        documentation. Default is 'auto'.
    ordering_method : str
        The ordering algorithm used to order the matrix. See CHOLMOD 
        documentation. Default is 'natural', which means no permutation 
        is performed.
    
    """
    
    def __init__(self, matrix, mode="auto", ordering_method="natural", 
                 source_id=None, range_id=None, name=None):
        self.__auto_init(locals())
        self.linear = True
        self.source = NumpyVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(matrix.shape[0], range_id)
        self.factor = cholesky(matrix, mode=mode, ordering_method=ordering_method)
        
        
    def apply(self, U, mu=None):
        assert U in self.source
        factor = self.factor
        L = factor.L()
        result = factor.apply_Pt(L.dot(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    def apply_inverse(self, U, mu=None, **kwargs):
        assert U in self.range
        factor = self.factor
        result = factor.solve_L(factor.apply_P(U.to_numpy().T), False)
        return self.source.from_numpy(result.T)
    
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        factor = self.factor
        Lt = factor.L().T
        result = Lt.dot(factor.apply_P(U.to_numpy().T))
        return self.source.from_numpy(result.T)
    
    
    def apply_inverse_adjoint(self, U, mu=None, **kwargs):
        assert U in self.source
        factor = self.factor
        result = factor.apply_Pt(factor.solve_Lt(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    def matrix(self):
        factor = self.factor
        return factor.apply_Pt(factor.L())


class ImplicitLuOperator(Operator):
    """
    
    An operator obtained by performing a LU factorization of a sparse 
    invertible matrix. It is an implicite representation of the inverse of 
    this matrix based on the scipy.sparse.linalg.SuperLU class.
    
    Attibutes
    ---------
    matrix : scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        The matrix to factorize.
    slu : scipy.sparse.linalg.SuperLU
        
    """
    def __init__(self, matrix, permc_spec="COLAMD", source_id=None, 
                 range_id=None, name=None):
        self.__auto_init(locals())
        self.linear = True
        self.source = NumpyVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(matrix.shape[0], range_id)
        self.slu = splu(matrix, permc_spec=permc_spec)


    def apply(self, U, mu=None):
        operator = NumpyMatrixOperator(self.matrix, self.source_id, self.range_id)
        return operator.apply(U)
    
    
    def apply_inverse(self, U, mu=None):
        assert U in self.range
        slu = self.slu
        result = slu.solve(U.to_numpy().T)
        return self.source.from_numpy(result.T)
    
    
    def apply_adjoint(self, U, mu=None):
        operator = NumpyMatrixOperator(self.matrix, self.source_id, self.range_id)
        return operator.apply_adjoint(U)
    
    
    def apply_inverse_adjoint(self, U, mu=None):
        assert U in self.range
        slu = self.slu
        result = slu.solve(U.to_numpy().T, trans='H')
        return self.source.from_numpy(result.T)