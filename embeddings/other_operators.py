#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:25:00 2023

@author: apasco
"""

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix
from pymor.operators.constructions import Operator
from sksparse.cholmod import cholesky
from scikits import umfpack



class CholeskyOperator(Operator):
    """
    
    An operator Q obtained by performing a cholesky factorization of a sparse
    symetric positive--definite matrix. The resulting operator is such as 
    Q^H @ Q = R
    
    Attibutes
    ---------
    operator : NumpyMatrixOperator 
        The operator to factorize. The most efficient format for the 
        corresponding matrix is csc_matrix.
    mode : str
        The algorithm to compute the Cholesky decomposition. See CHOLMOD 
        documentation. Default is 'auto'.
    ordering_method : str
        The ordering algorithm used to order the matrix. See CHOLMOD 
        documentation. Default is 'natural', which means no permutation 
        is performed.
    iscomplex : bool
    
    """
    
    def __init__(self, operator, mode="auto", ordering_method="default", iscomplex=False):
        self.__auto_init(locals())
        self.linear = True
        self.source = operator.source
        self.range = operator.range
        self.factor = cholesky(operator.matrix, mode=mode, ordering_method=ordering_method)
        
        
    def apply(self, U, mu=None):
        if self.iscomplex:
            Vr = self._apply_real(U.real, mu)
            Vi = self._apply_real(U.imag, mu)
            result = Vr + 1.j * Vi
        else:
            result = self._apply_real(U, mu)
        return result
    
    def apply_inverse(self, U, mu=None):
        if self.iscomplex:
            Vr = self._apply_inverse_real(U.real, mu)
            Vi = self._apply_inverse_real(U.imag, mu)
            result = Vr + 1.j * Vi
        else:
            result = self.apply_inverse_real(U, mu)
        return result

    def apply_adjoint(self, U, mu=None):
        if self.iscomplex:
            Vr = self._apply_adjoint_real(U.real, mu)
            Vi = self._apply_adjoint_real(U.imag, mu)
            result = Vr + 1.j * Vi
        else:
            result = self._apply_adjoint_real(U, mu)
        return result

    def apply_inverse_adjoint(self, U, mu=None):
        if self.iscomplex:
            Vr = self._apply_inverse_adjoint_real(U.real, mu)
            Vi = self._apply_inverse_adjoint_real(U.imag, mu)
            result = Vr + 1.j * Vi
        else:
            result = self._apply_inverse_adjoint_real(U, mu)
        return result


    def _apply_real(self, U, mu=None):
        
        factor = self.factor
        Lt = factor.L().T
        result = Lt.dot(factor.apply_P(U.to_numpy().T))
        return self.source.from_numpy(result.T)
    
    def _apply_inverse_real(self, U, mu=None, **kwargs):
        factor = self.factor
        result = factor.apply_Pt(factor.solve_Lt(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    def _apply_adjoint_real(self, U, mu=None):
        factor = self.factor
        L = factor.L()
        result = factor.apply_Pt(L.dot(U.to_numpy().T))
        return self.range.from_numpy(result.T)
    
    
    def _apply_inverse_adjoint_real(self, U, mu=None, **kwargs):
        assert U in self.source
        factor = self.factor
        result = factor.solve_L(factor.apply_P(U.to_numpy().T), False)
        return self.source.from_numpy(result.T)

    
    def get_matrix(self):
        factor = self.factor
        return factor.apply_Pt(factor.L()).conj().T


class InverseLuOperator(Operator):
    """
    
    An operator obtained by performing a LU factorization of a sparse 
    invertible matrix. It is an implicite representation of the inverse of 
    this matrix based on the scipy.sparse.linalg.SuperLU class.
    
    Attibutes
    ---------
    operator : NumpyMatrixOperator
        The operator to factorize. The corresponding matrix must be in 
        csc_format
    factorization : scipy.sparse.linalg.SuperLU or scikits.umfpack.umfpack.UmfpackContext, 
        depending if umfpack is used. 
        
    """
    def __init__(self, operator, factorization=None, permc_spec="COLAMD", use_umfpack=True):
        self.__auto_init(locals())
        self.linear = True
        self.source = operator.source
        self.range = operator.range
        matrix = operator.matrix
        if factorization is None:
            if use_umfpack:
                if not (type(matrix) is csc_matrix):
                    Warning("operator.matrix is not in csc_format.")
                    matrix = csc_matrix(matrix)
                if matrix.dtype == complex:
                    family = 'zi'
                else:
                    family = 'di'
                umf_facto = umfpack.UmfpackContext(family=family)
                umf_facto.numeric(matrix)
                self.factorization = umf_facto
            else:
                self.factorization = splu(matrix, permc_spec=permc_spec)


    def apply(self, U, mu=None):
        assert U in self.range
        if self.use_umfpack:
            result = self._apply_umfpack(U.to_numpy()).T
        else:
            slu = self.factorization
            result = slu.solve(U.to_numpy().T)
        return self.source.from_numpy(result.T)
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        if self.use_umfpack:
            result = self._apply_adjoint_umfpack(U.to_numpy()).T
        else: 
            slu = self.factorization
            result = slu.solve(U.to_numpy().T, trans='H')
        return self.source.from_numpy(result.T)


    def apply_inverse(self, U, mu=None):
        return self.operator.apply(U)

    def apply_inverse_adjoint(self, U, mu=None):
        return self.operator.apply_adjoint(U)

    def _apply_umfpack(self, x):
        result = np.zeros(x.shape, dtype=self.operator.matrix.dtype)
        for i in range(len(x)):
            sol = self.factorization(umfpack.UMFPACK_A, self.factorization.mtx, x[i])
            result[i] = sol
        return result
    
    def _apply_adjoint_umfpack(self, x):
        result = np.zeros(x.shape, dtype=self.operator.matrix.dtype)
        for i in range(len(x)):
            sol = self.factorization(umfpack.UMFPACK_At, self.factorization.mtx, x[i])
            result[i] = sol
        return result