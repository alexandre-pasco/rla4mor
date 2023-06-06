#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:34:38 2023

@author: apasco
"""

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix, diags
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.interface import Operator



def splu_symetric(matrix):
    factor = splu(
        matrix, permc_spec='MMD_AT_PLUS_A', 
        diag_pivot_thresh=0, options={'SymmetricMode':True}
        )
    return factor

def lu_to_cholesky(matrix=None, factor=None):
    """
    
    Parameters
    ----------
    matrix : csc_matrix, optional
        Sparse matrix to factorize, has to be symetric positive definite. 
        The default is None.
    factor : SuperLU, optional
        SuperLU factorization of matrix. The default is None.

    Returns
    -------
    Q : csc_matrix
        Cholesky factor, such that Q.conj().T @ Q == matrix .

    """
    if factor is None:
        assert not(matrix is None)
        factor = splu_symetric(matrix)
    
    n = factor.perm_c.shape[0]
    
    P = csc_matrix((np.ones(n), (factor.perm_r, np.arange(n))))
    D = diags(factor.U.diagonal()**0.5)
    
    Q = (P.T @ factor.L @ D).conj().T
    
    return Q


def operator_to_cholesky(operator=None, factor=None):
    """
    
    PYMOR wrapper for the lu_to_cholesky function
    
    Parameters
    ----------
    operator : NumpyMatrixOperator, optional
        Operator to factorize, has to be symetric positive definite. 
        The default is None.
    factor : SuperLU, optional
        SuperLU factorization of operator.matrix. The default is None.

    Returns
    -------
    Q : NumpyMatrixOperator
        Cholesky factor, such that Q.H @ Q == operator .

    """
    try: 
        matrix, source_id, range_id = operator.matrix, operator.source.id, operator.range.id
    except: 
        matrix, source_id, range_id = None, None, None
        
    M = lu_to_cholesky(matrix, factor)
    Q = NumpyMatrixOperator(M, source_id, range_id)
    return Q


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
    symetric : bool, optional.
        If True, the operator to factorized is considered to be symetric 
        positive definite. default is False
    kwargs : dict
        keyword arguments for the factorizing with splu.
        
    """
    def __init__(self, operator, factorization=None, symetric=False, **kwargs):
        self.__auto_init(locals())
        self.linear = True
        self.source = operator.range
        self.range = operator.source
        if factorization is None:
            if symetric:
                self.factorization = splu_symetric(operator.matrix)
            else:
                self.factorization = splu(operator.matrix, **kwargs)


    def apply(self, U, mu=None):
        assert U in self.source
        V = U.to_numpy()
        ptype = np.promote_types(self.operator.matrix.dtype, V.dtype)
        slu = self.factorization
        result = slu.solve(V.T).astype(ptype)
        return self.source.from_numpy(result.T)
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.source
        V = U.to_numpy()
        ptype = np.promote_types(self.operator.matrix.dtype, V.dtype)
        slu = self.factorization
        result = slu.solve(V.T, trans='H').astype(ptype)
        return self.source.from_numpy(result.T)

    def apply_inverse(self, U, mu=None):
        return self.operator.apply(U)

    def apply_inverse_adjoint(self, U, mu=None):
        return self.operator.apply_adjoint(U)



class CholmodOperator(Operator):
    """
    
    An operator Q obtained by performing a cholesky factorization of a sparse
    symetric positive--definite matrix. The resulting operator is such as 
    Q^H @ Q = R.
    
    Note that this class rely on the `scikit-sparse` library, and may not be the
    best approach. One should try using the `operator_to_cholesky` function 
    before trying CholmodOperator.
    
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
        
        try: from sksparse.cholmod import cholesky
        except: raise ImportError("sksparse.cholmod not found")
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



class UmfInverseLuOperator(Operator):
    """
    
    An operator obtained by performing a LU factorization of a sparse 
    invertible matrix. It is an implicite representation of the inverse of 
    this matrix based on the scikits.umfpack library.
    
    Note that this class rely on the `scikits.umfpack` library, and may not be the
    best approach. One should try using the `InverseLuOperator` class, which is based
    on the scipy SuperLU class before trying UmfInverseLuOperator.
    
    Attibutes
    ---------
    operator : NumpyMatrixOperator
        The operator to factorize. The corresponding matrix must be in 
        csc_format
    factorization : scikits.umfpack.umfpack.UmfpackContext
        Factorization based on the umfpack library.
        
    """
    def __init__(self, operator, factorization=None, **kwargs):
        self.__auto_init(locals())
        self.linear = True
        self.source = operator.source
        self.range = operator.range
        matrix = operator.matrix
        try: from scikits.umfpack import UmfpackContext
        except: raise ImportError("scikits.umfpack not found")
        
        if factorization is None:
            if not (type(matrix) is csc_matrix):
                self.logger.warning("operator.matrix is not in csc_format.")
                matrix = csc_matrix(matrix)
            if matrix.dtype == complex:
                family = 'zi'
            else:
                family = 'di'
            umf_facto = UmfpackContext(family=family)
            umf_facto.numeric(matrix)
            self.factorization = umf_facto


    def apply(self, U, mu=None):
        assert U in self.range
        result = self._apply_umfpack(U.to_numpy()).T
        return self.source.from_numpy(result.T)
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        result = self._apply_adjoint_umfpack(U.to_numpy()).T
        return self.source.from_numpy(result.T)


    def apply_inverse(self, U, mu=None):
        return self.operator.apply(U)

    def apply_inverse_adjoint(self, U, mu=None):
        return self.operator.apply_adjoint(U)

    def _apply_umfpack(self, x):
        from scikits.umfpack import UMFPACK_A
        result = np.zeros(x.shape, dtype=self.operator.matrix.dtype)
        for i in range(len(x)):
            sol = self.factorization(UMFPACK_A, self.factorization.mtx, x[i])
            result[i] = sol
        return result
    
    def _apply_adjoint_umfpack(self, x):
        from scikits.umfpack import UMFPACK_At
        result = np.zeros(x.shape, dtype=self.operator.matrix.dtype)
        for i in range(len(x)):
            sol = self.factorization(UMFPACK_At, self.factorization.mtx, x[i])
            result[i] = sol
        return result
