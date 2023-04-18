#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:48:03 2023

@author: apasco
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from pymor.operators.constructions import Operator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator


class ScipyLinearOperator(LinearOperator):
    """
    Class used to wrap a pymor Operator to a scipy LinearOperator, which can
    be used as a preconditioner for iterative solving method like GMRES.
    """
    def __init__(self, operator, dtype=None):
        self.operator = operator
        self.shape = (operator.range.dim, operator.source.dim)
        self.dtype = dtype
    
    def _matvec(self, x):
        if len(x.shape) == 2:
            x = x.reshape(-1)
        u = self.operator.source.from_numpy(x)
        return self.operator.apply(u).to_numpy().reshape(-1)
    
    def _rmatvec(self, x):
        if len(x.shape) == 2:
            x = x.reshape(-1)
        u = self.operator.range.from_numpy(x)
        return self.operator.apply_adjoint(u).to_numpy().reshape(-1)
    

def stack_lincomb_operators(operators):
    """
    Merge a list of lincomb operators by h-stacking the i-th affine terms of all
    operators into a new Lincomb Operators. This function is usefull when projecting
    a large operator on a large basis, requirering to divide the computations (for RAM issue).

    Parameters
    ----------
    operators : list of LincombOperator
        List of lincomb operators, each of them must have the same range and
        the same number of affine terms. 

    Returns
    -------
    result : LincombOperator
        The merged operator.

    """
    assert all(isinstance(op, LincombOperator) for op in operators)
    op0 = operators[0]
    n_op = len(op0.operators)
    assert all(len(op.operators) == n_op for op in operators)
    assert all(op.range == op0.range for op in operators)
    new_operators = []
    for i in range(n_op):
        new_op = np.hstack([op.operators[i].matrix for op in operators])
        new_op = NumpyMatrixOperator(new_op, source_id=op0.source.id, range_id=op0.range.id)
        new_operators.append(new_op)
    result = LincombOperator(new_operators, op0.coefficients)
    return result


# def cond_estimate(operator, inverse=None, tol=0, verbose=False):
#     A = ScipyLinearOperator(operator)    
#     if isinstance(inverse, Operator):
#         Ainv = ScipyLinearOperator(inverse)
#     if verbose:
#         print("==Cond number estimation==")
#         print("estimation highest sv")
    
#     # computing the highest singular value
#     _, smax, _ = svds(A, k=1, tol=tol)
    
#     # factorizing if necessary
#     if inverse is None:
#         if verbose:
#             print("factorizing A")
#         Ainv = ScipyLinearOperator(InverseLuOperator(operator))
        
#     if verbose:
#         print("estimation lowest sv")
#     # computing the lowest singular value
#     _, inv_smin, _ = svds(Ainv, k=1, tol=tol)
    
#     smin = 1 / inv_smin
#     cond = smax / smin
#     return cond




def power_iteration(operator, n_iter=100, tol=1e-2, product=None):
    n = operator.range.dim
    v_k = operator.source.from_numpy(np.random.normal(scale=1/np.sqrt(n), size=n))

    k = 0
    convergence = False
    while k<n_iter and not(convergence):
        v_k1 = product.apply_inverse(operator.apply(v_k))
        v_k1_norm = v_k1.norm(product)
        v_k1.scal(1/v_k1_norm)
        v_k = v_k1
        k += 1
    
    ev = v_k1.inner(operator.apply(v_k1), product) / v_k1_norm
    return ev



