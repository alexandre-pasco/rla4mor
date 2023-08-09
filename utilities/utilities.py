#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:48:03 2023

@author: apasco
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.projection import project

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
    

def concatenate_operators(operators, axis=0):
    """
    Concatenate a list of lincomb operators along a given axis, by concatenating 
    the i-th affine terms of all operators. This function is usefull when projecting
    a large operator on a large basis, requirering to divide the computations 
    (for RAM issue).
    
    Parameters
    ----------
    operators : list of LincombOperator
        List of lincomb operators, each of them must have the same number of 
        affine terms and the same shape on the given axis. 
    axis : int, optional
        Axis along which the operators are concatenated. See `numpy.concatenate`.
        Default is 0.

    Returns
    -------
    result : LincombOperator
        The concatenated operator.

    """
    op0 = operators[0]
    
    if all(isinstance(op, LincombOperator) for op in operators):
        n_op = len(op0.operators)
        new_operators = []
        for i in range(n_op):
            new_mat = np.concatenate([to_matrix(op.operators[i]) for op in operators], axis=axis)
            new_op = op0.operators[i].with_(matrix=new_mat)
            new_operators.append(new_op)
        result = op0.with_(operators=new_operators)
    
    elif all(isinstance(op, ZeroOperator) for op in operators):
        source = NumpyVectorSpace(np.sum([op.source.dim for op in operators]))
        result = ZeroOperator(op0.range, source)
    
    elif all(not(op.parametric) for op in operators):
        new_mat = np.concatenate([to_matrix(op) for op in operators], axis=axis)
        new_op = op0.with_(matrix=new_mat)
        result = new_op
    
    else:
        TypeError("Operators to concatenate are of different types")
    
    return result

    
def project_block(op, range_basis, source_basis, product=None, max_block_size=None):
    """
    Implementation of `pymor.algorithms.projection.project` by dividing the range
    or source basis into smaller blocks, in order to save RAM. 

    Parameters
    ----------
    op : operator
        See pymor doc.
    range_basis : VectorArray
        See pymor doc.
    source_basis : VectorArray
        See pymor doc.
    product : Operator, optional
        See pymor doc. The default is None.
    max_block_size : int, optional
        Maximal block size. If None, `project` is used. The default is None.

    Returns
    -------
    result : Operator
        Projected operator. See pymor doc.

    """
    if (max_block_size is None) and (source_basis is None) and (range_basis is None):
        result = project(op, range_basis, source_basis, product)
        
    elif not (source_basis is None):
        n = int(np.ceil(len(source_basis) // max_block_size))
        lst = []
        for i in range(n):
            Ui = source_basis[max_block_size*i:max_block_size*(i+1)]
            opi = project(op, range_basis, Ui, product)
            lst.append(opi)
        result = concatenate_operators(lst, axis=1)
    
    else:
        result = project_block(op.H, None, range_basis, product, max_block_size).H

    return result
    