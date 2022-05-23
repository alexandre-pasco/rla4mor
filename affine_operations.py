#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:34:58 2022

@author: pasco
"""

import numpy as np
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, ConcatenationOperator, AdjointOperator
from pymor.parameters.functionals import ConstantParameterFunctional, ParameterFunctional, ExpressionParameterFunctional, ConjugateParameterFunctional
from pymor.parameters.base import Mu
from pymor.algorithms.gram_schmidt import gram_schmidt


def apply_affine(A, U):
    """
    Build a LincombOperator of NumpyMatrixOperators from the application of 
    the affine terms of A to the vectors in U.

    Parameters
    ----------
    A : LincombOperator
        
    U : NumpyVectorArray
        The vector array

    Returns
    -------
    res : LincombOperator
        The affine terms are A_i @ U 

    """

    operators = []
    for op in A.operators:
        operators.append(
            NumpyMatrixOperator(
                op.apply(U).to_numpy().T,
                source_id=op.source.id,
                range_id=op.range.id,
                name=op.name
            )
        )
    res = LincombOperator(operators, A.coefficients, name=A.name)
    return res


def apply2_affine(A, V, U):
    """
    Build a LincombOperator of NumpyMatrixOperators from the affine terms of A, 
    such as the result is V^H @ A @ U.

    Parameters
    ----------
    A : LincombOperator
        
    V : NumpyVectorArray
        The range_vector_array
    
    U : NumpyVectorArray
        The source vector array
        
    Returns
    -------
    res : LincombOperator
        The affine terms are V^H @ A_i @ U 

    """

    operators = []
    for op in A.operators:
        mat = op.apply2(V, U)
        operators.append(NumpyMatrixOperator(mat))
    res = LincombOperator(operators, A.coefficients)
    return res


def op_compose_lincomb(A, B):
    """
    Build a LincombOperator of ConcatenationOperator from the concatenation of 
    A to the affine terms of B.


    Parameters
    ----------
    A : Operator
     
    B : LincombOperator

    Returns
    -------
    res : LincombOperator
        The operator, which terms are [A @ B_i]

    """
    if B is None: res = None
    else:
        operators = []
        for op in B.operators:
            operators.append(
                NumpyMatrixOperator(
                    A.apply(op.as_range_array()).to_numpy().T,
                    source_id=B.source.id, range_id=A.range.id
                    )
                )
        res = LincombOperator(operators, B.coefficients)
    return res

def lincomb_compose_op_implicit(A,B):
    """
    Build a LincombOperator of ConcatenationOperator from the concatenation of 
    the affine terms of A to B. This function should only be used when B is 
    in the form of an implicit matrix (wrapped in an operator).


    Parameters
    ----------
    A : LincombOperator
        A.operators must be a list of NumpyMatrixOperator with dense 
        rectangular matrix.
    B : Operator
        Implicite operator.
    Returns
    -------
    res : LincombOperator
        The operator, which terms are [A @ B_i]

    """
    operators = []
    for op in A.operators:
        operators.append(
            NumpyMatrixOperator(
                B.apply_adjoint(
                    B.source.from_numpy(op.matrix.conj())
                    ).to_numpy().conj(),
                source_id = B.source.id, range_id=A.range.id
                )
            )
    res = LincombOperator(operators, A.coefficients)
    return res

def lincomb_adjoint_compose_lincomb(A, B):
    """
    Build a LincombOperator of ConcatenationOperator from the concatenation of 
    the affine terms of A^H to the affine terms of B.


    Parameters
    ----------
    A : LincombOperator
     
    B : LincombOperator

    Returns
    -------
    res : LincombOperator
        The operator, which terms are [A_i^H @ B_i].

    """

    operators = []
    coefficients = []
    for i in range(len(A.operators)):
        for j in range(len(B.operators)):
            op = NumpyMatrixOperator(
                A.operators[i].apply_adjoint(B.operators[j].as_range_array()).to_numpy().T,
                source_id=B.source.id, range_id=A.source.id
                )
            
            # The ConjugateParameterFunctional needs a ParameterFunctional in input.
            if not isinstance(A.coefficients[i], ParameterFunctional):
                a = ConstantParameterFunctional(A.coefficients[i])
            else:
                a = A.coefficients[i]
            a_H = ConjugateParameterFunctional(a)
            coef = a_H * B.coefficients[j]
            operators.append(op)
            coefficients.append(coef)
    res = LincombOperator(operators, coefficients, name=A.name + '@' + B.name)
    return res


def lincomb_complex_to_real(A):
    """
    Construct the associated real LincombOperator, containing NumpyMatrixOperator. 
    The affine coefficients are supposed to be real valued. The resulting operator
    can be viewed as a map from R^2n to R^2m, where n and m are respectively 
    the dimensions of the source and range space.
    
    
    Parameters
    ----------
    A : LincombOperator
        The LincombOperator with complex affine terms that needs to be converted 
        in real.

    Returns
    -------
    A_block

    """
    operators = []
    for opi in A.operators:
        opi_mat = opi.matrix
        opi_mat_real = np.block([[opi_mat.real, -opi_mat.imag],
                                 [opi_mat.imag, opi_mat.real]])
        operators.append(NumpyMatrixOperator(
            opi_mat_real, source_id=opi.source.id, range_id=opi.range.id)
            )
    A_block = LincombOperator(operators, A.coefficients)
    return A_block


def lincomb_vector_complex_to_real(B):
    """
    Construct the associated real LincombOperator, containing NumpyMatrixOperator
    of dimensions (n,1). The affine coefficients are supposed to be real valued. 
    The resulting operator is actually a vector in R^2n wrapped as an Operator.
    

    Parameters
    ----------
    B : LincombOperator
        a LincombOperator of linear forms, represented by NumpyMAtrixOperator.

    Returns
    -------
    B_block : LincombOperator

    """
    operators = []
    for op in B.operators:
        op_mat = op.matrix
        op_mat_real = np.block([[op_mat.real],
                                 [op_mat.imag]])
        operators.append(NumpyMatrixOperator(
            op_mat_real, source_id=op.source.id, range_id=op.range.id
            ))
    B_block = LincombOperator(operators, B.coefficients)
    return B_block


def lincomb_join(A, B, axis=0):
    """
    Concatenate the matrices of the operators in A and B to form a new
    LincombOperator, where A and B have the same number of affin terms.
    The coefficients used in the result are the one from A.
    
    This method is used when adding vectors to a SketchedRom.

    Parameters
    ----------
    A : LincombOperator
        A.operators is a list of NumpyMatrixOperator.
    B : LincombOperator
        B.operators is a list of NumpyMatrixOperator

    Returns
    -------
    result : LincombOperator
        If A is None, B is returned.
        
    """
    
    if A is None:
        result = B
    else:
        assert len(A.operators) == len(B.operators)
        operators = []
        for i in range(len(A.operators)):
           matA = A.operators[i].matrix
           matB = B.operators[i].matrix
           mat = np.append(matA, matB, axis=axis)
           op = NumpyMatrixOperator(mat, source_id=A.source.id, range_id=A.range.id)
           operators.append(op)
        result = LincombOperator(operators, A.coefficients)
    return result


def lincomb_select_dofs(A, dofs, axis):
    """
    

    Parameters
    ----------
    A : LincombOperator
        
    dofs : list of int
        
    axis : int
        0 or 1.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    
    operators = []
    if type(dofs) is int:
        dofs = [dofs]
    for op in A.operators:
        if axis == 0:
            mat = op.matrix[dofs,:]
        if axis == 1:
            mat = op.matrix[:,dofs]
        op = NumpyMatrixOperator(mat, source_id=A.source.id, range_id=A.range.id)
        operators.append(op)
    result = LincombOperator(operators, A.coefficients)
    return result



def lincomb_ortho_range(A, U, product):
    """
    Compute an orthonormal basis of the range of A @ Ur based on the 
    affine representation of A, using gram schmidt.

    Parameters
    ----------
    A : LincombOperator
        Seen as a U->U' operator.
    U : NumpyVectorArray
        
    product : Operator
        The inner product matrix for orthonormalisation

    Returns
    -------
    Q : NumpyVectorArray
        The orthonormalised range array.
        
    """
    W = A.source.empty()
    for op in A.operators:
        w = product.apply_inverse(op.apply(U))
        W.append(w)
    Q, R = gram_schmidt(W, return_R=True)
    return Q, R


