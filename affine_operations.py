#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:34:58 2022

@author: pasco
"""

import numpy as np
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, ConcatenationOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.base import Mu

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

    operators = []
    for op in B.operators:
        operators.append(ConcatenationOperator([A, op]))
    res = LincombOperator(operators, B.coefficients, name=A.name + '@' + B.name)
    return res

def lincomb_compose_lincomb(A, B):
    """
    Build a LincombOperator of ConcatenationOperator from the concatenation of 
    the affine terms of A to the affine terms of B.


    Parameters
    ----------
    A : LincombOperator
     
    B : LincombOperator

    Returns
    -------
    res : LincombOperator
        The operator, which terms are [A_i @ B_i].

    """

    operators = []
    coefficients = []
    for i in range(len(A.operators)):
        for j in range(len(B.operators)):
            operators.append(ConcatenationOperator([A.operators[i], B.operators[j]]))
            coefficients.append(A.coefficients[i]* B.coefficients[j])
            
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


def lincomb_complex_to_real(B):
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



if __name__ == '__main__':
    
    parameters = {'mu0': 1, 'mu1': 1}
    alpha0 = ExpressionParameterFunctional('1.0 + mu0[0]',parameters)
    alpha1 = ExpressionParameterFunctional('1.0 + mu0[0] + mu1[0]',parameters)
    operators = [NumpyMatrixOperator(np.random.normal(size=(10,10))) for i in range(2)]
    coefficients = [alpha0, alpha1]
    A = LincombOperator(operators, coefficients)
    mu = Mu({'mu0': [1.], 'mu1': [2.]})
    U = A.source.from_numpy(np.random.normal(size=(3,10)))

    AU_mu = A.apply(U, mu)
    AAU_mu = A.apply(A.apply(U,mu),mu)

    AU = apply_affine(A, U)
    AA = lincomb_compose_lincomb(A, A)
    AAU = apply_affine(AA, U)
    AAU_bis = apply_affine(op_compose_lincomb(A.assemble(mu), A), U)

    err1 = (AU_mu - AU.as_range_array(mu)).norm() / AU_mu.norm()
    err2 = (AAU_mu - AAU.as_range_array(mu)).norm() / AAU_mu.norm()
    err3 = (AAU_mu - AAU_bis.as_range_array(mu)).norm() / AAU_mu.norm()

    print('Relative error on AU', err1.sum())
    print('Relative error on AAU', err2.sum())
    print('Relative error on AAU_bis', err3.sum())   


