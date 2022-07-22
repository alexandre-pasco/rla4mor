#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:34:53 2022

@author: pasco
"""


import numpy as np
from scipy.sparse import csc_matrix
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, ConcatenationOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.base import Mu

from affine_operations import *
from other_operators import *
from embeddings import *


if __name__ == '__main__':
    
    N = 10
    r = 3
    parameters = {'mu0': 1, 'mu1': 1}
    alpha0 = ExpressionParameterFunctional('1.0 + mu0[0]',parameters)
    alpha1 = ExpressionParameterFunctional('1.0 + mu0[0] + mu1[0]',parameters)
    mat0= csc_matrix(4 * np.eye(N))
    mat1 = csc_matrix(np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))
    A0 = NumpyMatrixOperator(mat0, source_id='id', range_id='id')
    A1 = NumpyMatrixOperator(mat1, source_id='id', range_id='id')
    operators = [A0, A1]
    coefficients = [alpha0, alpha1]
    A = LincombOperator(operators, coefficients)
    mu = Mu({'mu0': [1.], 'mu1': [2.]})
    A_mu = A.assemble(mu)
    U = A.source.from_numpy(np.random.normal(size=(r, N)))


    print("===Testing functions from affine_operations.py===")
    AU_mu = A_mu.apply(U)
    AAU_mu = A_mu.apply(A_mu.apply(U))

    AU = apply_affine(A, U)
    AA = lincomb_compose_lincomb(A, A)
    AAU = apply_affine(AA, U)
    AAU_bis = apply_affine(op_compose_lincomb(A_mu, A), U)

    err1 = (AU_mu - AU.as_range_array(mu)).norm() / AU_mu.norm()
    err2 = (AAU_mu - AAU.as_range_array(mu)).norm() / AAU_mu.norm()
    err3 = (AAU_mu - AAU_bis.as_range_array(mu)).norm() / AAU_mu.norm()

    print('Relative error on AU', err1.sum())
    print('Relative error on AAU', err2.sum())
    print('Relative error on AAU_bis', err3.sum())
    
    print("===Testing Functions from other_operators.py===")
    Q_mu = CholeskyOperator(A.assemble(mu).matrix, source_id=A.source.id, range_id=A.range.id)
    Ainv_mu = ImplicitLuOperator(A.assemble(mu).matrix, source_id=A.source.id, range_id=A.range.id)
    
    QU_mu = Q_mu.apply_adjoint(U)
    
    err4 = np.linalg.norm(QU_mu.inner(QU_mu) - A_mu.apply2(U, U)) / np.linalg.norm(A_mu.apply2(U, U))
    err5 = (Ainv_mu.apply_inverse(U) - A_mu.apply_inverse(U)).norm() / A_mu.apply_inverse(U).norm()
    
    print('Relative error on UQQ^HU', err4)
    print('Relative error on (LU)^-1 U', err5.sum())
    
    