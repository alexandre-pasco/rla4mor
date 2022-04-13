#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:06:59 2022

@author: pasco
"""

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv