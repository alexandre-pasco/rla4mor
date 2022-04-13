#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:06:59 2022

@author: pasco
"""

import numpy as np
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import Operator, LincombOperator, IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from scipy.sparse import csc_matrix

from pymor.vectorarrays.numpy import NumpyVectorSpace

class RandomEmbeddingOperator(Operator):
    """
    
    Random Embedding
    
    Attibutes
    ---------
    epsilon : float
        The relative error of the approximated squared norm of the sketched 
        vectors.
    delta : float
        The probability of failiure.
    oblivious_dim : int
        The dimension for which any subspace is embedded with probability
        delta and relative error epsilon
    seed : int
        If implemented, the seed for the random operator.
    dtype : data-type
        The type of the data
    l2_mapping: Operator
        A mapping from the self.source to the euclidian space. It is an 
        operator Q such as R = Q @ Q^H, where R is the stifness matrix for 
        the inner product in source ($\|u\| = u^H @ R @ u = \|Qu\|_{\ell_2}$).
    
    
    """

