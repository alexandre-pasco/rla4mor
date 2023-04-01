#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:35:17 2023

@author: apasco
"""


import numpy as np
from pymor.tools.frozendict import FrozenDict
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from srht import fht, srht



class RandomEmbedding(Operator):
    """
    Class implementing a random embedding.
    
    Attibutes
    ---------
    sqrt_product : Operator
        An operator Q such as Q^H @ Q = R, with R is a positive definite,
        self adjoint operator, enconding the inner product.
    options : FrozenDict
        Immutable dictionary containing options for the embedding. The possible 
        keys are 'range_dim', 'epsilon', 'delta', 'oblivious_dim' and 'seed'. 
    _random_matrix : np.ndarray
        The l2 -> l2 embedding matrix
    _matrix : np.ndarray
        The U -> l2 embedding matrix, i.e. sqrt_product @ _random_matrix.
    """
    

    @abstractmethod
    def compute_dim(self):
        """
        Compute the lower bound for the embedding dimension to ensure 
        a priori the oblivious embedding property.

        Returns
        -------
        range_dim : int

        """
        pass

    
    
    @abstractmethod
    def _compute_matrix(self):
        pass
    
    
    @abstractmethod
    def _compute_random_matrix(self):
        pass
    
    
    def get_matrix(self):
        """
        Returns the U -> l2 embedding matrix. It is obtained by applying the 
        adjoint of self.sqrt_product to the rows of the l2 -> l2 embedding 
        matrix.

        Returns
        -------
        numpy.ndarray
            array of size (k, n), where k is the embedding dimension, and n the 
            dimension of the vectors to embed.

        """
        if self._matrix is None:
            self._matrix = self._compute_matrix()
        return self._matrix
    

    def get_random_matrix(self):
        """
        Returns the l2 -> l2 embedding matrix.

        Returns
        -------
        numpy.ndarray
            array of size (k, n), where k is the embedding dimension, and n the 
            dimension of the vectors to embed.

        """
        if self._matrix is None:
            self._matrix = self._compute_random_matrix()
        return self._matrix
    
       
    
    def as_range_array(self):
        result = self.range.from_numpy(self.get_matrix().T)
        return result
    
    
    def as_source_array(self):
        result = self.source.from_numpy(self.get_matrix())
        return result



class SrhtEmbedding(RandomEmbedding):

    def __init__(self, source=None, sqrt_product=None, options=None):
    
        assert not(source is None) or not(sqrt_product is None)
        if options.get('seed') is None:
            options['seed'] = np.random.randint(0, high=2**32-1)
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim())
        self._matrix = None
        self._random_matrix = None
        self.linear = True

    def compute_dim(self):
        opt = self.options
        range_dim = opt.get('range_dim')
        eps = opt.get('epsilon')
        delta = opt.get('delta')
        d = opt.get('oblivious_dim')
        assert range_dim or all([eps, delta, d])
        n = self.source.dim
        if range_dim is None:
            a = 1
            if opt.get('dtype') == complex:
                a = 2
            range_dim = 2 / (eps**2 - eps**3/3)
            range_dim = range_dim * (np.sqrt(a*d) + np.sqrt(8*np.log(6*a*n/delta)))**2 
            range_dim = range_dim * np.log(3*a*d / delta)
            range_dim = int(np.ceil(range_dim))
        return range_dim

        
    def apply(self, U, mu=None):
        assert U in self.source
        qu = self.sqrt_product.apply(U)
        squ = srht(qu.to_numpy().T, self.range.dim, self.options.get('seed'))
        result = self.range.from_numpy(squ.T)
        return result
    
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        mat = NumpyMatrixOperator(self.get_matrix().T, source_id=self.range.id, range_id=self.source.id)
        return mat.apply(U)
    
    
    
    def _compute_matrix(self):
        Q = self.sqrt_product
        s = self.get_random_matrix()
        mat = Q.apply_adjoint(Q.range.from_numpy(s.conj())).to_numpy().conj()
        return mat
    
    
    def _compute_random_matrix(self):
        mat = self._get_rows([i for i in range(self.range.dim)])
        return mat
    
    
    def _get_rows(self, indices):
        n = self.sqrt_product.range.dim
        k = self.range.dim
        d = int(np.ceil(np.log2(n)))
        seed = self.options.get('seed')
        
        
        rademacher = np.random.RandomState(seed).choice([-1, 1], n, replace=True)
        sampling = np.random.RandomState(seed).choice(range(2**d), k, replace=True)

        P = np.zeros((len(indices), 2**d))
        for i in range(len(indices)):
            P[sampling[i],i] = 1
        fht(P)
        DHP = np.sqrt(1/k) * P[:n,:] * rademacher
        return DHP




class GaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source=None, sqrt_product=None, options=None):
    
        assert not(source is None) or not(sqrt_product is None)
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim())
        self._matrix = None
        self._random_matrix = self._compute_random_matrix()
        self.linear = True
        

    def compute_dim(self):
        opt = self.options
        range_dim = opt.get('range_dim')
        eps = opt.get('epsilon')
        delta = opt.get('delta')
        d = opt.get('oblivious_dim')
        assert range_dim or all([eps, delta, d])
        if range_dim is None:
            a = 1
            if opt.get('dtype') == complex:
                a = 2
            range_dim = 7.87 * (1/eps**2) * (a * 6.9 * d + np.log(1/delta))
            range_dim = int(np.ceil(range_dim))
        return range_dim
    
    
    def apply(self, U, mu=None):
        gauss = self._random_matrix
        Q = self.sqrt_product
        op = NumpyMatrixOperator(gauss, source_id=Q.range.id, range_id=self.range.id)
        return op.apply(Q.apply(U))
    
    
    def update(self):
        self._random_matrix = self._compute_random_matrix()
    
    
    def _compute_matrix(self):
        gauss = self._random_matrix
        Q = self.sqrt_product
        mat = Q.apply_adjoint(Q.range.from_numpy(gauss.conj())).to_numpy().conj()
        return mat
        
    
    def _compute_random_matrix(self):
        k = self.range.dim
        n = self.sqrt_product.range.dim
        seed = self.options.get('seed')
        gauss = np.random.RandomState(seed).normal(size=(k,n), loc=0, scale=1/np.sqrt(k))
        return gauss
   


