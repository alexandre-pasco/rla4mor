#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:35:17 2023

@author: apasco
"""


import numpy as np
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator, IdentityOperator, ConcatenationOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from fht import fht



class RandomEmbedding(Operator):
    """
    Class implementing a random embedding.
    
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
    dtype : data-type [to remove ?]
        The data type, float' or complex.
    sqrt_product : Operator
        An operator Q such as Q^H @ Q = R, with R is a positive definite,
        self adjoint operator, enconding the inner product.
    _random_matrix : np.ndarray
        The l2 -> l2 embedding matrix
    _matrix : np.ndarray
        The U -> l2 embedding matrix, i.e. sqrt_product @ _random_matrix.
    _seed : int
        If implemented, the seed for the random operator.
    """
    

    @abstractmethod
    def compute_dim(self):
        """
        Compute the lower bound for the embedding dimension to ensure 
        a priori the oblivious embedding property.

        Returns
        -------
        embedding_dim : int

        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Update the embedding according to self.options['seed'].

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
    
    
    def set_seed(self, seed=None):
        """
        Set the attribute self._seed. This attribute is the random state of 
        the embedding.

        Parameters
        ----------
        seed : int
            The seed used to perform operations with the random embedding.
            If None, the seed used is obtained randomly by
            numpy.random.randint(0,2**32-1).
            
        """
        if seed is None:
            new_seed = np.random.randint(0, high=2**32-1)
        else :
            new_seed = seed
        self.options['seed'] = new_seed
        self.update()
    
    
    def as_range_array(self):
        result = self.range.from_numpy(self.get_matrix().T)
        return result
    
    
    def as_source_array(self):
        result = self.source.from_numpy(self.get_matrix())
        return result



class SrhtEmbedding(RandomEmbedding):

    def __init__(self, source=None, sqrt_product=None, options=None):
    
        assert not(source is None) or not(sqrt_product is None)
        self.__auto_init(locals())
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
    
    
    def srht(self, x):
        """
        Compute the srht of each element of a numpy array ()

        Parameters
        ----------
        x : ndarray 
            Array of shape (N,k), viewed as k vectors of size N.

        Returns
        -------
        PHDx : ndarray
            The srht transform of each element of x

        """
        # Managin the dimension of x
        assert x.ndim <= 2
        if x.ndim == 1: x = x.reshape((-1,1))
            
        n = self.sqrt_product.range.dim
        k = self.range.dim
        d = int(np.ceil(np.log2(n)))
        seed = self.options.get('seed')
        rademacher = np.random.RandomState(seed).choice([-1, 1], (n,1), True)
        sampling = np.random.RandomState(seed).choice(range(2**d), k, True)

        Dx = rademacher*x
        # Adding zeros if the vectors are not of size 2**d
        Dx = np.append(Dx, np.zeros((2**d-n, x.shape[1])), axis=0)
        # Applying the inplace Fast Hadamard Transform
        fht(Dx)
        # sampling and rescaling
        PHDx = np.sqrt(1/k) * Dx[sampling, :]
        return PHDx
        
    def apply(self, U, mu=None):
        assert U in self.source
        qu = self.sqrt_product.apply(U)
        squ = self.srht(qu.to_numpy().T)
        result = self.range.from_numpy(squ.T)
        return result
    
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        mat = NumpyMatrixOperator(self.get_matrix().T, source_id=self.range.id, range_id=self.source.id)
        return mat.apply(U)
    
    
    def update(self):
        pass
    
    
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






