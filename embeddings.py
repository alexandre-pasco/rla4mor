#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:06:59 2022

@author: pasco
"""

import numpy as np
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

import ffht


class RandomEmbedding(Operator):
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
    _seed : int
        If implemented, the seed for the random operator.
    dtype : data-type
        The data type, float' or complex.
    """
    

    @abstractmethod
    def compute_dim(self):
        """
        Compute the lower bound for the embedding dimension to ensure 
        a priori an oblivious embedding.

        Returns
        -------
        embedding_dim : int

        """
        pass
    
    
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
        self._seed = new_seed
    
    
class GaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, name='gaussian', _seed=None):
        
        self.__auto_init(locals())
        self.set_seed(_seed)
        self.source = NumpyVectorSpace(source_dim, range_id)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)


    def compute_dim(self):
        if self.dtype is float: a = 1
        elif self.dtype is complex: a = 2
        else: 
            print('Wrong data-type. Considered as float.')
            a = 1
        bound = 7.87 * self.epsilon * (a * 6.9 * self.oblivious_dim + np.log(1/self.delta))
        return bound
    
    
    def apply(self, U, mu=None):
        U_np = U.to_numpy().T
        n = self.source.dim
        k = self.range.dim
        result = np.zeros((k, U_np.shape[1]), self.dtype)
        for i in range(k):
            gauss = np.random.RandomState(self._seed + i).normal(size=n, loc=0, scale=1)
            result[i,:] = np.dot(gauss, U_np)
        result = result / np.sqrt(k)
        return self.range.from_numpy(result.T)
    
    
    def matrix(self):
        n = self.source.dim
        k = self.range.dim
        mat = np.zeros((k,n))
        for i in range(k):
            gauss = np.random.RandomState(self._seed + i).normal(size=n, loc=0, scale=1)
            mat[i,:] = gauss / np.sqrt(k)
        return mat


class RademacherEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, name='rademacher', _seed=None):
        
        self.__auto_init(locals())
        self.set_seed(_seed)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.source = NumpyVectorSpace(source_dim, range_id)
        self.range = NumpyVectorSpace(embedding_dim, range_id)


    def compute_dim(self):
        if self.dtype is float: a = 1
        elif self.dtype is complex: a = 2
        else: 
            print('Wrong data-type. Considered as float.')
            a = 1
        bound = 7.87 * self.epsilon * (a * 6.9 * self.oblivious_dim + np.log(1/self.delta))
        return bound
    
    
    def apply(self, U, mu=None):
        U_np = U.to_numpy().T
        n = self.source.dim
        k = self.range.dim
        result = np.zeros((k, U_np.shape[1]), dtype=self.dtype)
        for i in range(k):
            rademacher = np.random.RandomState(self._seed + i).choice([-1, 1], n, replace=True)
            result[i,:] = np.dot(rademacher, U_np)
        result = result / np.sqrt(k)
        return self.range.from_numpy(result.T)
    
    
    def matrix(self):
        n = self.source.dim
        k = self.range.dim
        mat = np.zeros((k,n))
        for i in range(k):
            rademacher = np.random.RandomState(self._seed + i).choice([-1, 1], n, replace=True)
            mat[i,:] = rademacher / np.sqrt(k)
        return mat
    
    
class SrhtEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, seed=None, dtype=float, source_id=None, 
                 range_id=None, solver_option=None, name='srht'):
        
        assert dtype in (float, complex)
        self.__auto_init(locals())
        self.set_seed(seed)
        self.source = NumpyVectorSpace(source_dim, range_id)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)


    def compute_dim(self):
        if self.dtype is float: a = 1
        else: a = 2
        eps, delta, d, n = self.epsilon, self.delta, self.oblivious_dim, self.source.dim
        bound = 2 / (eps**2 - eps**3/3)
        bound = bound * (np.sqrt(a*d) + np.sqrt(8*np.log(6*a*n/delta)))**2 
        bound = bound * np.log(3*a*d / delta)
        return bound
    
    
    def srht_real(self, x):
        """
        Compute the srht of each element of a numpy array ()

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        PHDx : ndarray
            The srht transform of each element of x

        """
        # Managin the dimension of x
        assert x.ndim <= 2
        if x.ndim == 1: x = x.reshape((1, -1))
            
        n = self.source.dim
        k = self.range.dim
        d = int(np.ceil(np.log2(n)))
        rademacher = np.random.RandomState(self._seed).choice([-1, 1], n, replace=True)
        sampling = np.random.RandomState(self._seed).choice(range(2**d), k, replace=True)
        
        if hasattr(ffht, 'fht_'): ht = ffht.fht_
        else: ht = ffht.fht
        
        d = int(np.ceil(np.log2(n)))
        Dx = rademacher*x
        # Adding zeros if the vectors are not of size 2**d
        Dx = np.append(Dx, np.zeros((x.shape[0], 2**d-n)), axis=1)
        for Dx_i in Dx:
            # Applying the inplace Fast Hadamard Transform
            ht(Dx_i)
        # sampling and rescaling
        PHDx = np.sqrt(1/k) * Dx[:, sampling]
        return PHDx
    
    
    def srht(self, x):
        if self.dtype is float:
            result = self.srht_real(x)
        else:
            real = self.srht_real(np.real(x))
            imag = self.srht_real(np.imag(x))
            result = real + 1.j*imag
        return result
    
    
    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.from_numpy(self.srht(U.to_numpy()))