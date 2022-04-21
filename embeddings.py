#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:06:59 2022

@author: pasco
"""

import numpy as np
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator
from pymor.operators.numpy import NumpyMatrixOperator
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
    dtype : data-type
        The data type, float' or complex.
    _matrix : np.ndarray
        The full matrix.
    _seed : int
        If implemented, the seed for the random operator.
    _update : bool
        True if the embedding needs to be updated due to a seed change.
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
    
    @abstractmethod
    def update(self):
        """
        Update the embedding according to the attribute _seed.

        Returns
        -------
        None.

        """
        pass
    
    
    @abstractmethod
    def get_matrix(self):
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
        self.update()
    
    
class GaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, name='gaussian', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        self.source = NumpyVectorSpace(source_dim, source_id)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self.set_seed(_seed)


    def compute_dim(self):
        if self.dtype is float: a = 1
        elif self.dtype is complex: a = 2
        else: 
            print('Wrong data-type. Considered as float.')
            a = 1
        bound = 7.87 * (1/self.epsilon**2) * (a * 6.9 * self.oblivious_dim + np.log(1/self.delta))
        bound = int(np.ceil(bound))
        return bound
    
    
    def apply(self, U, mu=None):
        gauss = self._matrix
        op = NumpyMatrixOperator(gauss, source_id=self.source.id, range_id=self.range.id)
        return op.apply(U)
    
    
    def update(self):
        k = self.range.dim
        n = self.source.dim
        gauss = np.random.RandomState(self._seed).normal(size=(k,n), loc=0, scale=1/np.sqrt(k))
        self._matrix = gauss
    
    
    def get_matrix(self):
        return self._matrix


class GaussianEmbeddingRowWise(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, name='gaussian_row_wise', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        self.set_seed(_seed)
        self.source = NumpyVectorSpace(source_dim, source_id)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None


    def compute_dim(self):
        if self.dtype is float: a = 1
        elif self.dtype is complex: a = 2
        else: 
            print('Wrong data-type. Considered as float.')
            a = 1
        bound = 7.87 * (1/self.epsilon**2) * (a * 6.9 * self.oblivious_dim + np.log(1/self.delta))
        bound = int(np.ceil(bound))
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
    
    
    def update(self):
        pass
    
    
    def get_matrix(self):
        if self._matrix is None:
            self._matrix = self._compute_matrix()
        mat = self._matrix
        return mat
    
    
    def _compute_matrix(self):
        n = self.source.dim
        k = self.range.dim
        mat = np.zeros((k,n))
        for i in range(k):
            gauss = np.random.RandomState(self._seed + i).normal(size=n, loc=0, scale=1)
            mat[i,:] = gauss / np.sqrt(k)
        self._matrix = mat
        return mat


class RademacherEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, name='rademacher', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        self.set_seed(_seed)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.source = NumpyVectorSpace(source_dim, source_id)
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None


    def compute_dim(self):
        if self.dtype is float: a = 1
        elif self.dtype is complex: a = 2
        else: 
            print('Wrong data-type. Considered as float.')
            a = 1
        bound = 7.87 * (1/self.epsilon**2) * (a * 6.9 * self.oblivious_dim + np.log(1/self.delta))
        bound = int(np.ceil(bound))
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
    
    
    def update(self):
        pass


    def get_matrix(self):
        if self._matrix is None:
            self._matrix = self._compute_matrix()
        mat = self._matrix
        return mat

    
    def _compute_matrix(self):
        n = self.source.dim
        k = self.range.dim
        mat = np.zeros((k,n))
        for i in range(k):
            rademacher = np.random.RandomState(self._seed + i).choice([-1, 1], n, replace=True)
            mat[i,:] = rademacher / np.sqrt(k)
        self._matrix = mat
        return mat
    
    
class SrhtEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, seed=None, dtype=float, source_id=None, 
                 range_id=None, solver_option=None, name='srht'):
        
        assert dtype in (float, complex)
        self.linear = True
        self.__auto_init(locals())
        self.set_seed(seed)
        self.source = NumpyVectorSpace(source_dim, source_id)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None


    def compute_dim(self):
        if self.dtype is float: a = 1
        else: a = 2
        eps, delta, d, n = self.epsilon, self.delta, self.oblivious_dim, self.source.dim
        bound = 2 / (eps**2 - eps**3/3)
        bound = bound * (np.sqrt(a*d) + np.sqrt(8*np.log(6*a*n/delta)))**2 
        bound = bound * np.log(3*a*d / delta)
        bound = int(np.ceil(bound))
        return bound
    
    
    def srht_real(self, x):
        """
        Compute the srht of each element of a numpy array ()

        Parameters
        ----------
        x : ndarray 
            Array of shape (k,N), viewed as k vectors of size N.

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
    
    
    def update(self):
        pass
    
    
    def get_cols(self, indices):
        x = np.zeros((len(indices), self.source.dim))
        for i in indices:
            x[i,indices[i]] = 1
        result = self.srht_real(x)
        return result.T
    
    
    def get_matrix(self):
        if self._matrix is None:
            self._matrix = self._compute_matrix()
        mat = self._matrix
        return mat
        
    
    def _compute_matrix(self):
        n = self.source.dim
        k = self.range.dim
        d = int(np.ceil(np.log2(n)))
        rademacher = np.random.RandomState(self._seed).choice([-1, 1], n, replace=True)
        sampling = np.random.RandomState(self._seed).choice(range(2**d), k, replace=True)
        mat = np.zeros((k,n))
        
        for i in range(k):
            h_row = self._get_hadamard_row(sampling[i])
            row = h_row * rademacher / np.sqrt(k)
            mat[i] = row

        return mat
    
    
    def _get_hadamard_row(self, n):
        """
        Compute the n-th row of the truncated Hadamard matrix.

        Parameters
        ----------
        n : int
            The row to compute

        Returns
        -------
        row : np.array
            array of size (self.source.dim,)

        """
        row = np.zeros(self.source.dim)
        for i in range(self.source.dim):
            b = binary_inner(n, i)
            if b%2 == 0:
                row[i] = 1
            else : 
                row[i] = -1
        return row
        
        
    
def generate_embedding(embedding_type, source_dim=1, range_dim=1, epsilon=None, 
                       delta=None, oblivious_dim=None, seed=None, dtype=float, 
                       source_id=None, range_id=None, solver_option=None):
    kwargs = locals()
    _ = kwargs.pop('embedding_type')
    if embedding_type == 'srht':
        embedding = SrhtEmbedding(**kwargs)
    elif embedding_type == 'rademacher':
        embedding = RademacherEmbedding(**kwargs)
    elif embedding_type == 'gaussian_row_wise':
        embedding = GaussianEmbeddingRowWise(**kwargs)
    elif embedding_type == 'gaussian':
        embedding = GaussianEmbedding(**kwargs)
    else:
        print("Embedding type note implemented, gaussian used.")
        embedding = GaussianEmbedding(**kwargs)
    return embedding



def binary_inner(a,b):
    """
    Return the inner product of the binary vectors associated with a and b.

    Parameters
    ----------
    a : int
        
    b : int
    

    Returns
    -------
    res : int
        

    """
    a_bin, b_bin = bin(a), bin(b)
    i = 1
    ai, bi = a_bin[-i], b_bin[-i]
    res = 0
    while ai != 'b' and bi != 'b':
        res = res + int(ai) * int(bi)
        i += 1
        ai, bi = a_bin[-i], b_bin[-i]
    return res

if __name__ == '__main__':
    
    embedding = SrhtEmbedding(source_dim=100, range_dim = 20)
    u = embedding.source.from_numpy(np.random.normal(size=embedding.source.dim))
    mat = embedding.get_matrix()
    v = embedding.apply(u).to_numpy().T
    w = np.dot(mat, u.to_numpy().T)
    err = v-w
    print(np.linalg.norm(err) / np.linalg.norm(v))
