#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:06:59 2022

@author: pasco
"""

import numpy as np
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator, IdentityOperator, ConcatenationOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from scipy.sparse import eye
import ffht


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
    dtype : data-type
        The data type, float' or complex.
    sqrt_product : Operator
        An operator Q such as Q^H @ Q = Ru, with Ru is a positive definite,
        self adjoint operator, for example a stifness matrix.
    _matrix : np.ndarray
        The U -> l2 embedding matrix.
    _random_matrix : np.ndarray
        The l2 -> l2 embedding matrix
    _seed : int
        If implemented, the seed for the random operator.
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
        adjoint of self.sqrt_product to the row of the l2 -> l2 embedding 
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
        self._seed = new_seed
        self.update()
    
    
    def as_range_array(self):
        result = self.range.from_numpy(self.get_matrix().T)
        return result
    
    
    def as_source_array(self):
        result = self.source.from_numpy(self.get_matrix())
        return result
    

class GaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, sqrt_product=None, name='gaussian', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        if not(sqrt_product is None):
            self.source = sqrt_product.source
        else: 
            self.source = NumpyVectorSpace(source_dim, source_id)
            self.sqrt_product = IdentityOperator(self.source)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None
        self._random_matrix = None
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
        gauss = self._random_matrix
        Q = self.sqrt_product
        op = NumpyMatrixOperator(gauss, source_id=Q.range.id, range_id=self.range.id)
        return op.apply(Q.apply(U))
    
    
    def update(self):
        k = self.range.dim
        n = self.source.dim
        gauss = np.random.RandomState(self._seed).normal(size=(k,n), loc=0, scale=1/np.sqrt(k))
        self._random_matrix = gauss
    
    
    def _compute_matrix(self):
        gauss = self._random_matrix
        Q = self.sqrt_product
        mat = Q.apply_adjoint(Q.range.from_numpy(gauss.conj())).to_numpy().conj()
        return mat
        
    
    def _compute_random_matrix(self):
        return self._random_matrix
    


class GaussianEmbeddingRowWise(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, sqrt_product=None, name='gaussian_row_wise', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        self.set_seed(_seed)
        if not(sqrt_product is None):
            self.source = sqrt_product.source
        else: 
            self.source = NumpyVectorSpace(source_dim, source_id)
            self.sqrt_product = IdentityOperator(self.source)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None
        self._random_matrix = None


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
        QU = self.sqrt_product.apply(U).to_numpy().T
        n = self.sqrt_product.range.dim
        k = self.range.dim
        result = np.zeros((k, QU.shape[1]), self.dtype)
        for i in range(k):
            gauss = np.random.RandomState(self._seed + i).normal(size=n, loc=0, scale=1)
            result[i,:] = np.dot(gauss, QU)
        result = result / np.sqrt(k)
        return self.range.from_numpy(result.T)
    
    
    def update(self):
        pass
    

    def _compute_matrix(self):
        Q = self.sqrt_product
        gauss = self._compute_random_matrix()
        mat = Q.apply_adjoint(Q.range.from_numpy(gauss.conj())).to_numpy().conj()
        return mat
    
    
    def _compute_random_matrix(self):
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
                 solver_option=None, sqrt_product=None, name='rademacher', _seed=None):
        
        self.__auto_init(locals())
        self.linear = True
        self.set_seed(_seed)
        if not(sqrt_product is None):
            self.source = sqrt_product.source
        else: 
            self.source = NumpyVectorSpace(source_dim, source_id)
            self.sqrt_product = IdentityOperator(self.source)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None
        self._random_matrix = None


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
        QU = self.sqrt_product.apply(U).to_numpy().T
        n = self.sqrt_product.range.dim
        k = self.range.dim
        result = np.zeros((k, QU.shape[1]), dtype=self.dtype)
        for i in range(k):
            rademacher = np.random.RandomState(self._seed + i).choice([-1, 1], n, replace=True)
            result[i,:] = np.dot(rademacher, QU)
        result = result / np.sqrt(k)
        return self.range.from_numpy(result.T)
    
    
    def update(self):
        pass


    def _compute_matrix(self):
        Q = self.sqrt_product
        rad = self._compute_random_matrix()
        mat = Q.apply_adjoint(Q.range.from_numpy(rad.conj())).to_numpy().conj()
        return mat


    def _compute_random_matrix(self):
        n = self.source.dim
        k = self.range.dim
        mat = np.zeros((k,n))
        for i in range(k):
            rademacher = np.random.RandomState(self._seed + i).choice([-1, 1], n, replace=True)
            mat[i,:] = rademacher / np.sqrt(k)
        return mat
    
    
class SrhtEmbedding(RandomEmbedding):
    
    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, dtype=float, source_id=None, range_id=None, 
                 solver_option=None, sqrt_product=None, name='srht', _seed=None):
        
        assert dtype in (float, complex)
        self.linear = True
        self.__auto_init(locals())
        self.set_seed(_seed)
        if not(sqrt_product is None):
            self.source = sqrt_product.source
        else: 
            self.source = NumpyVectorSpace(source_dim, source_id)
            self.sqrt_product = IdentityOperator(self.source)
        if (epsilon is None) or (delta is None) or (oblivious_dim is None):
            embedding_dim = range_dim
        else :
            embedding_dim = self.compute_dim()
        self.range = NumpyVectorSpace(embedding_dim, range_id)
        self._matrix = None
        self._random_matrix = None


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
            
        n = self.sqrt_product.range.dim
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
        QU = self.sqrt_product.apply(U)
        return self.range.from_numpy(self.srht(QU.to_numpy()))
    
    
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
        rademacher = np.random.RandomState(self._seed).choice([-1, 1], n, replace=True)
        sampling = np.random.RandomState(self._seed).choice(range(2**d), k, replace=True)
        if hasattr(ffht, 'fht_'): ht = ffht.fht_
        else: ht = ffht.fht

        P = np.zeros((len(indices), 2**d))
        for i in range(len(indices)):
            P[i, sampling[i]] = 1

        for Pi in P:
            # Applying the inplace Fast Hadamard Transform
            ht(Pi)

        DHP = np.sqrt(1/k) * P[:,:n] * rademacher
        return DHP
        
    
    
    # def _get_hadamard_row(self, n):
    #     """
    #     Compute the n-th row of the truncated Hadamard matrix.

    #     Parameters
    #     ----------
    #     n : int
    #         The row to compute

    #     Returns
    #     -------
    #     row : np.array
    #         array of size (self.source.dim,)

    #     """
    #     row = np.zeros(self.source.dim)
    #     for i in range(self.source.dim):
    #         b = binary_inner(n, i)
    #         if b%2 == 0:
    #             row[i] = 1
    #         else : 
    #             row[i] = -1
    #     return row
        
        
class IdentityEmbedding(RandomEmbedding):

    def __init__(self, source_dim=1, range_dim=1, epsilon=None, delta=None, 
                 oblivious_dim=None, seed=None, dtype=float, source_id=None, 
                 range_id=None, solver_option=None, sqrt_product=None, name='identity', _seed=None):
        
        self.__auto_init(locals())
        if not(sqrt_product is None):
            self.source = sqrt_product.source
        else: 
            self.source = NumpyVectorSpace(source_dim, source_id)
            self.sqrt_product = IdentityOperator(self.source)
        self.range = NumpyVectorSpace(source_dim, range_id)
        self.epsilon = 0
        self.delta = 0
        self.oblivious_dim = source_dim
        self.name = name
        self.linear = True
        self._matrix = None
        self._random_matrix = None

    def compute_dim(self):
        return self.source.dim
    
    
    def apply(self, U, mu=None):
        return self.range.from_numpy(self.sqrt_product.apply(U).to_numpy())
    
    
    def update(self):
        pass
    
    
    def _compute_matrix(self):
        if hasattr(self.sqrt_product, 'get_matrix'):
            mat = self.sqrt_product.get_matrix()
        else:
            vec = self.source.from_numpy(np.eye(self.source.dim))
            mat = self.apply(vec).to_numpy().T
        return mat
    
    
    def _compute_random_matrix(self):
        return eye(self.source.dim)

    
    
def generate_embedding(embedding_type, source_dim=1, range_dim=1, epsilon=None, 
                       delta=None, oblivious_dim=None, dtype=float, 
                       source_id=None, range_id=None, solver_option=None, 
                       sqrt_product=None, name='srht', _seed=None):
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
    elif embedding_type == 'identity':
        embedding = IdentityEmbedding(**kwargs)
    else:
        print("Embedding type note implemented, gaussian used.")
        embedding = GaussianEmbedding(**kwargs)
    return embedding



# def binary_inner(a,b):
#     """
#     Return the inner product of the binary vectors associated with a and b.

#     Parameters
#     ----------
#     a : int
        
#     b : int
    

#     Returns
#     -------
#     res : int
        

#     """
#     a_bin, b_bin = bin(a), bin(b)
#     i = 1
#     ai, bi = a_bin[-i], b_bin[-i]
#     res = 0
#     while ai != 'b' and bi != 'b':
#         res = res + int(ai) * int(bi)
#         i += 1
#         ai, bi = a_bin[-i], b_bin[-i]
#     return res


def sketch_operator(operator, embeddings):
    """
    Sketch an operator into a l2 vector using three embeddings : sigma, omega 
    and gamma, inputed as a tuple. The sketch is computed by applying the 
    operator to the columns of sigma, then applying omega to the resulting columns.
    The resullting array in then vectorized so that gamma can be applied on it.
    
    The norm of the result approximates the Froboenius norm of the operator.
    
    
    Parameters
    ----------
    operator : Operator
        The linear operator to sketch.
    embeddings : tuple of RandomEmbedding or Operator
        The embeddings to use. 

    Returns
    -------
    result : NumpyVectorArray
        The vector representing the sketched operator.

    """
    sigma_vectors = embeddings[0].as_source_array().conj()
    omega = embeddings[1]
    gamma = embeddings[2]
    
    sketched_right = operator.apply(sigma_vectors)
    sketched = omega.apply(sketched_right).to_numpy().T
    vectorized = np.concatenate( [sketched[:,j] for j in range(sketched.shape[1])] )
    result = gamma.apply(gamma.source.from_numpy(vectorized))
    return result



if __name__ == '__main__':
    
    embedding = SrhtEmbedding(source_dim=100, range_dim=20)
    # embedding = GaussianEmbedding(source_dim=100, range_dim=20)
    # embedding = RademacherEmbedding(source_dim=100, range_dim=20)
    
    u = embedding.source.from_numpy(np.random.normal(size=embedding.source.dim))
    mat = embedding.get_matrix()
    v = embedding.apply(u).to_numpy().T
    w = np.dot(mat, u.to_numpy().T)
    err = v-w
    print("get_matrix error", np.linalg.norm(err) / np.linalg.norm(v))
    
    
    sigma = generate_embedding("srht", source_dim=100, range_dim=50)
    omega = generate_embedding("srht", source_dim=100, range_dim=50)
    gamma = generate_embedding("gaussian", source_dim=sigma.range.dim*omega.range.dim, range_dim=50)

    op1 = NumpyMatrixOperator(np.random.normal(size=(100,100)))
    op2 = NumpyMatrixOperator(np.random.normal(size=(100,100)))
    
    s1 = sketch_operator(op1, (sigma, omega, gamma))
    s2 = sketch_operator(op2, (sigma, omega, gamma))
    err_inner = (np.sum(op1.matrix*op2.matrix) - s1.inner(s2)[0,0]) / (np.linalg.norm(op1.matrix) * np.linalg.norm(op1.matrix))
    err_1 = 1 - s1.norm()[0]/np.linalg.norm(op1.matrix)
    err_2 = 1 - s2.norm()[0]/np.linalg.norm(op2.matrix)
    
    print("inner product error", err_inner)
    print("norm error 1", err_1)
    print("norm error 2", err_2)
    
    
