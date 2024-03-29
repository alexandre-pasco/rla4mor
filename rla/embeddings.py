#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:35:17 2023

@author: apasco
"""


import numpy as np
from scipy.sparse import eye
from pymor.tools.frozendict import FrozenDict
from pymor.core.base import abstractmethod
from pymor.operators.constructions import Operator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from rla.srht import fht_oop, srht



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
        keys are 'range_dim', 'epsilon', 'delta' and 'oblivious_dim'. 
    _random_matrix : np.ndarray
        The l2 -> l2 embedding matrix
    _matrix : np.ndarray
        The U -> l2 embedding matrix, i.e. sqrt_product @ _random_matrix.
    _seed : int
        The seed of the random number generator.
        
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
    
    def set_seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, high=2**32-1)
        self._seed = seed
        self.update()
        
    def update(self):
        if not(self._random_matrix is None):
            self._random_matrix = self._compute_random_matrix()
    
        if not(self._matrix is None):
            self._matrix = self._compute_matrix()
    
    def as_range_array(self):
        result = self.range.from_numpy(self.get_matrix().T)
        return result
    
    
    def as_source_array(self):
        result = self.source.from_numpy(self.get_matrix())
        return result



class SrhtEmbedding(RandomEmbedding):

    def __init__(self, source=None, sqrt_product=None, options=None,
                 range_id=None, _seed=None):
    
        assert not(source is None) or not(sqrt_product is None)
        if options is None: options = dict()
        if _seed is None:
            _seed = np.random.randint(0, high=2**32-1)
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim(), id=range_id)
        self._matrix = None
        self._random_matrix = None
        self.linear = True

    def update(self):
        pass

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
        qu = self.sqrt_product.apply(U).to_numpy()
        squ = srht(qu, self.range.dim, self._seed)
        result = self.range.from_numpy(squ)
        return result
    
    
    def apply_adjoint(self, U, mu=None):
        assert U in self.range
        mat = NumpyMatrixOperator(self.get_matrix().T, source_id=self.range.id, range_id=self.source.id)
        return mat.apply(U)
    
    
    
    def _compute_matrix(self):
        Q = self.sqrt_product
        rmat = self.get_random_matrix()
        mat = Q.apply_adjoint(Q.range.from_numpy(rmat)).to_numpy()
        return mat
    
    
    def _compute_random_matrix(self):
        self.logger.warning_once("Computing explicit SRHT matrix")
        mat = self._get_random_rows(np.arange(self.range.dim))
        return mat
    
    
    def _get_random_rows(self, indices):
        n = self.sqrt_product.range.dim
        k = self.range.dim
        d = int(np.ceil(np.log2(n)))
        seed = self._seed
        
        rademacher = np.random.RandomState(seed).choice([-1, 1], (n), replace=True)
        sampling = np.random.RandomState(seed).choice(range(2**d), k, replace=True)

        Pt = np.zeros((len(indices), 2**d))
        for i, ind in enumerate(indices):
            Pt[i, sampling[ind]] = 1
        Pt  = fht_oop(Pt)
        DHP = np.sqrt(n/k) * Pt[:,:n] * rademacher
        return DHP




class GaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source=None, sqrt_product=None, options=None,
                 range_id=None, _seed=None):
    
        assert not(source is None) or not(sqrt_product is None)
        if options is None: options = dict()
        if _seed is None:
            _seed = np.random.randint(0, high=2**32-1)
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim(), id=range_id)
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
    
    
    
    def _compute_matrix(self):
        gauss = self._random_matrix
        Q = self.sqrt_product
        mat = Q.apply_adjoint(Q.range.from_numpy(gauss.conj())).to_numpy().conj()
        return mat
        
    
    def _compute_random_matrix(self):
        k = self.range.dim
        n = self.sqrt_product.range.dim
        seed = self._seed
        gauss = np.random.RandomState(seed).normal(size=(k,n), loc=0, scale=1/np.sqrt(k))
        return gauss
   

        
class IdentityEmbedding(RandomEmbedding):

    def __init__(self, source=None, sqrt_product=None, options=None,
                 range_id=None, _seed=None):
        
        assert not(source is None) or not(sqrt_product is None)
        if options is None: options = dict()
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim(), id=range_id)
        self._matrix = None
        self._random_matrix = self._compute_random_matrix()
        self.linear = True

    def compute_dim(self):
        return self.source.dim
    
    def apply(self, U, mu=None):
        return self.sqrt_product.apply(U)
    
    def apply_adjoint(self, U, mu=None):
        
        return self.source.from_numpy(self.sqrt_product.apply_adjoint(U).to_numpy())
    
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


class EmbeddingVectorized(RandomEmbedding):
    """
    Sketch a whole vector array by vectorizing it then applying a gaussian
    sketch.
    """
    def __init__(self, source, n_vectors, embedding, options=None, range_id=None,
                 _seed=None):
        
        if options is None: options = dict()
        if _seed is None:
            _seed = np.random.randint(0, high=2**32-1)
        self.__auto_init(locals())
        self.options['range_dim'] = embedding.range.dim
        self.options = FrozenDict(options)
        self.range = embedding.range
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
        if range_dim is None:
            a = 1
            if opt.get('dtype') == complex:
                a = 2
            range_dim = 7.87 * (1/eps**2) * (a * 6.9 * d + np.log(1/delta))
            range_dim = int(np.ceil(range_dim))
        return range_dim
    
    def apply(self, U, mu=None):
        assert U in self.source
        assert len(U) == self.n_vectors
        x = U.to_numpy().T.flatten()
        x = self.embedding.source.from_numpy(x)
        result = self.embedding.apply(x)
        return result
    
    def apply_adjoint(self, U, mu=None):
        pass
    
    
    def _compute_matrix(self):
        return self.embedding._compute_matrix()
    
    
    def _compute_random_matrix(self):
        return self.embedding._compute_random_matrix()
    


class BlockGaussianEmbedding(RandomEmbedding):
    
    def __init__(self, source=None, sqrt_product=None, options=None,
                 range_id=None, _seed=None):
    
        assert not(source is None) or not(sqrt_product is None)
        assert "max_block_size" in options.keys()
        if options is None: options = dict()
        if _seed is None:
            _seed = np.random.randint(0, high=2**32-1)
        self.__auto_init(locals())
        self.options = FrozenDict(options)
        if sqrt_product is None:
            self.sqrt_product = IdentityOperator(source)
        self.source = self.sqrt_product.source
        self.range = NumpyVectorSpace(self.compute_dim(), id=range_id)
        self._matrix = None
        self._random_matrix = None
        self.linear = True
        
        # block sizes
        max_block_size = options.get("max_block_size")
        m = self.range.dim // max_block_size
        r = self.range.dim % max_block_size
        block_sizes = [max_block_size for i in range(m)]
        if r > 0: block_sizes.append(r)
        self.block_sizes = block_sizes
        self.n_blocks = len(block_sizes)
        
        # block_seeds
        block_seeds = np.random.RandomState(self._seed).randint(0,2**32-1,size=len(block_sizes))
        while len(np.unique(block_seeds)) < len(block_seeds):
            self._seed += 1
            block_seeds = np.random.RandomState(self._seed).randint(0,2**32-1,size=len(block_sizes))
        self.block_seeds = block_seeds
        
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
        Q = self.sqrt_product
        V = Q.apply(U)
        lst = []
        for i in range(len(self.block_sizes)):
            gauss = self._get_random_block(i)
            op = NumpyMatrixOperator(gauss, source_id=Q.range.id, range_id=self.range.id)
            lst.append(op.apply(V).to_numpy())
        result = np.hstack(lst)
        return self.range.from_numpy(result)

    
    def _compute_matrix(self):
        mat = self._compute_random_matrix()
        Q = self.sqrt_product
        result = Q.apply_adjoint(Q.range.from_numpy(mat.conj())).to_numpy().conj()
        return result
        
    
    def _compute_random_matrix(self):
        lst = []
        for i in range(self.n_blocks):
            block = self._get_random_block(i)
            lst.append(block)
        mat = np.vstack(lst)
        return mat
   
    def _get_random_block(self, ind):
        
        k = self.range.dim
        n = self.sqrt_product.range.dim
        b = self.block_sizes[ind]
        seed = self.block_seeds[ind]
        result = np.random.RandomState(seed).normal(
            size=(b,n), loc=0, scale=1/np.sqrt(k)
            )
        return result

    def get_block(self, ind):
        gauss = self._get_random_block(ind)
        Q = self.sqrt_product
        block = Q.apply_adjoint(Q.range.from_numpy(gauss.conj())).to_numpy().conj()
        return block



    
    
    
    
    
    