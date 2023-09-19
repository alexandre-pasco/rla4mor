#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:46:11 2023

@author: apasco
"""


import sys
import numpy as np
from time import perf_counter
from pymor.core.base import BasicObject, ImmutableObject, abstractmethod
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, ConcatenationOperator, \
    InverseOperator, AdjointOperator, LincombOperator, ZeroOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.algorithms.simplify import contract, expand
from pymor.parameters.base import Mu
from pymor.parameters.functionals import ParameterFunctional
from scipy.sparse import csc_matrix
from scipy.optimize import lsq_linear
import spams

from sklearn.linear_model import ridge_regression, lars_path, LassoLars
from rla4mor.inverse_problems.lars import lars_weighted_path

# from affine_operations import *
# from other_operators import *
# from embeddings import *
# from sparse_projection import *
# from lars import *




class RecoveryMap(ImmutableObject):
    
    
    def __init__(self, V, W, gramian=None, cross_gramian=None, product=None, log_level=20, 
                 manifold_distance=None):
        
        if product is None:
            product = IdentityOperator(V.space)
        
        if gramian is None:
            gramian = W.gramian(product)
        
        if cross_gramian is None:
            cross_gramian = W.inner(V, product)

        self.__auto_init(locals())
        self.logger.setLevel(log_level)
    
    @abstractmethod
    def project_background(self, indices):
        """
        Create a new RecoveryMap initialized with a restricted background basis.

        Parameters
        ----------
        indices : (k,) ndarray
            The indices to which the background basis `V` is restricted.

        Returns
        -------
        RecoveryMap
            Recovery map with therestricted basis.

        """
        pass
    
    @abstractmethod
    def project_observation(self, indices):
        """
        Create a new RecoveryMap initialized with a restricted observation basis.

        Parameters
        ----------
        indices : (k,) ndarray
            The indices to which the observation basis `W` is restricted.

        Returns
        -------
        RecoveryMap
            Recovery map with the restricted observation basis.

        """
        pass
    
    @abstractmethod
    def compute_state_(self, w, **kwargs):
        pass
    
    def compute_state(self, w, **kwargs):
        with self.logger.block("Computing background terms"):
            result = self.compute_state_(w, **kwargs)
        return result

    def compute_correction(self, w, v):
        eta = np.linalg.solve(self.gramian, w - self.cross_gramian @ v)
        return eta
    
    def solve(self, w, correct=True, **kwargs):
        v = self.compute_state(w, **kwargs)
        u = self.V.lincomb(v.T)
        if correct:
            eta = self.compute_correction(w, v)
            u += self.W.lincomb(eta.T)
        return u
    

class PbdwRecoveryMap(RecoveryMap):
    
    def __init__(self, V, W, gramian=None, cross_gramian=None, product=None, log_level=20):
        super().__init__(V, W, gramian, cross_gramian, product, log_level)

    def compute_state_(self, w):
        """
        

        Parameters
        ----------
        w : (m,k) ndarray
            Linear observations on `k` snapshots.

        Returns
        -------
        v : (n,k)
            Coefficients in the reduced space of the state.

        """
        if w.ndim == 1: w = w.reshape(-1,1)
            
        V, W = self.V, self.W
        n, m = len(V), len(W)
        WW, WV = self.gramian, self.cross_gramian
        A = np.block([[WW, WV],
                      [WV.conj().T, np.zeros((n,n))]])
        
        b = np.block([[w],
                      [np.zeros((n, w.shape[1]))]])
        v = np.linalg.solve(A, b)[m:,:]
    
        return v
    
    def project_background(self, indices):
        V = self.V[indices]
        CG = self.cross_gramian[:,indices]
        return self.with_(V=V, cross_gramian=CG)
    
    def project_observation(self, indices):
        W = self.W[indices]
        G = self.gramian[indices,:][:,indices]
        CG = self.cross_gramian[indices, :]
        return self.with_(W=W, gramian=G, cross_gramian=CG)
    
    
class DicRecoveryMap(RecoveryMap):

    def __init__(self, V, W, gramian=None, cross_gramian=None, product=None, log_level=20,
                 manifold_distance=None):
        
        super().__init__(V, W, gramian, cross_gramian, product, log_level, manifold_distance)
        assert np.allclose(self.gramian, np.eye(len(W)))
        assert (manifold_distance is None) or (len(V) + len(W) == manifold_distance.lhs.source.dim)
    

    def compute_state_path(self, w, alpha=0, weights=None, scale=1e3, solver='sklearn',
                       ols=True, return_path=True, **kwargs):
        """
        Compute the state path along the LARS algorithm. This method only calls
        lars_weighted_path.

        Parameters
        ----------
        w : (m,) ndarray
            Linear measurements on the snapshot.
        alpha : float, optional
            See `lars_weighted_path`. The default is 0.
        weights : (k,) ndarray, optional
            See `lars_weighted_path`. The default is None.
        scale : float, optional
            See `lars_weighted_path`. The default is 1e3.
        solver : str, optional
            See `lars_weighted_path`. The default is 'sklearn'.
        ols : bool, optional
            See `lars_weighted_path`. The default is True.
        return_path : bool, optional
            See `lars_weighted_path`. The default is True.
        **kwargs : dict
            See `lars_weighted_path`.

        Returns
        -------
        v : (K, path_length+1) ndarray
            Dictionary coefficient of the state along the LARS path.
        alphas : (path_length+1,) ndarray
            See `lars_weighted_path`.
            
        """
        v, alphas = lars_weighted_path(
            self.cross_gramian, w, alpha, weights, scale, solver, ols, return_path, **kwargs
            )
        return v, alphas
        
    def compute_correction_path(self, w, v):
        """
        Compute the correction coefficients associated to the state coefficients 
        along the LARS path.

        Parameters
        ----------
        w : (m,) ndarray
            Linear measurements on the snapshot.
        v : (K, path_length) ndarray
            Dictionary coefficients of the states along the LARS path.

        Returns
        -------
        eta : (m, path_length) ndarray
            Correction coefficients.

        """
        assert w.ndim == 1
        ind = np.zeros(v.shape[1], dtype=int)
        eta = self.compute_correction(w.reshape(-1,1)[:,ind], v)
        return eta
    
    def compute_state_(self, w, alpha=0, weights=None, scale=1e3, solver='sklearn',
                       ols=True, return_path=True, **kwargs):
        """
        Compute the dictionary based multispace recovery, given `m` linear measurements
        and a dictionary of size `K`. 

        Parameters
        ----------
        w : (m,k) ndarray
            Linear measurements on the `k` snapshots.
        alpha : float, optional
            See `lars_weighted_path`. The default is 0.
        weights : (k,) ndarray, optional
            See `lars_weighted_path`. The default is None.
        scale : float, optional
            See `lars_weighted_path`. The default is 1e3.
        solver : str, optional
            See `lars_weighted_path`. The default is 'sklearn'.
        ols : bool, optional
            See `lars_weighted_path`. The default is True.
        return_path : bool, optional
            See `lars_weighted_path`. The default is True.
        **kwargs : dict
            See `lars_weighted_path`.

        Returns
        -------
        v : (K, k) ndarray
            Dictionary coefficients of the states whose corresponding recoveries 
            are the closest to the manifold.
            
        """
        if w.ndim == 1: w = w.reshape(-1,1)
        v = np.zeros((len(self.V), w.shape[1]))
        for i in range(w.shape[1]):
            v[:,i] = self.compute_state__(w[:,i], alpha, weights, scale, solver, ols, return_path, **kwargs)
        return v
        
    def compute_state__(self, w, alpha=0, weights=None, scale=1e3, solver='sklearn',
                       ols=True, return_path=True, **kwargs):
        """
        Compute the dictionary based multispace recovery given linear measurements
        on one snapshot.

        Parameters
        ----------
        w : (m,) ndarray
            Linear measurements on the snapshot.
        alpha : float, optional
            See `lars_weighted_path`. The default is 0.
        weights : (k,) ndarray, optional
            See `lars_weighted_path`. The default is None.
        scale : float, optional
            See `lars_weighted_path`. The default is 1e3.
        solver : str, optional
            See `lars_weighted_path`. The default is 'sklearn'.
        ols : bool, optional
            See `lars_weighted_path`. The default is True.
        return_path : bool, optional
            See `lars_weighted_path`. The default is True.
        **kwargs : dict
            See `lars_weighted_path`.

        Returns
        -------
        v : (K,) ndarray
            Dictionary coefficients of the state whose corresponding recovery 
            is the closest to the manifold.  
        
        """
        assert w.ndim == 1
        v, _ = self.compute_state_path(w, alpha, weights, scale, solver, ols, return_path, **kwargs)
        eta = self.compute_correction_path(w, v)
        coefs = np.block([[v], [eta]])
        distances, _ = self.manifold_distance.evaluate(coefs)
        v = coefs[:v.shape[0],distances.argmin()]
        return v

    def solve_path(self, w, alpha=0, weights=None, scale=1e3, solver='sklearn',
                   ols=True, return_path=True, **kwargs):
        assert w.ndim == 1
        v, _ = self.compute_state_path(w, alpha, weights, scale, solver, ols, return_path, **kwargs)
        eta = self.compute_correction_path(w, v)
        u = self.V.lincomb(v.T) + self.W.lincomb(eta.T)
        
        coefs = np.block([[v], [eta]])
        distances, _ = self.manifold_distance.evaluate(coefs)
        return u, distances

    def project_background(self, indices):
        indices = np.array(indices)
        V = self.V[indices]
        G = self.cross_gramian[:,indices]
        ind = np.concatenate( (indices, len(self.V)+np.arange(len(self.W))) )
        mdist = self.manifold_distance.project(ind)
        return self.with_(V=V, cross_gramian=G, manifold_distance=mdist)
    
    def project_observation(self, indices):
        indices = np.array(indices)
        W = self.W[indices]
        G = self.gramian[indices,:][:,indices]
        CG = self.cross_gramian[indices, :]
        ind = np.concatenate( (np.arange(len(self.V)), len(self.V) + indices) ) 
        mdist = self.manifold_distance.project(ind)
        return self.with_(W=W, gramian=G, cross_gramian=CG, manifold_distance=mdist)






