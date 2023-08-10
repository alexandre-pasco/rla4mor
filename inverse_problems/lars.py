#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:41:15 2023

@author: apasco
"""


import numpy as np
import spams
from sklearn.linear_model import lars_path



def _lars_path(D, X, alpha, solver, ols, return_path, **kwargs):
    """
    Compute the LARS regularization path for the dictionary D and data X, based on 
    the `sklearn.linear_model.lars_path` implementation of LARS. Note that the
    latter does not rescale the data X, and scaling up X leads to longer path.

    Parameters
    ----------
    D : (m, K) ndarray
        Dictionary of K basis vectors.
    X : (m,) ndarray
        Data of m observations.
    alpha_min : float
        See sklearn doc.
    ols : bool
        If True, compute the orthogonal least-square projection along the lars path.
    return_path : bool
        If True, return the path along the LARS algorithm. Else, return only the last
        coefficients of the last iteration.
    **kwargs : dict
        Key word arguments for the specific solver used.

    Returns
    -------
    path : (K, path_length + 1)
        Solution path. If return_path is false,
    alphas : (path_length + 1,)
        See sklearn doc.

    """
    if solver == 'sklearn':
        if return_path == False:
            path, alphas = np.zeros((D.shape[1], X.shape[1])), np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                pp, aa = _lars_path_sklearn(D, X[:,i], alpha, ols, False, **kwargs)
                path[:,i], alphas[i] = pp.reshape(-1), aa
        else:
            path, alphas = _lars_path_sklearn(D, X, alpha, ols, True, **kwargs)
    elif solver == 'spams':
        path, alphas = _lars_path_spams(D, X, alpha, ols, return_path, **kwargs)
    return path, alphas




def _lars_path_sklearn(D, X, alpha, ols, return_path, **kwargs):
    """
    Compute the LARS regularization path for the dictionary D and data X, based on 
    the `sklearn.linear_model.lars_path` implementation of LARS. Note that the
    latter does not rescale the data X, and scaling up X leads to longer path.

    Parameters
    ----------
    D : (m, K) ndarray
        See sklearn doc.
    X : (m,) ndarray
        See sklearn doc.
    alpha_min : float
        See sklearn doc.
    ols : bool
        If True, compute the orthogonal least-square projection along the lars path.
    return_path : bool
        See sklearn doc.
    **kwargs : dict
        See sklearn doc.

    Returns
    -------
    path : (K, path_length + 1)
        Solution path.
    alphas : (path_length + 1,)
        See sklearn doc.

    """
    alphas, _, coefs = lars_path(
        D, X, alpha_min=alpha/D.shape[0], method='lasso', 
        return_path=return_path, **kwargs
    
        )
    # sklearn solves the lasso pb rescaled by 1/n_samples
    alphas = alphas * D.shape[0] 
    
    if not(return_path): 
        coefs = coefs.reshape(-1,1)
    
    # Orthogonal least squares at each lars step
    if ols: 
        path = np.zeros(coefs.shape)
        for i in range(coefs.shape[1]):
            ind = coefs[:,i].nonzero()[0]
            x, _, _, _ = np.linalg.lstsq(D[:,ind], X, rcond=None)
            path[ind,i] = x
    
    # Plain lars path
    else: path = coefs
        
    return path, alphas


def _lars_path_spams(D, X, alpha, ols, return_path, **kwargs):
    """
    Compute the LARS regularization path for the dictionary D and data X, based on 
    the `sklearn.linear_model.lars_path` implementation of LARS. Note that the
    latter does not rescale the data X, and scaling up X leads to longer path.

    Parameters
    ----------
    D : (m, K) ndarray
        See sklearn doc.
    X : (m,) ndarray
        See sklearn doc.
    alpha : float
        See sklearn doc.
    ols : bool
        If True, compute the orthogonal least-square projection along the lars path.
    return_path : bool
        See sklearn doc.
    **kwargs : dict
        See sklearn doc.

    Returns
    -------
    path : (K, path_length + 1)
        Solution path.
    alphas : (path_length + 1,)
        See sklearn doc.

    """
    if return_path:
        Warning("Warning: using spams for LARS path leads to memory leak, and does not gives alpha's path")
    res = spams.lasso(X, D, lambda1=alpha, return_reg_path=return_path, **kwargs)
    if not(return_path):
        path = res.toarray()
        alphas = np.array([alpha])
    else:
        path_sp = res[1]
        last_ind = np.abs(path_sp).max(axis=0).nonzero()[0][-1]
        path = path_sp[:,:last_ind+1]
        alphas = alpha * np.ones(path.shape[1])
    return path, alphas




def lars_weighted_path(D, X, alpha=0, weights=None, scale=1e3 ,solver='sklearn', 
                       ols=True, return_path=True, **kwargs):
    """
    

    Parameters
    ----------
    D : (m, K)
        Dictionary.
    X : (m,)
        Data.
    alpha : float
        Minimal regularization parameter.
    weights : (K,) ndarray, optional
        Weights. If None, the weights are 1. The default is None.
    scale : float, optional
        Scaling of the date X. Larger scale implies longer path. 
        The default is 1e3.
    solver : str, optional
        LARS implementation. Must be `sklearn` or `spams`. Note that `spams`
        leads to memory leaks. The default is 'sklearn'.
    ols : bool, optional
        If True, compute the orthogonal least-square projection along the lars path. 
        The default is False.
    return_path : bool
        See sklearn doc.
    **kwargs : dict
        See sklearn doc.

    Returns
    -------
    path : (K, path_length)
        Solution path.
    alphas : (path_length,)
        See sklearn doc.

    """

    if weights is None:
        weights = np.ones(D.shape[1])
    elif weights.ndim == 1:
        weights = weights
    
    D_ = np.asfortranarray(D / weights)
    X_ = np.asfortranarray(X * scale)
    alpha_ = alpha * scale / D.shape[1]
    
    path_, alphas_ = _lars_path(D_, X_, alpha_, solver, ols, return_path, **kwargs)
    
    path = path_ / weights.reshape(-1,1) / scale
    alphas = alphas_ / scale
    
    if return_path: path = path[:,1:]
    
    return path, alphas
