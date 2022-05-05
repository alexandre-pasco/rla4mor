#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:33:07 2022

@author: pasco
"""


import numpy as np
import spams
from sklearn.linear_model import orthogonal_mp


def sparse_minres_solver(V, F, r=None, tol=None, return_reg_errors=False, solver='stepwise'):
    if solver == 'omp':
        ar, reg_errors = sparse_minres_solver_omp(V, F, r=r, tol=tol, return_reg_errors=return_reg_errors)
    elif solver == 'omp_sklearn':
        ar, reg_errors = sparse_minres_solver_sklearn(V, F, r=r, tol=tol, return_reg_errors=return_reg_errors)
    elif solver == 'omp_spams':
        ar, reg_errors = sparse_minres_solver_spams(V, F, r=r, tol=tol, return_reg_errors=return_reg_errors)
    elif solver == 'stepwise':
        ar, reg_errors = sparse_minres_solver_stepwise(V, F, r=r, tol=tol, return_reg_errors=return_reg_errors)
    else:
        print("Sparse minres solver not valid.") 
    return ar, reg_errors

def sparse_minres_solver_spams(V, F, r=None, tol=None, return_reg_errors=False):
    """
    Use the Orthogonal Matching Pursuit from SPAMS to approximate the 
    solution of the least square problem 
        \min_x \|x\|_0 such that \| Vx - F \| < tol
    with the constraint \|x\|_0 <=r.
    

    Parameters
    ----------
    V : (N,K) ndarray
        The dictionary of column vectors, with real entries.
    F : (N,1) ndarray
        The constant term of the minimisation problem, with real entries.
    r : int, optional
        The maximul number of vectors to use. If None, all vectors can be used. 
    tol : float, optional
        The residual norm target. If None, 0 is taken as tolerance.
    return_reg_errors : Bool, optional
        If True, the errors along the regularization path are returned.
        /!/ It seems like spams does not give back the memory if the reg_path
        is returned. It can lead to memory issues. The default is False.

    Returns
    -------
    ar : (K,) ndarray
        The approximated solution to the sparse least squares problem
    reg_errors : (K,) ndarray
        the residuals norms along the regularization path, normalized by $\|F\|$.

    """
    
    if tol is None : tol = 0
    # normalize V (required by spams ?) and F
    normV = np.linalg.norm(V, axis=0)
    normF = np.linalg.norm(F)
    W = V / normV
    B = F / normF

    # tranForme them in fortran array, so that spams.omp can use them
    Wf = np.asfortranarray(W)
    Bf = np.asfortranarray(B)

    if return_reg_errors:
        (x, reg_path) = spams.omp(Bf, Wf, L=r, eps=tol ** 2, return_reg_path=True, numThreads=1)
    else:
        x = spams.omp(Bf, Wf, L=r, eps=tol ** 2, return_reg_path=False, numThreads=1)
    # rescale the solution, so that it corresponds to min(V*ar - F)
    ar = (x.toarray().reshape(-1) / normV) * normF

    # comput the reg errors if necessary
    indices = x.indices
    W_actif = W[:,indices]
    reg_errors = np.zeros(r)
    if return_reg_errors:
        i = 0
        while i < r and np.max(np.abs(reg_path[:, i])) > 0:
            x = reg_path[indices, i]
            residual = np.dot(W_actif, x) - B[:, 0]
            err = np.linalg.norm(residual)
            reg_errors[i] = err
            i = i + 1
        while i < r:
            reg_errors[i] = reg_errors[i - 1]
            i = i + 1
    return ar, reg_errors


def sparse_minres_solver_sklearn(V, F, r=None, tol=None, return_reg_errors=False):
    """
    Use the Orthogonal Matching Pursuit from sklearn to approximate the 
    solution of the least square problem 
        \min_x \|x\|_0 such that \| Vx - F \| < tol
    with the constraint \|x\|_0 <=r.
    

    Parameters
    ----------
    V : (N,K) ndarray
        The dictionary of column vectors, with real entries.
    F : (N,1) ndarray
        The constant term of the minimisation problem, with real entries.
    r : int, optional
        The maximul number of vectors to use. If None, all vectors can be used. 
    tol : float, optional
        The residual norm target. If None, 0 is taken as tolerance.
    return_reg_errors : Bool, optional
        If True, the errors along the regularization path are returned.

    Returns
    -------
    ar : (K,) ndarray
        The approximated solution to the sparse least squares problem
    reg_errors : (K,) ndarray
        the residuals norms along the regularization path, normalized by $\|F\|$.

    """
    
    
    if (tol is None) or (tol == 0): eps = None
    else: eps = tol ** 2

    # normalize V and F
    normV = np.linalg.norm(V, axis=0)
    normF = np.linalg.norm(F)
    W = V / normV
    B = F / normF

    x = orthogonal_mp(W, B, n_nonzero_coefs=r, tol=eps,
                      return_path=return_reg_errors, precompute=False)

    # comput the reg errors if necessary
    reg_errors = np.zeros(r)
    if x.ndim>1:
        ar = (x[:, -1] / normV) * normF
        err_omp = np.linalg.norm(np.dot(W, x) - B, axis=0)
        if return_reg_errors:
            reg_errors[:err_omp.shape[0]] = err_omp
            for i in range(err_omp.shape[0], r): 
                reg_errors[i] = reg_errors[i - 1]
    else:
        ar = (x / normV) * normF
        err_omp = np.linalg.norm(np.dot(W, x) - B[:, 0])

    return ar, reg_errors


def sparse_minres_solver_omp(V, F, r=None, tol=None, return_reg_errors=False):
    """
    Use the Orthogonal Matching Pursuit to approximate the 
    solution of the least square problem 
        \min_x \|x\|_0 such that \| Vx - F \| < tol
    with the constraint \|x\|_0 <=r.
    

    Parameters
    ----------
    V : (N,K) ndarray
        The dictionary of column vectors, with real or complex entries.
    F : (N,1) ndarray
        The constant term of the minimisation problem, with real or complex entries.
    r : int, optional
        The maximul number of vectors to use. If None, all vectors can be used. 
    tol : float, optional
        The residual norm target. If None, 0 is taken as tolerance.
    return_reg_errors : Bool, optional
        If True, the errors along the regularization path are returned.

    Returns
    -------
    ar : (K,) ndarray
        The approximated solution to the sparse least squares problem
    reg_errors : (K,) ndarray
        the residuals norms along the regularization path, normalized by $\|F\|$.

    """
    
    if r is None: r = V.shape[1]
    if tol is None: tol = 0
    
    normV = np.linalg.norm(V, axis=0)
    normF = np.linalg.norm(F)
    W = V / normV
    B = F / normF
    i = 0
    residual = B[:, 0]
    indices = []
    reg_errors = np.zeros(r)
    err = 1.0
    while i < min(V.shape[1], r) and err > tol:
        p_i = int(np.argmax(np.abs(np.dot(residual.conj().T, W))))
        w = W[:, p_i]
        for p_j in indices:
            w = w - W[:,p_j] * np.dot(W[:,p_j].conj().T, w)
            w = w / np.linalg.norm(w)
        W[:, p_i] = w
        indices.append(p_i)
        residual = residual - w * np.dot(w.conj().T, residual)
        err = np.linalg.norm(residual)
        if return_reg_errors:
            reg_errors[i] = err
        i = i + 1
    while i < r:
        reg_errors[i] = reg_errors[i - 1]
        i = i + 1
    indices = np.array(indices)
    
    V_actif = V[:,indices]
    x, _, _, _ = np.linalg.lstsq(V_actif, B[:,0], rcond=None)
    ar = np.zeros(V.shape[1], dtype = x.dtype)
    ar[indices] = x * normF
    
    return ar, reg_errors
    


def sparse_minres_solver_stepwise(V, F, r=None, tol=None, return_reg_errors=False):
    """
    Use the Stepwise Projection Algorithm to approximate the 
    solution of the least square problem 
        \min_x \|x\|_0 such that \| Vx - F \| < tol
    with the constraint \|x\|_0 <=r.
    

    Parameters
    ----------
    V : (N,K) ndarray
        The dictionary of column vectors, with real or complex entries.
    F : (N,1) ndarray
        The constant term of the minimisation problem, with real or complex entries.
    r : int, optional
        The maximul number of vectors to use. If None, all vectors can be used. 
    tol : float, optional
        The residual norm target. If None, 0 is taken as tolerance.
    return_reg_errors : Bool, optional
        If True, the errors along the regularization path are returned.

    Returns
    -------
    ar : (K,) ndarray
        The approximated solution to the sparse least squares problem
    reg_errors : (K,) ndarray
        the residuals norms along the regularization path, normalized by $\|F\|$.

    """
    
    if r is None: r = V.shape[1]
    if tol is None: tol = 0
    
    normV = np.linalg.norm(V, axis=0)
    normF = np.linalg.norm(F)
    W = V / normV
    B = F / normF
    i = 0
    residual = B[:, 0]
    indices = []
    remaining_indices = [k for k in range(V.shape[1])]
    reg_errors = np.zeros(r)
    err = 1.0
    while i < min(V.shape[1], r) and err > tol:
        p_i = int(np.argmax(np.abs(np.dot(residual.conj().T, W))))
        indices.append(p_i)
        remaining_indices.remove(p_i)
        w = W[:, p_i]
        residual = residual - w * np.dot(w.conj().T, residual)
        inner = np.dot(w.conj().T, W)
        W = W - w.reshape(-1, 1) * inner
        for j in remaining_indices:
            W[:, j] = W[:, j] / np.linalg.norm(W[:, j])
        err = np.linalg.norm(residual)
        if return_reg_errors:
            reg_errors[i] = err
        i = i + 1
    while i < r:
        reg_errors[i] = reg_errors[i - 1]
        i = i + 1
    indices = np.array(indices)
    
    V_actif = V[:,indices]
    x, _, _, _ = np.linalg.lstsq(V_actif, B[:,0], rcond=None)
    ar = np.zeros(V.shape[1], dtype = x.dtype)
    ar[indices] = x * normF
    
    return ar, reg_errors
    
    



