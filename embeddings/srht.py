#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:37:56 2023

@author: apasco
"""

import numpy as np
from numba import njit

@njit(['void(f8[:])', 'void(c16[:])'])
def _fht_1d(a) -> None:
    """
    In-place Fast Hadamard Transform of array a.

    Parameters
    ----------
    a : ndarray of size (2**d,)
        The array on which the FHT is apply.
        
    """
    n = a.shape[0]
    d = int(np.log2(n))
    h = 1
    for p in range(d):
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    a /= 2**(d/2)

@njit(['void(f8[:,:])', 'void(c16[:,:])'])
def _fht_2d_sequential(a) -> None:
    """
    In-place Fast Hadamard Transform of the 2d array a. This function is 
    usefull only to apply the FHT on a few, moderate sized vectors.

    Parameters
    ----------
    a : ndarray of size (2**d,k)
        The array on which the FHT is apply, column by column, sequentially.

    """
    n = a.shape[0]
    d = int(np.log2(n))
    for k in range(a.shape[1]):
        h = 1
        for p in range(d):
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = a[j,k]
                    y = a[j + h,k]
                    a[j,k] = x + y
                    a[j + h,k] = x - y
            h *= 2
    a /= 2**(d/2)

@njit(['void(f8[:,:])', 'void(c16[:,:])'])
def _fht_2d(a) -> None:
    """
    In-place Fast Hadamard Transform of the 2d array a. This function
    outperforms the sequential one when a.shape[1] gets larger.

    Parameters
    ----------
    a : ndarray of size (2**d,k)
        The array on which the FHT is apply, all columns at the same time.

    """
    n = a.shape[0]
    d = int(np.log2(n))
    h = 1
    x = np.empty(a.shape[1], dtype=a.dtype)
    y = np.empty(a.shape[1], dtype=a.dtype)
    for p in range(d):
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x[:] = a[j,:]
                y[:] = a[j + h,:]
                a[j,:] = x + y
                a[j + h,:] = x - y
        h *= 2
    a /= 2**(d/2)




def fht(a) -> None:
    """
    In-place Fast Hadamard Transform of array a with n=2**d rows. Handle 
    float64 and complex128 types of data.

    Parameters
    ----------
    a : ndarray of size (2**d,) or (2**d,k)
        The array on which the FHT is apply. If a is a 2d array, the FHT is 
        apply on each column at the same time.

    """
    d = np.log2(a.shape[0])
    assert d%1 == 0
    assert a.ndim <= 2
    if a.ndim == 1:
        _fht_1d(a)
    elif a.shape[1] == 1:
        _fht_1d(a[:,0])
    elif a.ndim == 2:
        _fht_2d(a)


def srht(x, k, seed=None):
    """
    Compute the srht of each element of a numpy array ()

    Parameters
    ----------
    x : ndarray of shape (n,) or (n,m)
        Array on which the SRHT embedding is computed.
    k : int
        Dimension of the embedding space.
    seed : int, optional
        Seed of the random number generator. Default is None.
    Returns
    -------
    y : ndarray
        The srht transform of each columns of x.

    """
    # Managin the dimension of x
    assert x.ndim <= 2
    y = x.copy()
    if x.ndim == 1: 
        y = y.reshape(-1,1)
        
    n = y.shape[0]
    d = int(np.ceil(np.log2(n)))
    rademacher = np.random.RandomState(seed).choice([-1, 1], (n,1), True)
    sampling = np.random.RandomState(seed).choice(range(2**d), k, True)

    y = rademacher*y
    # Adding zeros if the vectors are not of size 2**d
    y = np.append(y, np.zeros((2**d-n, y.shape[1])), axis=0)
    # Applying the inplace Fast Hadamard Transform
    fht(y)
    # sampling and rescaling
    y = np.sqrt(n/k) * y[sampling, :]
    
    # reshape to the same ndim as x if necessary
    if x.ndim == 1:
        y = y.reshape(-1)
        
    return y

