#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:21:42 2023

@author: apasco
"""

import numpy as np
from pymor.core.base import ImmutableObject, abstractmethod
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, LincombOperator, ZeroOperator
from pymor.parameters.functionals import ParameterFunctional
from pymor.algorithms.projection import project
from pymor.algorithms.simplify import contract, expand
from scipy.optimize import lsq_linear



class ManifoldDistance(ImmutableObject):

    """
    Class allowing to compute the distance to the manifold of some (reduced) vector.

    """
    
    @abstractmethod
    def evaluate_(self, u, **kwargs):
        pass
    
    def evaluate(self, u, return_mu=False, **kwargs):
        """
        Compute the distance of `u` to the manifold. It is most likely that it is not
        the vector itself, but rather the coefficients in some reduced basis.

        Parameters
        ----------
        u : (n,k) ndarray
            The coefficients of k vectors of size dimension n whose distanced 
            to the manifold is to be computed.
        return_mu : bool, optional
            If True, returns also the parameter value which minimizes the residual. 
            The default is False.
        **kwargs : dict
            Key word arguments for the `evaluate_` method of the specific subclass.

        Returns
        -------
        distances : (k,) ndarray of float
            Distance to the manifold 
        mus : (k,) list of Mu
            Parameter value minimizing the residual norm.

        """
        if isinstance(u, np.ndarray):
            u = self.lhs.source.from_numpy(u.T)
        with self.logger.block("Estimating distances to manifold"):
            distances = np.zeros(len(u))
            mus = []
            for i, ui in enumerate(u):
                dist, mu_min = self.evaluate_(ui, **kwargs)
                distances[i] = dist
                mus.append(mu_min)
                self.logger.info(f"Vector {i} distance to manifold {dist:.3e} for parameter value {mu_min}")
        return distances, mus
    
    def project(self, indices):
        """
        Create a new ManifoldDistance initialized with an operator with restricted 
        source dofs.

        Parameters
        ----------
        indices : (k,) ndarray
            The source dofs to which the `self.lhs` is restricted.

        Returns
        -------
        ManifoldDistance
            The projected ManifoldDistance.

        """
        lhs = self.lhs
        u = lhs.source.zeros(len(indices))
        u = u.to_numpy()
        u[np.arange(len(indices)), indices] = 1.
        u = lhs.source.from_numpy(u)
        new_lhs = project(lhs, None, u)
        return self.with_(lhs=new_lhs, check_valid=False)
    
class ResidualDistanceDiscrete(ManifoldDistance):
    """
    Class allowing to compute the distance to the manifold for general operator and rhs,
    by computing the residual norm on a finite parameter set.

    Parameters
    ----------
    lhs : Operator
        Left-hand side of the residual.
    rhs : Operator
        Right-hand side of the residual.
    mus : list of Mu
        Finite parameter set on which the residual is minimized.
    product : Operator, optional
        Product operator w.r.t which the residual norm is computed.
        The default is None.
    log_level : int, optional
        Logger level. The default is 20.

    """
    
    def __init__(self, lhs, rhs, mus, product=None, log_level=20, **kwargs):
        
        if product is None:
            product = IdentityOperator(lhs.range)
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        
    
    def evaluate_(self, u):
        mus = self.mus
        rnorms = np.zeros(len(mus))

        for i, mu in enumerate(mus):
            residual = self.lhs.apply(u, mu) - self.rhs.as_range_array(mu)
            rnorms[i] = residual.norm(self.product)
        
        ind = rnorms.argmin()
        mu_min = mus[ind]
        distance = rnorms[ind]
        
        return distance, mu_min
            

class ResidualDistanceAffine(ManifoldDistance):
    
    """
    Class allowing to compute the distance to the manifold when the operator and 
    rhs both depend affinely on the parameter, which lies in a rectangle.

    Parameters
    ----------
    lhs : LincombOperator
        Left-hand side of the residual.
    rhs : LincombOperator or NumpyMatrixOperator
        Right-hand side of the residual.
    param_bounds : tuple of numpy.ndarray
        Parameter bounds for the constrained least-squares. See more details in 
        the doc of `lsq_linear`.
    log_level : int, optional
        Logger level. The default is 20.
    check_valid : bool, optional
        If True, the lhs and rhs are copied and re-arranged into a suitable form.
        More precisely, the last affine coefficient must be non-parametric.
        The default is True.

    """
    
    def __init__(self, lhs, rhs, param_bounds, log_level=20, check_valid=True):
        
        assert isinstance(lhs, LincombOperator)
        assert isinstance(rhs, (LincombOperator, NumpyMatrixOperator))
        assert all([not(o.parametric) for o in lhs.operators])
        
        if check_valid:
            lhs = contract(expand(lhs))
            rhs = contract(expand(rhs))
            
            # make the rhs a LincombOperator for simplicity
            if isinstance(rhs, NumpyMatrixOperator):
                rhs = 1.*rhs
            
            # if no non-parametric part
            if all([isinstance(c,ParameterFunctional) for c in lhs.coefficients]):
                n = lhs.source.dim
                op_lst = lhs.operators
                op_lst.append(ZeroOperator(lhs.range, lhs.source))
                c_lst = lhs.coefficients
                c_lst.append(1.)
                lhs = lhs.with_(operators=op_lst, coefficients=c_lst)
            
            # same for the rhs
            if all([isinstance(c,ParameterFunctional) for c in rhs.coefficients]):
                n = lhs.source.dim
                op_lst = lhs.lhs + (ZeroOperator(lhs.range, lhs.source),)
                c_lst = lhs.coefficients + (1.,)
                lhs = lhs.with_(operators=op_lst, coefficients=c_lst)
        
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
    
    def build_ls(self, u):
        """
        Build the least-squares system for the residual minimization.

        Parameters
        ----------
        u : VectorArray of length 1.
            The coefficients of the vector whose distance to the manifold is to be computed.

        Returns
        -------
        G : (k, p) ndarray.
            Left-hand side of the least-squares system. `p` is the number of parameter
            and `k` is the range dimension of `self.lhs`.
        g : (k,) ndarray.
            Right-hand side of the least-squares system.

        """
        op = project(self.lhs, None, u)
        
        # lhs for least squares
        G = [o.as_range_array().to_numpy().reshape(-1) for o in op.operators[:-1]] 
        G = G + [-o.as_range_array().to_numpy().reshape(-1) for o in self.rhs.operators[:-1]] 
        G = np.array(G).T
        
        # rhs for least squares
        g = (self.rhs.operators[-1].as_range_array() - op.operators[-1].as_range_array()).to_numpy().reshape(-1)
        
        return G, g
    
    def evaluate_(self, u):
        G, g = self.build_ls(u)
        res = lsq_linear(G, g, self.param_bounds)
        distance = np.linalg.norm(res.fun)
        mu_min = self.lhs.parameters.parse(res.x)
        return distance, mu_min
    