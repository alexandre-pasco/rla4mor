#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:00:29 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.simplify import expand, contract
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.core.defaults import set_defaults
from pymor.operators.constructions import IdentityOperator, InverseOperator, LincombOperator
from pymor.operators.interface import as_array_max_length
from pymor.parameters.base import Mu

from pymor.algorithms.projection import project

from embeddings.embeddings import IdentityEmbedding
from preconditioners.preconditioned_rom import PreconditionedRom


class PreconditionedReductor(BasicObject):
    """
    Class implementing a sketched preconditionned reductor. Several attributes 
    of this class are dictionaries. The associated keys correspond to the different
    spectral norm estimators (i.e. sketched HS norms).
    
    Attibutes
    ---------
    fom : StationaryModel
        The full order model to reduce.
    reduced_basis : VectorArray
        The reduced basis used for the Galerkin projection.
    source_bases : dict
        Dictionary containing the bases of the source space for the error 
        estimators.
    range_bases : dict
        Dictionary containing the bases of the range space for the error 
        estimators.
    source_embeddings : dict
        Dictionary containing the embeddings of the source space for the error 
        estimators.
    range_embeddings : dict
        Dictionary containing the embeddings of the range space for the error 
        estimators.
    vec_embeddings : dict
        Dictionary containing the embeddings to vectorize the sketched range 
        space for the error estimators.
    intermediate_bases : dict
        Dictionary containing the image of some operators, in order to improve 
        the numerical stability when assembling the Galerkin system. The key 
        'lhs' contains an ortho basis of fom.operator with reduced_basis as source.
        The key 'rhs' contains an ortho basis of fom.rhs.
    product : Operator
        A symetric positive definite operator, encoding the inner product.
    inv_product: Operator
        The inverse of the product operator.
    stable_galerkin : bool, optional
        If True, uses the intermediate basis projection to improve stability of
        the preconditioned galerkin system.
    
    hs_estimators_lhs: dict
        Dictionary containing lists of LincombOperator. For each key, 
        the ith element of the list is the affine decomposition of the sketch 
        of Pi @ self.fom.lhs, where Pi is the ith direction in the 
        preconditioners space.
    hs_estimators_rhs: dict
        Dictionary containing a ndarray, which are the sketch of the identity 
        operator for the different keys.
    sketched_source_bases: dict of VectorArray
        
    sketched_range_bases: dict of VectorArray
    
    dtype: type

    """
    
    
    def __init__(self, fom, reduced_basis, source_bases, range_bases,
                 source_embeddings, range_embeddings, vec_embeddings,
                 residual_embedding, intermediate_bases=None, product=None, 
                 inv_product=None, stable_galerkin=True, dtype=float, log_level=20):
        
        assert source_bases.keys() == range_bases.keys()
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        self.mu_added = []
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
        if inv_product is None:
            self.inv_product = InverseOperator(self.product)
        
        if intermediate_bases is None:
            self.stable_galerkin = False
        
        # for the Galerkin system
        self.prom = PreconditionedRom(
            fom, reduced_basis, residual_embedding, intermediate_bases=intermediate_bases,
            product=self.product, stable_galerkin=self.stable_galerkin, log_level=log_level)
        
        # for the HS indicators
        self.sketched_source_bases = dict()
        self.sketched_range_bases = dict()
        for key, V in source_bases.items():
            S = source_embeddings[key]
            if isinstance(S, IdentityEmbedding) or V is None:
                self.sketched_source_bases[key] = V
            else:
                self.sketched_source_bases[key] = V.lincomb(S.get_matrix())
        
        for key, V in range_bases.items():
            S = range_embeddings[key]
            if isinstance(S, IdentityEmbedding) or V is None:
                self.sketched_range_bases[key] = V
            else:
                self.sketched_range_bases[key] = V.lincomb(S.get_matrix())
        
        # Initialize the hs_estimator systems
        self.hs_estimators_lhs = dict()
        self.hs_estimators_rhs = dict()
        for key in source_bases.keys():
            self.hs_estimators_lhs[key] = []
            self.hs_estimators_rhs[key] = self.sketch_operator(
                self.product, key).matrix.reshape(-1)
        
        # change default value to be able to use to_matrix
        if len(reduced_basis) >= as_array_max_length():
            set_defaults({'pymor.operators.interface.as_array_max_length.value':1+len(reduced_basis)})
        
    def sketch_operator(self, operator, key):
        """
        Compute the sketch of a U -> U' operator, and return the projected operator, 
        which is a LincombOperator with a source dimension equal to 1. It will
        be used to build the columns of the systems for the HS norm estimations.

        Parameters
        ----------
        operator : Operator
            The operator to sketch.
        key : str
            The dictionary key to select the bases and embeddings to use.

        Returns
        -------
        result : LincombOperator
            The sketched operator, which is actually a linear form.

        """
        self.logger.info(f"sketching {operator.name} {key}")
        Vr = self.sketched_range_bases[key]
        Vs = self.sketched_source_bases[key]
        Ru = self.product
        
        if Vr is None:
            Vr = self.product.apply_inverse(self.range_embeddings[key].as_source_array())
        
        if Vs is None:
            op = project(operator.H, None, Vr)
            new_op = contract(expand(self.source_embeddings[key] @ self.inv_product @ op)).H
            
            # optional : simplify the conjugate conjugate functionals
            if isinstance(new_op, LincombOperator):
                # we do not use the op.coefficients, because the ordering may have change
                coefs = [c if np.isscalar(c) else c.functional.functional for c in new_op.coefficients]
                new_op = new_op.with_(coefficients=coefs)
            
    
        else:
            # we want to first compute operator.H @ Vr
            op = project(operator, Vr, None)
            new_op = project(op, None, Vs)
        
        result = contract(expand(self.vec_embeddings[key] @ new_op))
        
        return result
    
 
    def estimate_quasi_optimality(self, mu_p):
        assert 'u_ur' in self.range_bases.keys()
        delta_2 = self._estimate_hs(mu_p, 'u_ur')
        delta_3 = self._compute_spectral(mu_p)
        if delta_3 >= 1:
            self.logger.warning("Quasi optimality bound not defined")
            result = 0
        else:
            result = 1 + delta_2 / (1 - delta_3)
        return result
    
    def _compute_spectral(self, mu_p):
        A, b = self.assemble_rom_system(mu_p)
        _, s, _ = np.linalg.svd(A - np.eye(A.shape[0]))
        snorm = s.max()
        return snorm

    
    def _estimate_hs(self, mu_p, key):
        W, h = self.assemble_hs_estimator(mu_p, key)
        residual = W @ mu_p['precond'] - h
        rnorm = np.linalg.norm(residual)
        return rnorm
    
    def assemble_hs_estimator(self, mu, key):
        """
        Assemble the least-squares system for the corresponding key and parameter
        value. Minizing $y = \min_{x} ||Wx - h||$ gives the preconditioner 
        $P=\Sum_{i=1}^p y_i P_i$ which minimizes the sketched HS norm for the
        corresponding key and mu.^

        Parameters
        ----------
        mu : Mu
            Parameter value.
        key : string
            The key to identify some HS norm estimator.

        Returns
        -------
        W : (k,p) ndarray
            Each column is the sketch of Pi @ self.fom.operator, for Pi in 
            the added precontitioners.
        h : (k,) ndarray
            Sketch of the identity operator.

        """
        lst = self.hs_estimators_lhs.get(key)
        assert not(lst is None) and len(lst)>0
        h = self.hs_estimators_rhs[key]
        W = np.zeros((lst[0].range.dim, len(lst)), dtype=self.dtype)
        for i, column_op in enumerate(lst):
            W[:,i] = column_op.assemble(mu).matrix.reshape(-1)
        return W, h

    def minimize_hs_estimator(self, mu, key):
        """
        Minimize the sketched HS norm of (P @ self.fom.lhs - I) for the 
        corresponding key.

        Parameters
        ----------
        mu : Mu
            The parameter value.
        key : string
            The key to identify some HS norm estimator which is to minimize.

        Returns
        -------
        mu_p : Mu
            Parameter value, extended with a 'precond' key, which contains
            the coefficients of the preconditioners which minimises the 
            sketched HS norm for the corresponding key.
        rnorm : float
            Sketched HS norm of P @ self.fom.operator - I

        """
        W, h = self.assemble_hs_estimator(mu, key)
        x, rnorm2, _, _ = np.linalg.lstsq(W, h, rcond=None)
        mu_p_ = dict()
        for k in mu.keys():
            mu_p_[k] = mu[k]
        mu_p_['precond'] = x
        mu_p = Mu(mu_p_)
        rnorm = np.sqrt(rnorm2)
        return mu_p, rnorm
    
    def assemble_rom_system(self, mu_p):
        """
        

        Parameters
        ----------
        mu_p : Mu
            Parameter value, extended with a 'precond' key, which contains
            the coefficients of the preconditioners.

        Returns
        -------
        reduced_lhs : (r,r) ndarray
            The preconditioned galerkin lhs.
        reduced_rhs : (r,) ndarray
            The preconditioned galerkin rhs.

        """
        prom = self.prom
        # use to_matrix to avoid pymor warnings
        reduced_lhs = to_matrix(prom.rom.operator, None, mu_p)
        reduced_rhs = to_matrix(prom.rom.rhs, None, mu_p).reshape(-1)
        return reduced_lhs, reduced_rhs
    
    
    def solve(self, mu, key):
        """
        Solve the preconditioned galerkin system for parameter value mu,
        where the preconditioner is obtained by minimizing the HS norm 
        associated to 'key'.

        Parameters
        ----------
        mu : Mu
            Parameter value.
        key : string
            The key to identify some HS norm estimator.

        Returns
        -------
        u : NumpyVectorArray
            The coefficients within self.reduced_basis.

        """
        # Compute the parameter of the preconditioner
        mu_p, _ = self.minimize_hs_estimator(mu, key)
        # solve the corresponding preconditioned galerkin rom
        u = self.prom.rom.solve(mu_p)
        return u


    def add_preconditioner(self, P, mu=None):
        """
        Add the operator P within the preconditioner, for every spectral 
        norm estimators and for the galerkin system.

        Parameters
        ----------
        P : Operator
            The Operator to add to the preconditioners. If P is the inverse of
            self.fom.operator(mu) for some mu, then it is a new interpolation 
            point for approximating the inverse of the inverse of self.fom.operator.
        mu : Mu, optional
            The parameter corresponding to P. The default is None.
            
        Returns
        -------
        None.

        """
        with self.logger.block(f"Adding preconditioner at {mu}"):
            
            for key in self.hs_estimators_lhs.keys():
                op = self.sketch_operator(self.product @ P @ self.fom.operator, key)
                self.hs_estimators_lhs[key].append(op)
            
            self.prom.add_preconditioner(P, mu)
            self.mu_added.append(mu)

