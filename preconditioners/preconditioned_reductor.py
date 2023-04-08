#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:00:29 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.simplify import expand, contract
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.basic import StationaryModel
from pymor.models.interface import Model
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.constructions import IdentityOperator, LincombOperator, ZeroOperator, ConcatenationOperator, InverseOperator, VectorArrayOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Mu

from pymor.algorithms.image import estimate_image
from pymor.algorithms.projection import project

from embeddings.embeddings import *



class SketchedPreconditionedReductor(BasicObject):
    """
    Class implementing a sketched preconditionned reductor.
    
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
    """
    
    
    def __init__(self, fom, reduced_basis, source_bases, range_bases,
                 source_embeddings, range_embeddings, vec_embeddings,
                 intermediate_bases=None, product=None):
        
        assert source_bases.keys() == range_bases.keys()
        self.__auto_init(locals())
        self.mu_added = []
        self._last_rom = None
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
            
        # to allow separation of the galerkin system
        if intermediate_bases is None:
            intermediate_bases = dict()
            intermediate_bases['lhs'] = estimate_image(
                fom.operator.operators, domain=reduced_basis, 
                product=product, riesz_representatives=True)

            intermediate_bases['rhs'] = estimate_image(
                vectors=(fom.rhs,), product=product, riesz_representatives=True)
            
            self.intermediate_bases = intermediate_bases
        
        # for the HS indicators
        self.error_indicators_ls_lhs = dict()
        self.error_indicators_ls_rhs = dict()
        for key in source_bases.keys():
            self.error_indicators_ls_lhs[key] = []
            self.error_indicators_ls_rhs[key] = self._sketch_operator(
                IdentityOperator(fom.solution_space),key).matrix.reshape(-1)

    def _sketch_operator(self, operator, key):
        """
        Compute the sketch of an operator, and return the projected operator, 
        which is a linear form, whose range is the range of self.vec_embeddings[key]

        Parameters
        ----------
        operator : Operator
            The operator to sketch.
        key : str
            The dictionary key to select the bases and embeddings to use.

        Returns
        -------
        result : Operator
            The sketched operator, which is actually a linear form.

        """
        Vr = self.range_bases[key]
        op = self.vec_embeddings[key] @ self.range_embeddings[key]
        if not(Vr is None):
            op = op @ VectorArrayOperator(Vr, adjoint=True) @ self.product
        if self.source_bases[key] is None:
            Vs = self.product.apply_inverse(self.source_embeddings[key].as_source_array())
        else:
            Vs = self.source_bases[key].lincomb(self.source_embeddings[key].get_matrix())
        
        result = project(op @ operator, None, Vs)
        return result
    
 
    def estimate_quasi_optimality(self, mu_p):
        assert 'u_ur' in self.range_bases.keys()
        delta_2 = self._estimate_spectral(mu_p, 'u_ur')
        delta_3 = self._compute_spectral(mu_p)
        if delta_3 >= 1:
            Warning("Quasi optimality bound not defined")
            result = 0
        else:
            result = 1 + delta_2 / (1 - delta_3)
        return result
    
    def _compute_spectral(self, mu_p):
        A, b = self._assemble_rom_system(mu_p)
        _, s, _ = np.linalg.svd(A - np.eye(A.shape[0]))
        snorm = s.max()
        return snorm

    
    def _estimate_spectral(self, mu_p, key):
        W, h = self._assemble_spectral_estimator(mu_p, key)
        residual = W @ mu_p['precond'] - h
        rnorm = np.linalg.norm(residual)
        return rnorm
    
    def _assemble_spectral_estimator(self, mu, key):
        lst = self.error_indicators_ls_lhs.get(key)
        assert not(lst is None) and len(lst)>0
        h = self.error_indicators_ls_rhs[key]
        W = np.zeros((lst[0].range.dim, len(lst))) # to do : add dtype
        for i, column_op in enumerate(lst):
            W[:,i] = column_op.assemble(mu).matrix.reshape(-1)
        return W, h

    def _minimize_spectral_estimator(self, mu, key):
        W, h = self._assemble_spectral_estimator(mu, key)
        x, rnorm2, _, _ = np.linalg.lstsq(W, h, rcond=None)
        mu_p_ = dict()
        for k in mu.keys():
            mu_p_[k] = mu[k]
        mu_p_['precond'] = x
        mu_p = Mu(mu_p_)
        rnorm = np.sqrt(rnorm2)
        return mu_p, rnorm
    
    def _assemble_rom_system(self, mu_p):
        rom = self._last_rom
        # use to_matrix to avoid pymor warnings
        reduced_lhs = to_matrix(rom.operator, None, mu_p)
        reduced_rhs = to_matrix(rom.rhs, None, mu_p).reshape(-1)
        return reduced_lhs, reduced_rhs
    
    
    def solve(self, mu, key):
        # Compute the parameter of the preconditioner
        mu_p, _ = self._minimize_spectral_estimator(mu, key)
        # solve the corresponding preconditioned galerkin rom
        u = self._last_rom.solve(mu_p)
        return u
    
    def _add_preconditioner_to_rom(self, P):
        """
        Compute the rom with the new preconditioner P.

        Parameters
        ----------
        P : Operator
            The Operator to add to the preconditioners. If P is the inverse of
            self.fom.operator(mu) for some mu, then it is a new interpolation 
            point for approximating the inverse of the inverse of self.fom.operator.

        Returns
        -------
        rom : StationaryModel
            The reduced order model obtained by adding P to the preconditioner
            part of the last rom.

        """
        Ru = self.product
        RB = self.reduced_basis
        op_lhs_1 = project(P.H @ Ru, self.intermediate_bases['lhs'], RB, Ru)
        op_rhs_1 = project(P.H @ Ru, self.intermediate_bases['rhs'], RB, Ru)
        
        np = len(self.mu_added)
        coefficients = []
        for i in range(np+1):
            coefficients.append(ProjectionParameterFunctional('precond', size=np+1, index=i))
        
        last_rom = self._last_rom
        if last_rom is None:
            op_lhs_2 = project(self.fom.operator, self.intermediate_bases['lhs'], RB)
            op_rhs_2 = project(self.fom.rhs, self.intermediate_bases['rhs'], None)
            operators_lhs_1 = [op_lhs_1.H]
            operators_rhs_1 = [op_rhs_1.H]
        else:
            assert isinstance(last_rom.operator, ConcatenationOperator)
            assert isinstance(last_rom.rhs, ConcatenationOperator)
            operators_lhs_1 = last_rom.operator.operators[0].operators + (op_lhs_1.H,)
            operators_rhs_1 = last_rom.rhs.operators[0].operators + (op_rhs_1.H,)
            
            op_lhs_2 = last_rom.operator.operators[1]
            op_rhs_2 = last_rom.rhs.operators[1]
            
        solver_options = {'inverse': 'to_matrix'}
        reduced_lhs = ConcatenationOperator((LincombOperator(operators_lhs_1, coefficients), op_lhs_2), solver_options)
        reduced_rhs = ConcatenationOperator((LincombOperator(operators_rhs_1, coefficients), op_rhs_2), solver_options)
        
        rom = StationaryModel(reduced_lhs, reduced_rhs)
        return rom
        

    def add_preconditioner(self, P, mu=None):
        """
        Add the operator P within the preconditioner, for ervery spectral 
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
        for key in self.error_indicators_ls_lhs.keys():
            op = self._sketch_operator(P @ self.fom.operator, key)
            self.error_indicators_ls_lhs[key].append(op)
        
        self._last_rom = self._add_preconditioner_to_rom(P)
        self.mu_added.append(mu)
    
