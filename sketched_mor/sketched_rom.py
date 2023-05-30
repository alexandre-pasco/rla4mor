#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:48:21 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.simplify import expand, contract
from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.base import BasicObject, ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import Operator, ZeroOperator ,IdentityOperator, LincombOperator, InverseOperator, ConcatenationOperator, VectorArrayOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Mu
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.residual import ResidualOperator

from pymor.algorithms.projection import project
from embeddings.embeddings import IdentityEmbedding
from utilities.utilities import concatenate_operators

class SketchedReductor(BasicObject):
    
    def __init__(self, fom, embedding_primal=None, embedding_online=None, 
                 product=None, save_rb=True, orthonormalize=True, 
                 log_level=20, rom_log_level=30):
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        self.mu_basis = []
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
        if embedding_primal is None:
            self.embedding_primal = IdentityEmbedding(fom.solution_space)
        if embedding_online is None:
            self.embedding_online = IdentityEmbedding(embedding_primal.range)
            
        self.srb = self.embedding_primal.range.empty()
        self.rb = fom.solution_space.empty()
        self.output_functional = None
        self.rom = None
        # self.srhs = None
        # self.slhs = None
        self.residual = None
        
        

    def extend_basis(self, U, **kwargs):
        
        if self.save_rb:
            self.rb.append(U)
        
        # project the output_functional
        output_proj = project(self.fom.output_functional, None, U)
        if not(self.output_functional is None):
            output_proj = concatenate_operators((self.output_functional, output_proj), axis=1)
        self.output_functional = output_proj
        
        # sketch the basis
        s = self.embedding_primal
        su = s.apply(U)
        self.srb.append(su)
    
        # sketch the residual
        op = s @ InverseOperator(self.product) @ self.fom.operator
        sop = project(op, None, U)
        
        if self.residual is None:
            srhs = s @ InverseOperator(self.product) @ self.fom.rhs
            srhs = contract(expand(srhs))
            residual_operator = ResidualOperator(sop, srhs)
        
        else:
            slhs = concatenate_operators((self.residual.operator, sop), axis=1)
            residual_operator = self.residual.with_(operator=slhs)
        
        self.residual = residual_operator
        
        # orthonormalise the basis and apply the transformation to the residual
        if self.orthonormalize:
            self.orthonormalize_basis(offset=len(self.srb)-len(U))
        


    def orthonormalize_basis(self, offset=0, T=None, return_T=False):
    
        if T is None:
            Q, R = gram_schmidt(self.srb, offset=offset, return_R=True)
            T = np.linalg.pinv(R)
        else:
            Q = self.srb.lincomb(T.T)
        
        if self.save_rb:
            self.rb = self.rb.lincomb(T.T)
            
        self.srb = Q
        
        slhs = project(self.residual.operator, None, self.residual.source.from_numpy(T.T))
        self.residual = self.residual.with_(operator=slhs)
        
        result = None
        if return_T:
            result = T
        
        return result

    
    def solve_rb(self, mus, projection):
        if not isinstance(mus, (list, np.ndarray)):
            mus = [mus]
        ls = False
        if projection == 'galerkin':
            Ar = project(self.residual.operator, self.srb, None)
            br = project(self.residual.rhs, self.srb, None)
        elif projection == 'minres':
            Ar = self.residual.operator
            br = self.residual.rhs
            ls = True
        
        coefs = Ar.source.empty(len(mus))
        for mu in mus:
            # compute the reduced solutions
            u = Ar.apply_inverse(br.as_range_array(mu), mu, least_squares=ls)
            coefs.append(u)
        
        return coefs
    
    
    def sketch_residual(self, embedding=None):
        
        if embedding is None:
            embedding = self.embedding_online
            
        lhs = contract(expand(embedding @ self.residual.operator))
        rhs = contract(expand(embedding @ self.residual.rhs))
        residual = ResidualOperator(lhs, rhs)
        
        return residual
        
    
    def reduce(self, seed=None, rom_log_levels=30):
        
        self.embedding_online.set_seed(seed)
        error_estimator = ResidualErrorEstimator(self.sketch_residual())
        
        # Build the galerkin system
        reduced_lhs = project(self.residual.operator, self.srb, None)
        reduced_rhs = project(self.residual.rhs, self.srb, None)
        
        
        rom = StationaryModel(reduced_lhs, reduced_rhs, self.output_functional, 
                              error_estimator=error_estimator)
        rom.logger.setLevel(self.rom_log_level)
        return rom
    

class ResidualErrorEstimator(ImmutableObject):
    
    def __init__(self, residual_operator, name=None):
        self.__auto_init(locals())
        self.logger.setLevel(20)
        
    def estimate_error(self, U, mu, m=None):
        residual = self.residual_operator.apply(U, mu)
        error = residual.norm()
        return error

