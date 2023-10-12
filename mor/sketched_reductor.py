#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:48:21 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.simplify import expand, contract
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.base import BasicObject, ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import IdentityOperator, InverseOperator
from pymor.reductors.residual import ResidualOperator, ResidualReductor
from pymor.algorithms.projection import project

try:
    from rla4mor.rla.embeddings import IdentityEmbedding
    from rla4mor.utilities.other_operators import LsOperator
    from rla4mor.utilities.utilities import concatenate_operators
except:
    from rla.embeddings import IdentityEmbedding
    from utilities.other_operators import LsOperator
    from utilities.utilities import concatenate_operators

class SketchedReductor(BasicObject):
    
    def __init__(self, fom, embedding_primal=None, embedding_online=None, 
                 product=None, inverse_product=None, save_rb=True, orthonormalize=True, 
                 projection='galerkin', log_level=20):
        assert projection in ('galerkin', 'minres')
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        self.mu_basis = []
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
        if inverse_product is None:
            self.inverse_product = InverseOperator(product)
        if embedding_primal is None:
            self.embedding_primal = IdentityEmbedding(fom.solution_space)
        if embedding_online is None:
            self.embedding_online = IdentityEmbedding(embedding_primal.range)
            
        self.srb = self.embedding_primal.range.empty()
        self.rb = fom.solution_space.empty()
        self.residual = None
        self.output_functional = None
        self.rom = None
        
        
        

    def extend_basis(self, U, **kwargs):
        
        if self.save_rb:
            self.rb.append(U)
        
        # project the output_functional
        self.logger.info("Projection the output functional")
        output_proj = project(self.fom.output_functional, None, U)
        if not(self.output_functional is None):
            output_proj = concatenate_operators((self.output_functional, output_proj), axis=1)
        self.output_functional = output_proj
        
        # sketch the basis
        self.logger.info("Sketch the basis")
        s = self.embedding_primal
        su = s.apply(U)
        self.srb.append(su)
    
        # sketch the residual
        self.logger.info("Sketch the residual")
        op = s @ self.inverse_product @ self.fom.operator
        sop = project(op, None, U)
        
        if self.residual is None:
            srhs = s @ self.inverse_product @ self.fom.rhs
            srhs = contract(expand(srhs))
            residual_operator = ResidualOperator(sop, srhs)
        
        else:
            slhs = concatenate_operators((self.residual.operator, sop), axis=1)
            residual_operator = self.residual.with_(operator=slhs)
        
        self.residual = residual_operator
        
        # orthonormalize the basis and apply the transformation to the residual
        self.logger.info("Orthonormalize the basis and apply the transformation to the residual")
        if self.orthonormalize:
            self.orthonormalize_basis(offset=len(self.srb)-len(U))
        


    def orthonormalize_basis(self, offset=0, T=None, return_T=False, **kwargs):
    
        self.logger.info("Orthonormalize the sketched basis")
        if T is None:
            Q, R = gram_schmidt(self.srb, offset=offset, return_R=True, **kwargs)
            T = np.linalg.pinv(R)
        else:
            Q = self.srb.lincomb(T.T)
        
        if self.save_rb:
            self.rb = self.rb.lincomb(T.T)
            
        self.srb = Q
        
        V = self.residual.source.from_numpy(T.T)
        
        self.logger.info("Apply orthonormalization matrix to the residual")
        slhs = project(self.residual.operator, None, V)
        self.residual = self.residual.with_(operator=slhs)
        
        self.logger.info("Apply orthonormalization matrix to the output")
        self.output_functional = project(self.output_functional, None, V)
        
        
        result = None
        if return_T:
            result = T
        
        return result
    

    def reduce(self, embedding=None, seed=None, rom_log_level=30):
        
        if len(self.srb) == 0:
            rom = self._reduce_empty()
        
        elif self.projection == 'galerkin':
            if embedding is None:
                embedding = self.embedding_online.with_(_seed=seed)
            rom = self._reduce_galerkin(embedding)
        
        elif self.projection == 'minres':
            if not(hasattr(seed, '__len__')):
                seed = (seed, seed)
            if embedding in (None, (None, None)) :
                embedding = (self.embedding_online.with_(_seed=seed[0]),
                             self.embedding_online.with_(_seed=seed[1]))
            rom = self._reduce_minres(embedding)
        
        rom.logger.setLevel(rom_log_level)
        
        return rom
    
    def _sketch_residual(self, embedding=None):
        
        if embedding is None:
            embedding = self.embedding_online
            
        lhs = contract(expand(embedding @ self.residual.operator))
        rhs = contract(expand(embedding @ self.residual.rhs))
        residual = ResidualOperator(lhs, rhs)
        
        return residual

    def _reduce_galerkin(self, embedding):
        
        # error estimator
        sketched_residual = self._sketch_residual(embedding)
        error_estimator = ResidualErrorEstimator(sketched_residual)
        
        # galerkin system
        reduced_lhs = project(self.residual.operator, self.srb, None)
        reduced_rhs = project(self.residual.rhs, self.srb, None)
    
        # rom
        rom = StationaryModel(reduced_lhs, reduced_rhs, self.output_functional, 
                              error_estimator=error_estimator)
        
        return rom

    def _reduce_minres(self, embedding):
        
        # minres system
        op = self._sketch_residual(embedding[0])
        lhs = LsOperator(op.operator)
        rhs = op.rhs
        
        # error estimator
        sketched_residual = self._sketch_residual(embedding[1])
        error_estimator = ResidualErrorEstimator(sketched_residual)

        # rom
        rom = StationaryModel(lhs, rhs, self.output_functional, 
                              error_estimator=error_estimator)
        
        # TO DO ?: add self.rom = rom ? 
        
        return rom

    def _reduce_empty(self):
        # reduced system
        lhs = project(self.fom.operator, self.rb, self.rb)
        rhs = project(self.fom.rhs, self.rb, None)
        
        # error estimator
        residual = ResidualReductor(
            self.rb, self.fom.operator, self.fom.rhs, 
            product=self.product, riesz_representatives=True
            ).reduce()
        error_estimator = ResidualErrorEstimator(residual)
        
        # output_functional
        output_functional = project(self.fom.output_functional, None, self.rb)
        
        # rom
        rom = StationaryModel(lhs, rhs, output_functional, 
                              error_estimator=error_estimator)

        return rom

class ResidualErrorEstimator(ImmutableObject):
    
    def __init__(self, operator, name=None):
        self.__auto_init(locals())
        self.logger.setLevel(20)
        
    def estimate_error(self, U, mu, m=None):
        residual = self.operator.apply(U, mu)
        error = residual.norm()
        return error

