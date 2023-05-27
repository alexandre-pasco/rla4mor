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
from pymor.operators.constructions import IdentityOperator, LincombOperator, InverseOperator, ConcatenationOperator, VectorArrayOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Mu
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.numpy import NumpyMatrixOperator

from pymor.algorithms.projection import project
from embeddings.embeddings import IdentityEmbedding


class SketchedRom(BasicObject):
    
    
    def __init__(self, fom, embedding=None, product=None, save_rb=True, log_level=20):
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        self.mu_basis = []
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
        if embedding is None:
            self.embedding = IdentityEmbedding(fom.solution_space)
            
        self.srb = self.embedding.range.empty()
        self.rb = fom.solution_space.empty()
        self.srhs = None
        self.slhs = None
        self.output_functional = None
        
    def extend_basis(self, U, mus, check_mus=True):
        if isinstance(mus, Mu):
            mus = [mus]
        if self.save_rb:
            self.rb.append(U)
        if check_mus:
            assert all([mu not in self.mu_basis for mu in mus])
        for mu in mus:
            self.mu_basis.append(mu)
        
        # project the output_functional
        output_proj = project(self.fom.output_functional, None, U)
        if not(isinstance(output_proj, LincombOperator)):
            output_proj = LincombOperator([output_proj], [1])
        if self.output_functional is None:
            self.output_functional = output_proj
        else:
            new_operators = [
                NumpyMatrixOperator(np.hstack((to_matrix(o1), to_matrix(o2)))) 
                for o1,o2 in zip(self.output_functional.operators, output_proj.operators) ] 
            self.output_functional = output_proj.with_(operators=new_operators)
        
        # sketchthe rhs if not done
        s = self.embedding
        if self.srhs is None:
            op = s @ InverseOperator(self.product) @ self.fom.rhs
            self.srhs = contract(expand(op)).with_(range_id=s.range.id)
        
        # sketch the basis
        su = s.apply(U)
        self.srb.append(su)
        
        # sketch the projected operator
        op = s @ InverseOperator(self.product) @ self.fom.operator
        sop = project(op, None, U)
        if self.slhs is None:
            self.slhs = sop
        else:
            # hstack the previous soperator with the newly computed
            new_operators = [
                NumpyMatrixOperator(np.hstack((o1.matrix, o2.matrix)), range_id=o1.range.id) 
                for o1,o2 in zip(self.slhs.operators, sop.operators) ] 
            self.slhs = self.slhs.with_(operators=new_operators)
        
    def orthonormalize_basis(self, T=None, offset=0, return_T=False):
    
        if T is None:
            Q, R = gram_schmidt(self.srb, offset=offset, return_R=True)
            T = np.linalg.pinv(R)
        else:
            Q = self.srb.lincomb(T.T)
        
        if self.save_rb:
            self.rb = self.rb.lincomb(T.T)
            
        self.srb = Q
        slhs = contract(expand(self.slhs @ NumpyMatrixOperator(T)))
        self.slhs = slhs.with_(
            operators=[o.with_(range_id=self.slhs.range.id) for o in slhs.operators])
        
        result = None
        if return_T:
            result = T
        
        return result

    
    def solve_rb(self, mus, projection):
        if not isinstance(mus, (list, np.ndarray)):
            mus = [mus]
        ls = False
        if projection == 'galerkin':
            Ar = project(self.slhs, self.srb, None)
            br = project(self.srhs, self.srb, None)
        elif projection == 'minres':
            Ar = self.slhs
            br = self.srhs
            ls = True
        
        coefs = Ar.source.empty(len(mus))
        for mu in mus:
            # compute the reduced solutions
            u = Ar.apply_inverse(br.as_range_array(mu), mu, least_squares=ls)
            coefs.append(u)
        
        return coefs
    
    
    def from_sketch(self, srom):
        
        s = self.embedding
        self.srb = s.apply(srom.srb)
        self.slhs = contract(expand(s @ srom.slhs))
        self.srhs = contract(expand(s @ srom.srhs))
    
                