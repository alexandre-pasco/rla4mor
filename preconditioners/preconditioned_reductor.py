#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:00:29 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.simplify import expand, contract
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

# class PreconditionedStationaryModel(Model):
    
#     def __init__(self, operator, rhs, output_functional=None, products=None,
#                   error_estimator=None, visualizer=None, name=None):

#         if isinstance(rhs, VectorArray):
#             assert rhs in operator.range
#             rhs = VectorOperator(rhs, name='rhs')
#         output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), operator.source)

#         assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
#         assert output_functional.source == operator.source

#         super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

#         self.__auto_init(locals())
#         self.solution_space = operator.source
#         self.linear = operator.linear and output_functional.linear
#         self.dim_output = output_functional.range.dim


class SketchedPreconditionedReductor(BasicObject):
    """
    Class implementing a sketched preconditionned reductor.
    
    Attibutes
    ---------
    fom : StationaryModel
        The full order model to reduce.
    reduced_basis : VectorArray
        The reduced basis used for the Galerkin projection.
    source_bases : dict o
        Dictionary containing the 
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
    
    
    def _build_ls_lhs(self, mu, key):
        lst = self.error_indicators_ls_lhs.get(key)
        assert not(lst is None) and len(lst)>0
        W = np.zeros((lst[0].range.dim, len(lst))) # to do : add dtype
        for i, column_op in enumerate(lst):
            W[:,i] = column_op.assemble(mu).matrix.reshape(-1)
        return W

    def _solve_ls(self, mu, key):
        W = self._build_ls_lhs(mu, key)
        h = self.error_indicators_ls_rhs[key]
        x, rnorm2, _, _ = np.linalg.lstsq(W, h, rcond=None)
        rnorm = np.sqrt(rnorm2)
        return x, rnorm
    
    def assemble_rom(self, mu, key):
        x, _ = self._solve_ls(mu, key)
        mu_p = Mu({'precond': x})
        rom = self._last_rom
        reduced_lhs_1 = rom.operator.operators[0].assemble(mu_p)
        reduced_lhs_2 = rom.operator.operators[1].assemble(mu)
        reduced_lhs = contract(reduced_lhs_1 @ reduced_lhs_2).matrix
        reduced_rhs_1 = rom.rhs.operators[0].assemble(mu_p)
        reduced_rhs_2 = rom.rhs.operators[1].assemble(mu)
        reduced_rhs = contract(reduced_rhs_1 @ reduced_rhs_2).matrix.reshape(-1)
        return reduced_lhs, reduced_rhs
    
    
    
    def _add_preconditioner_to_rom(self, P):
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
            reduced_lhs = (coefficients[0] * op_lhs_1.H) @ op_lhs_2
            reduced_rhs = (coefficients[0] * op_rhs_1.H) @ op_rhs_2
        else:
            assert isinstance(last_rom.operator, ConcatenationOperator)
            assert isinstance(last_rom.rhs, ConcatenationOperator)
            operators_lhs_1 = last_rom.operator.operators[0].operators + (op_lhs_1.H,)
            operators_rhs_1 = last_rom.rhs.operators[0].operators + (op_rhs_1.H,)
            
            op_lhs_2 = last_rom.operator.operators[1]
            op_rhs_2 = last_rom.rhs.operators[1]
            
            reduced_lhs = LincombOperator(operators_lhs_1, coefficients) @ op_lhs_2
            reduced_rhs = LincombOperator(operators_rhs_1, coefficients) @ op_rhs_2
        
        rom = StationaryModel(reduced_lhs, reduced_rhs)
        return rom
        
        
    # def preassemble_rom(self, P):
    
    def add_preconditioner(self, P, mu=None):
        for key in self.error_indicators_ls_lhs.keys():
            op = self._sketch_operator(P @ self.fom.operator, key)
            self.error_indicators_ls_lhs[key].append(op)
        
        self._last_rom = self._add_preconditioner_to_rom(P)
        self.mu_added.append(mu)
    


# class PreconditionedModel(Model):
    


