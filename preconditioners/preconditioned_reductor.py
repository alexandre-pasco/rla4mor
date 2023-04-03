#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:00:29 2023

@author: apasco
"""


from pymor.core.base import BasicObject
from pymor.models.basic import StationaryModel
from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.constructions import IdentityOperator, VectorOperator, ZeroOperator, ConstantOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor.algorithms.image import estimate_image
from pymor.algorithms.projection import project

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
    
    def __init__(self, fom, source_bases, range_bases, intermediate_basis=None, 
                 product=None):
        
        assert source_bases.keys() == range_bases.keys()
        self.__auto_init(locals())
        self._last_rom = None
        self.projected_operators = dict()
        if intermediate_basis is None:
            self.intermediate_basis = dict()
            for key in source_bases.keys():
                V = estimate_image(
                (fom.operator), domain=source_bases['key'], product=product)
                self.intermediate_basis[key] = V
                
                # projected_operator = project(fom.operator, V, source_basis, product=)
                # self.projectied
        
        
    def add_preconditioner(self, P):
        
        
        
    
    