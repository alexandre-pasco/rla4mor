#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:25:00 2023

@author: apasco
"""

from pymor.operators.constructions import Operator


class LsOperator(Operator):
    """
    
    Class for parsing `least_squares=True` by default when calling apply_inverse.
    
    
    """
    
    def __init__(self, operator):
        self.__auto_init(locals())
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
    
    def apply(self, U, mu=None, **kwargs):
        return self.operator.apply(U, mu, **kwargs)
    
    def apply_adjoint(self, U, mu=None, **kwargs):
        return self.operator.apply_adjoint(U, mu, **kwargs)
    
    def apply_inverse(self, U, mu=None, **kwargs):
        return self.operator.apply_inverse(U, mu, least_squares=True, **kwargs)
    
    def apply_inverse_adjoint(self, U, mu=None, **kwargs):
        return self.operator.apply_inverse_adjoint(U, mu, least_squares=True, **kwargs)
    
    def assemble(self, mu=None):
        return self.operator.assemble(mu)
