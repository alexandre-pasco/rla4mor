#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:43:53 2022

@author: pasco
"""

from pymor.algorithms.greedy import WeakGreedySurrogate




from affine_operations import *
from other_operators import *
from embeddings import *
from sketched_rom import *


class MorSolver:
    """
    A class to perform sketched model order reduction.
    
    Attributes
    ----------
    primal_sketch : SketchedRom
    
    online_sketch : SketchedRom
    
    certif_sketch : SketchedRom
    
    """
    
    def __init__(self, primal_sketch, online_sketch, certif_sketch):
        for key, val in locals().items():
            self.__setattr__(key, val)
        
    def generate_online_sketch(self, seed=None):
        self.online_sketch.embedding.set_seed(seed)
        self.online_sketch.from_sketch(self.primal_sketch)
    
    def add_vectors(self, U):
        self.primal_sketch.add_vectors(U)
        self.certif_sketch.from_sketch(self.primal_sketch)
    
    def orthonormalize_basis(self, offset=0):
        Q, R = gram_schmidt(self.primal_sketch.SUr, offset=offset, return_R=True)
        T = InverseOperator(ImplicitLuOperator(csc_matrix(R)))
        self.primal_sketch.orthonormalize_basis(T=T)
        self.certif_sketch.orthonormalize_basis(T=T)
        

class SketchedSurrogate(WeakGreedySurrogate):
    """
    A class to perform sketched model order reduction.
    
    Attributes
    ----------
    primal_sketch : SketchedRom
    
    online_sketch : SketchedRom
    
    certif_sketch : SketchedRom
    
    """
    
    def __init__(self, primal_sketch, online_sketch, certif_sketch):
        for key, val in locals().items():
            self.__setattr__(key, val)
        
    def generate_online_sketch(self, seed=None):
        self.online_sketch.embedding.set_seed(seed)
        self.online_sketch.from_sketch(self.primal_sketch)
    
    def add_vectors(self, U):
        self.primal_sketch.add_vectors(U)
        self.certif_sketch.from_sketch(self.primal_sketch)
    
    def orthonormalize_basis(self, offset=0):
        Q, R = gram_schmidt(self.primal_sketch.SUr, offset=offset, return_R=True)
        T = InverseOperator(ImplicitLuOperator(csc_matrix(R)))
        self.primal_sketch.orthonormalize_basis(T=T)
        self.certif_sketch.orthonormalize_basis(T=T)


    def extend(self, mu, U=None):
        if isinstance(mu, list):
            mus = mu
        else:
            mus = [mu]

        if U is None:
            lhs = self.sketched_rom.lhs
            rhs = self.sketched_rom.rhs
            U = lhs.source.empty()
            for i in range(len(mus)):
                b = rhs.as_range_array(mus[i])
                U.append(lhs.apply_inverse(b, mus[i]))

        r = len(self.sketched_rom.SUr)
        self.add_vectors(U)
    
    def evaluate(self, mus, return_all_values=False):
        pass
    
    def estimate_errors(self, coefficients):
        
        
        
        
        return errors