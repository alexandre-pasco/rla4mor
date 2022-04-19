#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:43:53 2022

@author: pasco
"""



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
        
        
        
    
    