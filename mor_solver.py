#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:43:53 2022

@author: pasco
"""

from pymor.algorithms.greedy import WeakGreedySurrogate

import numpy as np
from time import perf_counter
from scipy.sparse import csc_matrix

from affine_operations import *
from other_operators import *
from embeddings import *
from sketched_rom import *


class SketchedSurrogate(WeakGreedySurrogate):
    """
    A class to perform sketched model order reduction.
    
    Attributes
    ----------
    primal_sketch : SketchedRom
    
    online_sketch : SketchedRom
    
    certif_sketch : SketchedRom
    
    projection : str
        The projection method used. Must be 'galerkin', 'minres_ls', 
        'minres_normal'.
    
    """
    
    def __init__(self, primal_sketch, online_sketch, certif_sketch, projection):
        for key, val in locals().items():
            self.__setattr__(key, val)
        
        
    def generate_online_sketch(self, seed=None):
        self.online_sketch.embedding.set_seed(seed)
        self.online_sketch.from_sketch(self.primal_sketch)
    
    
    def add_vectors(self, U, mus):
        if not hasattr(mus, '__len__'):
            mus = [mus]
        self.primal_sketch.add_vectors(U, mus)
        self.certif_sketch.from_sketch(self.primal_sketch)
    
    
    def orthonormalize_basis(self, offset=0):
        Q, R = gram_schmidt(self.primal_sketch.SUr, offset=offset, return_R=True, reiterate=False)
        T = ImplicitInverseOperator(
            csc_matrix(R), source_id=self.primal_sketch.SVr.source.id, 
            range_id=self.primal_sketch.lhs.source.id
            )
        self.primal_sketch.orthonormalize_basis(T=T)
        self.certif_sketch.orthonormalize_basis(T=T)
    

    def extend(self, mu, U=None):
        pass
    
    
    def evaluate(self, mus, return_all_values=False):
        pass
    
    
    def solve_rom(self, mus):
        if self.projection == 'galerkin':
            coefs, times = self.primal_sketch.solve_rom(mus, self.projection)
        else:
            coefs, times = self.online_sketch.solve_rom(mus, self.projection)
        return coefs, times
    
    
    def solve_fom(self, mus):
        U = self.primal_sketch.lhs.source.empty()
        for mu in mus:
            b = self.primal_sketch.rhs.as_range_array(mu)
            u = self.primal_sketch.lhs.apply_inverse(b, mu)
            U.append(u)
        return U
    
    
    def weak_greedy(self, r_max, tol, mu_train=None, U_train=None, n_train=None, 
                    n_per_iter=1, ortho_basis=True, mu_generator=None):
        
        if not mu_train is None:
            mu_train = np.array(mu_train)
        t_deb = perf_counter()
        lhs = self.primal_sketch.lhs
        rhs = self.primal_sketch.rhs
        
        # Initialization
        max_errors_certif = []
        max_errors = []
        max_err = 1.0
        mu_added = []
        eps_online = []  # the upper bound of eps for online embedding
        times = {'exact_solving': [], 'adding_vectors': [], 'ortho_sketch': [],
                 'online_sketch_generation': [], 'offline_assembling': [],
                 'rom_solving': [], 'error_calc': [],
                 'certif_error_calc': [], 'times_mu': []}
        
        r = len(self.primal_sketch.SUr)
        
        while tol < max_err and r < r_max:
            print(f'===== Greedy iteration {r // n_per_iter} adding {n_per_iter} per iter')
            r = r + n_per_iter
            
            if not(mu_generator is None) : 
                mu_train = mu_generator(n_train, seed=r)
            
            # Generating online sketch
            tic = perf_counter()
            self.online_sketch.from_sketch(self.primal_sketch, seed=None)
            times['online_sketch_generation'] = perf_counter() - tic
            
            # Evaluating the errors on the training set with the provisional rom.
            coefs, times_online = self.solve_rom(mu_train)
            tic = perf_counter()
            errors_online, _ = self.online_sketch.residual_norms(coefs, mu_train)
            t_error = perf_counter() - tic
            
            for key, t in times_online.items():
                times[key].append(t)
                print(f"  {str(key)} in {t:.3f}")
            times['error_calc'].append(t_error)
            print(f"  Error calc in {t_error:.3f}")
            
            # Compute the certification quantities
            tic = perf_counter()
            errors_certif, _ = self.certif_sketch.residual_norms(coefs, mu_train)
            times['certif_error_calc'].append(perf_counter() - tic)
            
            eps_certif = self.certif_sketch.embedding.epsilon
            ratio = (errors_online / errors_certif)**2
            w_bar =  max(1 - (1 - eps_certif) * ratio.min(), (1 + eps_certif) * ratio.max() - 1)
            
            eps_online.append(w_bar)
            max_errors_certif.append(np.max(errors_certif))
            print(f"  ub for eps : {w_bar:.3e}")
            print(f"  Max certif error : {max_errors_certif[-1]:.3e}")
            
            # Spot the n_per_iter biggest errors
            indices_added = list(np.argsort(errors_online)[-n_per_iter:])
            mu_add = mu_train[indices_added]
            max_err = errors_online[indices_added[-1]]
            max_errors.append(max_err)
            for mu in mu_add:
                mu_added.append(mu_add)
            print(f"  Maximal error : {max_err:.3e}")
            
            # Computing the snapshots (if U_train is None)
            tic = perf_counter()
            if U_train is None:
                U_add = self.solve_fom(mu_add)
            else:
                U_add = U_train[indices_added]
            t = perf_counter() - tic
            times['exact_solving'].append(t)
            print(f"  Exact solving in {t:.3f}")
            
            # Adding the vectors to the sketch
            tic = perf_counter()
            self.add_vectors(U_add, mu_add)
            t = perf_counter() - tic
            times['adding_vectors'].append(t)
            print(f"  Adding vectors in {t:.3f}")
            
            # Orthonormalize the basis
            tic = perf_counter()
            if ortho_basis:
                self.orthonormalize_basis(offset=r-n_per_iter)
            t = perf_counter() - tic
            times['ortho_sketch'].append(t)
            print(f"  Ortho sketch in {t:.3f}")

        print(f"===== weak greedy algo performed in {perf_counter() - t_deb:.3f}s =====")
        for key, t in times.items():
            print(f"{str(key)} : {np.sum(t):.3f}s")
        print("================================================")
        
        results = dict()
        results['max_errors'] = max_errors
        results['times'] = times
        results['mu_added'] = mu_added
        results['tol'] = tol
        results['max_errors_certif'] = max_errors_certif
        results['eps_online'] = eps_online

        return results


