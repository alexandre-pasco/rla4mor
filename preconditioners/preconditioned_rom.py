#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:42:21 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.simplify import expand, contract
from pymor.core.base import BasicObject, ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import IdentityOperator, LincombOperator, ConcatenationOperator, VectorArrayOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorArray

from pymor.algorithms.projection import project


class PreconditionedRom(BasicObject):
    
    
    def __init__(self, fom, reduced_basis, residual_embedding, residual_lhs=None,
                 residual_rhs=None, intermediate_bases=None, product=None, 
                 stable_galerkin=True, rom=None, log_level=20):
        
        self.__auto_init(locals())
        self.logger.setLevel(log_level)
        self.mu_added = []
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
            
        if intermediate_bases is None:
            self.stable_galerkin = False
        
        
    def _add_preconditioner(self, P):
        """
        Compute the rom with the new preconditioner P, where rom.operator and 
        rom.rhs are LincombOperators. This is made by computing (P.H @ RB).H @ Ai for all
        Ai in self.fom.operator.operators. The rom.operators.operators are then
        Pj @ Ai for all Pj in the added preconditioners and Ai in the affine terms
        of self.fom.operators. 
        
        Note that this approach can be numerically unstable and involves a lot 
        of affine terms.

        """
        self.logger.info("Preassembing new ROM")
        Ru = self.product
        RB = self.reduced_basis
        n_p = len(self.mu_added)
        
        func = ProjectionParameterFunctional('precond', size=n_p+1, index=n_p)
        
        self.logger.info("Preassembling galerkin")
        # for Galerkin
        op_gal = project(P, RB, None, product=Ru)
        op_gal = op_gal * func
        
        op_gal_lhs = project(op_gal @ self.fom.operator, None, RB)
        op_gal_rhs = contract(expand(op_gal @ self.fom.rhs))
        
        self.logger.info("Preassembling residual-based estimator")
        # for residual-based estimator
        op_res = project(P, self.residual_embedding.as_source_array(), None)
        op_res_lhs = func * project(op_res @ self.fom.operator, None, RB)
        op_res_rhs = contract(expand(op_res*func @ self.fom.rhs))
        
        self.logger.info("Adding to last ROM")
        # last rom
        last_rom = self.rom
        
        if last_rom is None:
            reduced_lhs = op_gal_lhs
            reduced_rhs = op_gal_rhs
            residual_lhs = op_res_lhs
            residual_rhs = op_res_rhs
            
        else:
            # function to add 1 to the size attribute of the ProjectionParameterFunctional
            # in the coefficients of the preconditioner parameters.
            def update_functional_size(operator):
                last_coefs = operator.coefficients
                new_coefs = []
                
                for coef in last_coefs:

                    if not(coef.parametric):
                        new_coefs.append(coef)
                    elif isinstance(coef, ProjectionParameterFunctional):
                        new_coefs.append(coef.with_(size=n_p+1))
                    # else if it is a product, the precond term must be the first
                    else:
                        new_factors = []
                        for f in coef.factors:
                            if np.isscalar(f):
                                new_factors.append(f)
                            elif f.parametric and f.parameter=='precond':
                                new_factors.append(f.with_(size=n_p+1))
                            else:
                                new_factors.append(f)
                        new_coefs.append(coef.with_(factors=new_factors))
                result = operator.with_(coefficients=new_coefs)
                return result
            
            # for galerkin
            last_reduced_lhs = update_functional_size(last_rom.operator)
            last_reduced_rhs = update_functional_size(last_rom.rhs)
            
            reduced_lhs = last_reduced_lhs + op_gal_lhs
            reduced_rhs = last_reduced_rhs + op_gal_rhs
            
            # for error estimator
            last_residual_lhs = update_functional_size(last_rom.error_estimator.lhs)
            last_residual_rhs = update_functional_size(last_rom.error_estimator.rhs)
            
            residual_lhs = last_residual_lhs + op_res_lhs
            residual_rhs = last_residual_rhs + op_res_rhs
        
        error_estimator = ErrorEstimator(residual_lhs, residual_rhs)
        rom = StationaryModel(reduced_lhs, reduced_rhs, error_estimator=error_estimator)
        return rom
    
    
    def _add_preconditioner_stable(self, P):
        """
        Compute the rom with the new preconditioner P, where rom.operator and 
        rom.rhs are ConcatenationOperator. this is made by computing 
        P @ V, where V is an orthonormal basis of the range of
        self.fom.operators with self.reduced_basis as source basis.
        
        This is a more stable way to compute the affine decomposition of the 
        preconditioned Galerkin, involving less affine terms.

        """
        self.logger.info("Preassembing new ROM with stable affine terms")
        Ru = self.product
        RB = self.reduced_basis
        S = self.residual_embedding
        
        # for both galerkin and residual-based estimator
        op_lhs_1 = project(P @ Ru, None, self.intermediate_bases['lhs'])
        op_rhs_1 = project(P @ Ru, None, self.intermediate_bases['rhs'])
        
        # for galerkin
        op_gal_lhs_1 = project(op_lhs_1, RB, None, Ru)
        op_gal_rhs_1 = project(op_rhs_1, RB, None, Ru)
        
        # for residual-based estimator
        op_res_lhs_1 = contract(expand(S @ op_lhs_1))
        op_res_rhs_1 = contract(expand(S @ op_rhs_1))
        
        
        n_p = len(self.mu_added)
        coefficients = []
        for i in range(n_p+1):
            coefficients.append(ProjectionParameterFunctional('precond', size=n_p+1, index=i))
        
        last_rom = self.rom
        if last_rom is None:
        
            output_func = project(self.fom.output_functional, None, RB)
            
            # for both galerkin and residual-based estimator
            op_lhs_2 = project(self.fom.operator, self.intermediate_bases['lhs'], RB)
            op_rhs_2 = project(self.fom.rhs, self.intermediate_bases['rhs'], None)
            
            # for galerkin
            operators_gal_lhs_1 = [op_gal_lhs_1]
            operators_gal_rhs_1 = [op_gal_rhs_1]
            
            # for residual-based estimator
            operators_res_lhs_1 = [op_res_lhs_1]
            operators_res_rhs_1 = [op_res_rhs_1]
            
            
        else:
            assert isinstance(last_rom.operator, ConcatenationOperator)
            assert isinstance(last_rom.rhs, ConcatenationOperator)
            output_func = last_rom.output_functional
            
            # for both galerkin and residual-based estimator
            op_lhs_2 = last_rom.operator.operators[1]
            op_rhs_2 = last_rom.rhs.operators[1]
            
            # for galerkin
            operators_gal_lhs_1 = last_rom.operator.operators[0].operators + (op_gal_lhs_1,)
            operators_gal_rhs_1 = last_rom.rhs.operators[0].operators + (op_gal_rhs_1,)
            
            # for residual-based estimator
            operators_res_lhs_1 = last_rom.error_estimator.lhs.operators[0].operators + (op_res_lhs_1,)
            operators_res_rhs_1 = last_rom.error_estimator.rhs.operators[0].operators + (op_res_rhs_1,)
            
        # for galerkin
        solver_options = {'inverse': 'to_matrix'}
        reduced_lhs = ConcatenationOperator((LincombOperator(operators_gal_lhs_1, coefficients), op_lhs_2), solver_options)
        reduced_rhs = ConcatenationOperator((LincombOperator(operators_gal_rhs_1, coefficients), op_rhs_2), solver_options)
        
        # for residual-based estimator
        residual_lhs = ConcatenationOperator((LincombOperator(operators_res_lhs_1, coefficients), op_lhs_2))
        residual_rhs = ConcatenationOperator((LincombOperator(operators_res_rhs_1, coefficients), op_rhs_2))

        # build the resulting stationary model
        error_estimator = ErrorEstimator(residual_lhs, residual_rhs)
        rom = StationaryModel(reduced_lhs, reduced_rhs, output_functional=output_func,
                              error_estimator=error_estimator)
        return rom


    def add_preconditioner(self, P, mu=None):
        """
        Compute the rom with the new preconditioner P.

        Parameters
        ----------
        P : Operator
            The Operator to add to the preconditioners. If P is the inverse of
            self.fom.operator(mu) for some mu, then it is a new interpolation 
            point for approximating the inverse of the inverse of self.fom.operator.
        
        mu : Mu
            Parameter value.

        """
        with self.logger.block(f"Adding preconditioner to ROM"):
            if self.stable_galerkin:
                rom = self._add_preconditioner_stable(P)
            else:
                rom = self._add_preconditioner(P)
            self.mu_added.append(mu)
            self.rom = rom



class ErrorEstimator(ImmutableObject):

    def __init__(self, lhs, rhs):
        self.__auto_init(locals())
        
    def estimate_error(self, U, mu, m):
        residual = self.lhs.apply(U, mu) - self.rhs.as_range_array(mu)
        err = residual.norm()
        return err
