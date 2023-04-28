#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:00:29 2023

@author: apasco
"""

import numpy as np
from pymor.algorithms.simplify import expand, contract
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.core.defaults import set_defaults
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import IdentityOperator, LincombOperator, ConcatenationOperator, InverseOperator
from pymor.operators.interface import as_array_max_length
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Mu

from pymor.algorithms.projection import project

from embeddings.embeddings import IdentityEmbedding



class PreconditionedReductor(BasicObject):
    """
    Class implementing a sketched preconditionned reductor. Several attributes 
    of this class are dictionaries. The associated keys correspond to the different
    spectral norm estimators (i.e. sketched HS norms).
    
    Attibutes
    ---------
    fom : StationaryModel
        The full order model to reduce.
    reduced_basis : VectorArray
        The reduced basis used for the Galerkin projection.
    source_bases : dict
        Dictionary containing the bases of the source space for the error 
        estimators.
    range_bases : dict
        Dictionary containing the bases of the range space for the error 
        estimators.
    source_embeddings : dict
        Dictionary containing the embeddings of the source space for the error 
        estimators.
    range_embeddings : dict
        Dictionary containing the embeddings of the range space for the error 
        estimators.
    vec_embeddings : dict
        Dictionary containing the embeddings to vectorize the sketched range 
        space for the error estimators.
    intermediate_bases : dict
        Dictionary containing the image of some operators, in order to improve 
        the numerical stability when assembling the Galerkin system. The key 
        'lhs' contains an ortho basis of fom.operator with reduced_basis as source.
        The key 'rhs' contains an ortho basis of fom.rhs.
    product : Operator
        A symetric positive definite operator, encoding the inner product.
    stable_galerkin : bool, optional
        If True, uses the intermediate basis projection to improve stability of
        the preconditioned galerkin system.
    
    hs_estimators_lhs: dict
        Dictionary containing lists of LincombOperator. For each key, 
        the ith element of the list is the affine decomposition of the sketch 
        of Pi @ self.fom.lhs, where Pi is the ith direction in the 
        preconditioners space.
    hs_estimators_rhs: dict
        Dictionary containing a ndarray, which are the sketch of the identity 
        operator for the different keys.
    sketched_source_bases: dict of VectorArray
        
    sketched_range_bases: dict of VectorArray
        

    """
    
    
    def __init__(self, fom, reduced_basis, source_bases, range_bases,
                 source_embeddings, range_embeddings, vec_embeddings,
                 intermediate_bases=None, product=None, stable_galerkin=True):
        
        assert source_bases.keys() == range_bases.keys()
        self.__auto_init(locals())
        self.logger.setLevel(20)
        self.mu_added = []
        self._last_rom = None
        if product is None:
            self.product = IdentityOperator(fom.solution_space)
            
        if intermediate_bases is None:
            self.stable_galerkin = False
        
        # for the HS indicators
        self.sketched_source_bases = dict()
        self.sketched_range_bases = dict()
        for key, V in source_bases.items():
            S = source_embeddings[key]
            if isinstance(S, IdentityEmbedding) or V is None:
                self.sketched_source_bases[key] = V
            else:
                self.sketched_source_bases[key] = V.lincomb(S.get_matrix())
        
        for key, V in range_bases.items():
            S = range_embeddings[key]
            if isinstance(S, IdentityEmbedding) or V is None:
                self.sketched_range_bases[key] = V
            else:
                self.sketched_range_bases[key] = V.lincomb(S.get_matrix())
        
        # Initialize the hs_estimator systems
        self.hs_estimators_lhs = dict()
        self.hs_estimators_rhs = dict()
        for key in source_bases.keys():
            self.hs_estimators_lhs[key] = []
            self.hs_estimators_rhs[key] = self.sketch_operator(
                IdentityOperator(fom.solution_space), key).matrix.reshape(-1)
        
        # change default value to be able to use to_matrix
        if len(reduced_basis) >= as_array_max_length():
            set_defaults({'pymor.operators.interface.as_array_max_length.value':1+len(reduced_basis)})
        
    def sketch_operator(self, operator, key):
        """
        Compute the sketch of a U -> U operator, and return the projected operator, 
        which is a LincombOperator with a source dimension equal to 1. It will
        be used to build the columns of the systems for the HS norm estimations.

        Parameters
        ----------
        operator : Operator
            The operator to sketch.
        key : str
            The dictionary key to select the bases and embeddings to use.

        Returns
        -------
        result : LincombOperator
            The sketched operator, which is actually a linear form.

        """
        self.logger.info(f"sketching {operator.name} {key}")
        Vr = self.sketched_range_bases[key]
        Vs = self.sketched_source_bases[key]
        Ru = self.product
        
        if Vr is None:
            Vr = self.product.apply_inverse(self.range_embeddings[key].as_source_array())
        
        if Vs is None:
            if isinstance(operator, ConcatenationOperator):
                V = operator.operators[0].apply_adjoint(Ru.apply(Vr))
                op = project(operator.operators[1], V, None)
                op = contract(expand(self.source_embeddings[key] @ InverseOperator(Ru) @ op.H)).H
            else:
                op = project(self.source_embeddings[key] @ InverseOperator(Ru) @ operator.H @ Ru, None, Vr).H
                # op = project(operator @ InverseOperator(Ru) @ self.source_embeddings[key].H, Vr, None, Ru)
        else:
            # we want to first compute operator.H @ Vr
            op = project(operator, Vr, None, Ru)
            op = project(op, None, Vs)
        
        result = contract(expand(self.vec_embeddings[key] @ op))
        
        return result
    
 
    def estimate_quasi_optimality(self, mu_p):
        assert 'u_ur' in self.range_bases.keys()
        delta_2 = self._estimate_hs(mu_p, 'u_ur')
        delta_3 = self._compute_spectral(mu_p)
        if delta_3 >= 1:
            self.logger.warning("Quasi optimality bound not defined")
            result = 0
        else:
            result = 1 + delta_2 / (1 - delta_3)
        return result
    
    def _compute_spectral(self, mu_p):
        A, b = self.assemble_rom_system(mu_p)
        _, s, _ = np.linalg.svd(A - np.eye(A.shape[0]))
        snorm = s.max()
        return snorm

    
    def _estimate_hs(self, mu_p, key):
        W, h = self.assemble_hs_estimator(mu_p, key)
        residual = W @ mu_p['precond'] - h
        rnorm = np.linalg.norm(residual)
        return rnorm
    
    def assemble_hs_estimator(self, mu, key):
        """
        Assemble the least-squares system for the corresponding key and parameter
        value. Minizing $y = \min_{x} ||Wx - h||$ gives the preconditioner 
        $P=\Sum_{i=1}^p y_i P_i$ which minimizes the sketched HS norm for the
        corresponding key and mu.^

        Parameters
        ----------
        mu : Mu
            Parameter value.
        key : string
            The key to identify some HS norm estimator.

        Returns
        -------
        W : (k,p) ndarray
            Each column is the sketch of Pi @ self.fom.operator, for Pi in 
            the added precontitioners.
        h : (k,) ndarray
            Sketch of the identity operator.

        """
        lst = self.hs_estimators_lhs.get(key)
        assert not(lst is None) and len(lst)>0
        h = self.hs_estimators_rhs[key]
        W = np.zeros((lst[0].range.dim, len(lst))) # to do : add dtype
        for i, column_op in enumerate(lst):
            W[:,i] = column_op.assemble(mu).matrix.reshape(-1)
        return W, h

    def minimize_hs_estimator(self, mu, key):
        """
        Minimize the sketched HS norm of (P @ self.fom.lhs - I) for the 
        corresponding key.

        Parameters
        ----------
        mu : Mu
            The parameter value.
        key : string
            The key to identify some HS norm estimator which is to minimize.

        Returns
        -------
        mu_p : Mu
            Parameter value, extended with a 'precond' key, which contains
            the coefficients of the preconditioners which minimises the 
            sketched HS norm for the corresponding key.
        rnorm : float
            Sketched HS norm of P @ self.fom.operator - I

        """
        W, h = self.assemble_hs_estimator(mu, key)
        x, rnorm2, _, _ = np.linalg.lstsq(W, h, rcond=None)
        mu_p_ = dict()
        for k in mu.keys():
            mu_p_[k] = mu[k]
        mu_p_['precond'] = x
        mu_p = Mu(mu_p_)
        rnorm = np.sqrt(rnorm2)
        return mu_p, rnorm
    
    def assemble_rom_system(self, mu_p):
        """
        

        Parameters
        ----------
        mu_p : Mu
            Parameter value, extended with a 'precond' key, which contains
            the coefficients of the preconditioners.

        Returns
        -------
        reduced_lhs : (r,r) ndarray
            The preconditioned galerkin lhs.
        reduced_rhs : (r,) ndarray
            The preconditioned galerkin rhs.

        """
        rom = self._last_rom
        # use to_matrix to avoid pymor warnings
        reduced_lhs = to_matrix(rom.operator, None, mu_p)
        reduced_rhs = to_matrix(rom.rhs, None, mu_p).reshape(-1)
        return reduced_lhs, reduced_rhs
    
    
    def solve(self, mu, key):
        """
        Solve the preconditioned galerkin system for parameter value mu,
        where the preconditioner is obtained by minimizing the HS norm 
        associated to 'key'.

        Parameters
        ----------
        mu : Mu
            Parameter value.
        key : string
            The key to identify some HS norm estimator.

        Returns
        -------
        u : NumpyVectorArray
            The coefficients within self.reduced_basis.

        """
        # Compute the parameter of the preconditioner
        mu_p, _ = self.minimize_hs_estimator(mu, key)
        # solve the corresponding preconditioned galerkin rom
        u = self._last_rom.solve(mu_p)
        return u
    
    def _add_preconditioner_to_rom(self, P):
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
        np = len(self.mu_added)
        
        op = project(P, RB, None, product=Ru)
        op = op * ProjectionParameterFunctional('precond', size=np+1, index=np)
        
        op_lhs = project(op @ self.fom.operator, None, RB)
        op_rhs = contract(expand(op @ self.fom.rhs))
        
        last_rom = self._last_rom
        
        if last_rom is None:
            reduced_lhs = op_lhs
            reduced_rhs = op_rhs
            
        else:
            # function to add 1 to the size attribute of the ProjectionParameterFunctional
            # in the coefficients of the preconditioner parameters.
            def update_functional_size(operator):
                last_coefs = operator.coefficients
                new_coefs = []
                
                for coef in last_coefs:
                    # if the functional is not a product
                    if isinstance(coef, ProjectionParameterFunctional):
                        new_coefs.append(coef.with_(size=np+1))
                    # else if it is a product
                    else:
                        new_factors = [coef.factors[0].with_(size=np+1)] + [f for f in coef.factors[1:]]
                        new_coefs.append(coef.with_(factors=new_factors))
                result = operator.with_(coefficients=new_coefs)
                return result
            
            last_reduced_lhs = update_functional_size(self._last_rom.operator)
            last_reduced_rhs = update_functional_size(self._last_rom.rhs)
            
            reduced_lhs = last_reduced_lhs + op_lhs
            reduced_rhs = last_reduced_rhs + op_rhs
            
        rom = StationaryModel(reduced_lhs, reduced_rhs)
        return rom
    
            
    def _add_preconditioner_to_rom_stable(self, P):
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
            operators_lhs_1 = [op_lhs_1.H]
            operators_rhs_1 = [op_rhs_1.H]
        else:
            assert isinstance(last_rom.operator, ConcatenationOperator)
            assert isinstance(last_rom.rhs, ConcatenationOperator)
            operators_lhs_1 = last_rom.operator.operators[0].operators + (op_lhs_1.H,)
            operators_rhs_1 = last_rom.rhs.operators[0].operators + (op_rhs_1.H,)
            
            op_lhs_2 = last_rom.operator.operators[1]
            op_rhs_2 = last_rom.rhs.operators[1]
            
        solver_options = {'inverse': 'to_matrix'}
        reduced_lhs = ConcatenationOperator((LincombOperator(operators_lhs_1, coefficients), op_lhs_2), solver_options)
        reduced_rhs = ConcatenationOperator((LincombOperator(operators_rhs_1, coefficients), op_rhs_2), solver_options)
        
        rom = StationaryModel(reduced_lhs, reduced_rhs)
        return rom


    def add_preconditioner_to_rom(self, P):
        """
        Compute the rom with the new preconditioner P

        Parameters
        ----------
        P : Operator
            The Operator to add to the preconditioners. If P is the inverse of
            self.fom.operator(mu) for some mu, then it is a new interpolation 
            point for approximating the inverse of the inverse of self.fom.operator.

        Returns
        -------
        rom : StationaryModel
            The reduced order model obtained by adding P to the preconditioner
            part of the last rom.

        """
        if self.stable_galerkin:
            result = self._add_preconditioner_to_rom_stable(P)
        else:
            result = self._add_preconditioner_to_rom(P)
        return result


    def add_preconditioner(self, P, mu=None):
        """
        Add the operator P within the preconditioner, for every spectral 
        norm estimators and for the galerkin system.

        Parameters
        ----------
        P : Operator
            The Operator to add to the preconditioners. If P is the inverse of
            self.fom.operator(mu) for some mu, then it is a new interpolation 
            point for approximating the inverse of the inverse of self.fom.operator.
        mu : Mu, optional
            The parameter corresponding to P. The default is None.
            
        Returns
        -------
        None.

        """
        with self.logger.block(f"Adding preconditioner at {mu}"):
            
            for key in self.hs_estimators_lhs.keys():
                op = self.sketch_operator(P @ self.fom.operator, key)
                self.hs_estimators_lhs[key].append(op)
            
            self._last_rom = self.add_preconditioner_to_rom(P)
            self.mu_added.append(mu)
