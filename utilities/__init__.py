#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:52:35 2023

@author: apasco
"""


from pymor.algorithms.rules import match_generic
from pymor.algorithms.projection import ProjectRules
from pymor.core.exceptions import RuleNotMatchingError
from pymor.vectorarrays.numpy import NumpyVectorSpace


@match_generic(lambda op: op.linear and not op.parametric, 'linear and not parametric')
def action_apply_basis_corrected(self, op):
    range_basis, source_basis = self.range_basis, self.source_basis
    if source_basis is None:
        try:
            V = op.apply_adjoint(range_basis)
        except NotImplementedError as e:
            raise RuleNotMatchingError('apply_adjoint not implemented') from e
        if isinstance(op.source, NumpyVectorSpace):
            from pymor.operators.numpy import NumpyMatrixOperator
            return NumpyMatrixOperator(V.to_numpy().conj(), source_id=op.source.id, name=op.name)
        else:
            from pymor.operators.constructions import VectorArrayOperator
            return VectorArrayOperator(V, adjoint=True, name=op.name)
    else:
        if range_basis is None:
            V = op.apply(source_basis)
            if isinstance(op.range, NumpyVectorSpace):
                from pymor.operators.numpy import NumpyMatrixOperator
                return NumpyMatrixOperator(V.to_numpy().T, range_id=op.range.id, name=op.name)
            else:
                from pymor.operators.constructions import VectorArrayOperator
                return VectorArrayOperator(V, adjoint=False, name=op.name)
        else:
            from pymor.operators.numpy import NumpyMatrixOperator
            return NumpyMatrixOperator(op.apply2(range_basis, source_basis), name=op.name)


ProjectRules.insert_rule(3, action_apply_basis_corrected)