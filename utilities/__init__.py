#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:52:35 2023

@author: apasco
"""


from pymor.algorithms.rules import match_generic, match_class
from pymor.algorithms.projection import ProjectRules
from pymor.algorithms.simplify import ExpandRules
from pymor.core.exceptions import RuleNotMatchingError
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import ConcatenationOperator, LincombOperator

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

@match_class(ConcatenationOperator)
def action_ConcatenationOperator_corrected(self, op):
    op = self.replace_children(op)

    # merge child ConcatenationOperators
    if any(isinstance(o, ConcatenationOperator) for o in op.operators):
        ops = []
        for o in op.operators:
            if isinstance(o, ConcatenationOperator):
                ops.extend(o.operators)
            else:
                ops.append(o)
        op = op.with_(operators=ops)

    # expand concatenations with LincombOperators
    if any(isinstance(o, LincombOperator) for o in op.operators):
        i = next(iter(i for i, o in enumerate(op.operators) if isinstance(o, LincombOperator)))
        left, right = op.operators[:i], op.operators[i+1:]
        ops = [ConcatenationOperator(left + (o,) + right) for o in op.operators[i].operators]
        op = op.operators[i].with_(operators=ops)

        # there can still be LincombOperators within the summands so we recurse ..
        op = self.apply(op)

    return op


ProjectRules.insert_rule(3, action_apply_basis_corrected)
ExpandRules.insert_rule(1, action_ConcatenationOperator_corrected)


