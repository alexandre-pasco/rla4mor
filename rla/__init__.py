#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:52:03 2023

@author: apasco
"""

from pymor.algorithms.rules import match_class
from pymor.algorithms.simplify import ExpandRules, ContractRules
try:
    from rla4mor.utilities.factorization import InverseLuOperator, CholmodOperator, UmfInverseLuOperator
    from rla4mor.utilities.other_operators import LsOperator
    from rla4mor.rla.embeddings import RandomEmbedding
except:
    from utilities.factorization import InverseLuOperator, CholmodOperator, UmfInverseLuOperator
    from utilities.other_operators import LsOperator
    from rla.embeddings import RandomEmbedding

@match_class(RandomEmbedding, InverseLuOperator, CholmodOperator, UmfInverseLuOperator, LsOperator)
def action_Nothing(self, op):
    return op


ContractRules.insert_rule(2, action_Nothing)
ExpandRules.insert_rule(2, action_Nothing)
    