#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:52:03 2023

@author: apasco
"""

from pymor.algorithms.rules import match_class
from pymor.algorithms.simplify import ExpandRules, ContractRules
from embeddings.other_operators import CholeskyOperator, InverseLuOperator, LsOperator
from embeddings.embeddings import RandomEmbedding

@match_class(RandomEmbedding, CholeskyOperator, InverseLuOperator, LsOperator)
def action_Nothing(self, op):
    return op


ContractRules.insert_rule(2, action_Nothing)
ExpandRules.insert_rule(2, action_Nothing)
    