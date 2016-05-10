"""
Defines the optimizer for expression graph

These are all global optimizers (in theano parlance), in that they get full expression tree
and should return optimized expression tree.

Implementation specific optimizers should implement the Optimizer class
"""

import abc
import copy

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import elementwise as ew


class Optimizer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize(self, expr_graph):
        pass


class ElementwiseOpt(Optimizer):
    """
    Fold 'broadcastable' expressions into a single elementwise operator.
    """
    def optimize(self, formula):
        """

        :param formula:
        :return:
        """
        optf = copy.deepcopy(formula)
        front = [optf]
        while len(front):
            f = front.pop()
            curatoms = self._foldSubGraph(f)
            if len(curatoms) > 0:
                new_op = ew.ElementwiseOP(f, curatoms)
                optf.substitute(f, new_op)
                front.extend([ca for ca in curatoms if not isinstance(ca, exprgraph.Atom)])
        return optf

    def _foldSubGraph(self, formula, myatoms=[]):
        if formula.is_broadcastable():
            for op in formula.operands:
                if not op.is_broadcastable():
                    myatoms.append(op)
                else:
                    self._foldSubGraph(op, myatoms)
        else:
            return []
        return myatoms


class SimplifyOpt(Optimizer):
    def optimize(self, expr_graph):
        return expr_graph.simplify()


class ConstantFoldingOpt(Optimizer):
    def optimize(self, expr_graph):
        pass    # TODO
