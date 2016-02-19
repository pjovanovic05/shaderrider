"""
Defines the optimizer for expression graph

These are all global optimizers (in theano parlance), in that they get full expression tree
and should return optimized expression tree.

Implementation specific optimizers should implement the Optimizer class
"""

import abc


class Optimizer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize(self, exprgraph):
        pass


class ElementwiseOpt(Optimizer):

    def optimize(self, exprgraph):
        pass