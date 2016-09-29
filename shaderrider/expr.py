"""
Abstract syntax tree for expression graphs that support automatic differentiation.
"""

import weakref
from pyopencl import array as clarray


class Expression(object):
    def __init__(self, parents=None):
        self._parents = weakref.WeakSet(parents)

    def evaluate(self, valuation):
        return self._evaluate(valuation, {})        # TODO mozda cache treba da bude istog tipa kao valuacija?

    def _evaluate(self, valuation, cache):
        raise NotImplementedError

    def fwd_grad(self, wrt, valuation):
        cache = {}
        self._evaluate(valuation, cache)
        return self._fwd_grad(wrt, valuation, cache)

    def _fwd_grad(self, wrt, valuation, cache):
        raise NotImplementedError

    def rev_grad(self, valuation):
        cache = {}
        res = self._evaluate(valuation, cache)
        adjoint = clarray.empty_like(res).fill(1.0)
        grad = {key: 0 for key in valuation}
        self._rev_grad(valuation, adjoint, grad, cache)
        return grad

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        raise NotImplementedError


class Variable(Expression):
    def __init__(self, name, parents=None):
        super(Variable, self).__init__(parents)
        self.name = name
        # TODO dtype? shape?

    def _evaluate(self, valuation, cache):
        cache[id(self)] = valuation[self.name]
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        return wrt[self.name]

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        gradient[self.name] += adjoint


class Constant(Expression):
    def __init__(self, value, parents=None):
        super(Constant, self).__init__(parents)
        self.value = value

    def _evaluate(self, valuation, cache):
        cache[id(self)] = self.value
        return self.value

    def _fwd_grad(self, wrt, valuation, cache):
        return 0

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        # do nothing
        pass
