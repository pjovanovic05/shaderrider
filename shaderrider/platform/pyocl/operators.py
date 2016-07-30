"""
Operators for the PyOpenCL platform.

WRITEME
"""

import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clmath
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators
# from shaderrider.platform.pyocl.aux import clblaswrap



######################## REIMP SA AD ###############################################

class Add(exprgraph.Formula):
    def __init__(self, op1, op2, parents=None):
        super(Add, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) + e2(valuation, cache)
        return cache[id(self)]

    def _forward_grad(self, wrt, valuation, cache):
        return self.ops[0]._forward_grad(wrt, valuation, cache) + \
               self.ops[1]._forward_grad(wrt, valuation, cache)

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        self.ops[0]._reverse_grad(valuation, adjoint, grad, cache)
        self.ops[1]._reverse_grad(valuation, adjoint, grad, cache)


class Sub(exprgraph.Formula):
    def __init__(self, op1, op2, parents=None):
        super(Sub, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) - e2(valuation, cache)
        return cache[id(self)]

    def _forward_grad(self, wrt, valuation, cache):
        return self.ops[0]._forward_grad(wrt, valuation, cache) - \
               self.ops[1]._forward_grad(wrt, valuation, cache)

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        self.ops[0]._reverse_grad(valuation, adjoint, grad, cache)
        self.ops[1]._reverse_grad(valuation, -adjoint, grad, cache)


class Mul(exprgraph.Formula):
    def __init__(self, op1, op2, parents=None):
        super(Mul, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        # if they are clarrays
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) * e2(valuation, cache)
        return cache[id(self)]

    def _forward_grad(self, wrt, valuation, cache):
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        return rhs*self.ops[0]._forward_grad(wrt, valuation, cache) + \
               lhs*self.ops[1]._forward_grad(wrt, valuation, cache)

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        self.ops[0]._reverse_grad(valuation, adjoint*rhs, grad, cache)
        self.ops[1]._reverse_grad(valuation, adjoint*lhs, grad, cache)


class Div(exprgraph.Formula):
    def __init__(self, op1, op2, parents=None):
        super(Div, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) / e2(valuation, cache)
        return cache[id(self)]

    def _forward_grad(self, wrt, valuation, cache):
        num = cache[id(self.ops[0])]
        den = cache[id(self.ops[1])]
        dnum = self.ops[0]._forward_grad(wrt, valuation, cache)
        dden = self.ops[1]._forward_grad(wrt, valuation, cache)

        # TODO elementwise this maybe?
        return (den*dnum - num*dden) / den**2

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        num = cache[id(self.ops[0])]
        den = cache[id(self.ops[1])]
        self.ops[0]._reverse_grad(valuation, adjoint/den, grad, cache)
        self.ops[1]._reverse_grad(valuation, -adjoint*num/den**2, grad, cache)


class Pow(exprgraph.Formula):
    def __init__(self, op1, op2, parents=None):
        super(Pow, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) ** e2(valuation, cache)
        return cache[id(self)]

    def _forward_grad(self, wrt, valuation, cache):
        base = cache[id(self.ops[0])]
        exp = cache[id(self.ops[1])]
        dbase = self.ops[0]._forward_grad(wrt, valuation, cache)
        dexp = self.ops[1]._forward_grad(wrt, valuation, cache)

        # FIXME check if base is zero
        # return base**(exp-1) * (exp*dbase + base*dexp*log(base))
        raise NotImplementedError

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        base = cache[id(self.ops[0])]
        exp = cache[id(self.ops[1])]
        # self.ops[0]._reverse_grad(valuation, adjoint*exp*base**(exp-1), grad, cache)
        # self.ops[1]._reverse_grad(valuation, adjoint*log(base)*base**exp, grad, cache)
        raise NotImplementedError
