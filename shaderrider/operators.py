"""
Operator definitions.

"""

from shaderrider import expr


class Add(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Add, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) + e2(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        return self.ops[0]._fwd_grad(wrt, valuation, cache) + \
               self.ops[1]._fwd_grad(wrt, valuation, cache)

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        self.ops[0]._rev_grad(valuation, adjoint, gradient, cache)
        self.ops[1]._rev_grad(valuation, adjoint, gradient, cache)


class Sub(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Sub, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) - e2(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        return self.ops[0]._fwd_grad(wrt, valuation, cache) - \
               self.ops[1]._fwd_grad(wrt, valuation, cache)

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        self.ops[0]._rev_grad(valuation, adjoint, gradient, cache)
        self.ops[1]._rev_grad(valuation, -adjoint, gradient, cache)


class Mul(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Mul, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        # if they are clarrays
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) * e2(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        return rhs * self.ops[0]._fwd_grad(wrt, valuation, cache) + \
               lhs * self.ops[1]._fwd_grad(wrt, valuation, cache)

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        self.ops[0]._rev_grad(valuation, adjoint * rhs, gradient, cache)
        self.ops[1]._rev_grad(valuation, adjoint * lhs, gradient, cache)


class Div(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Div, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) / e2(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        num = cache[id(self.ops[0])]
        den = cache[id(self.ops[1])]
        dnum = self.ops[0]._fwd_grad(wrt, valuation, cache)
        dden = self.ops[1]._fwd_grad(wrt, valuation, cache)

        # TODO elementwise this maybe?
        return (den*dnum - num*dden) / den**2

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        num = cache[id(self.ops[0])]
        den = cache[id(self.ops[1])]
        self.ops[0]._rev_grad(valuation, adjoint / den, gradient, cache)
        self.ops[1]._rev_grad(valuation, -adjoint * num / den ** 2, gradient, cache)
