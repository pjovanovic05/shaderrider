"""
Operator definitions.

"""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clmath
from shaderrider import expr
from shaderrider import linalg
from shaderrider import conv


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


class Pow(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Pow, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) ** e2(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        base = cache[id(self.ops[0])]
        exp = cache[id(self.ops[1])]
        dbase = self.ops[0]._fwd_grad(wrt, valuation, cache)
        dexp = self.ops[1]._fwd_grad(wrt, valuation, cache)

        # TODO div by zero check (base == 0)
        return base ** (exp-1) * (exp*dbase + base*dexp*clmath.log(base))

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        base = cache[id(self.ops[0])]
        exp = cache[id(self.ops[1])]
        self.ops[0]._rev_grad(valuation, adjoint*exp*base**(exp-1), gradient, cache)
        self.ops[1]._rev_grad(valuation, adjoint*clmath.log(base)*base**exp, gradient, cache)


class Exp(expr.Expression):
    def __init__(self, op, parents=None):
        super(Exp, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            cache[id(self)] = clmath.exp(self.ops[0]._evaluate(valuation, cache))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        ex = cache[id(self.ops[0])]
        dex = self.ops[0]._fwd_grad(wrt, valuation, cache)
        return clmath.exp(ex)*dex

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        ex = cache[id(self.ops[0])]
        self.ops[0]._rev_grad(valuation, adjoint*clmath.exp(ex), gradient, cache)


class Reciprocal(expr.Expression):
    def __init__(self, op, parents=None):
        super(Reciprocal, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            cache[id(self)] = 1.0 / self.ops[0]._evaluate(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        op = cache[id(self.ops[0])]
        dop = self.ops[0]._fwd_grad(wrt, valuation, cache)
        return -1.0 / op**2 * dop

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        op = cache[id(self.ops[0])]
        self.ops[0]._rev_grad(valuation, -adjoint/op**2, gradient, cache)


class Sigmoid(expr.Expression):
    def __init__(self, op, parents=None):
        super(Sigmoid, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            cache[id(self)] = 1.0/(1 + clmath.exp(-self.ops[0]._evaluate(valuation, cache)))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        ev = cache[id(self)]
        dop = self.ops[0]._fwd_grad(wrt, valuation, cache)
        return ev*(1.0-ev)*dop

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        ev = cache[id(self)]
        self.ops[0]._rev_grad(valuation, adjoint*ev*(1-ev), gradient, cache)


class Neg(expr.Expression):
    def __init__(self, op, parents=None):
        super(Neg, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            cache[id(self)] = -self.ops[0]._evaluate(valuation, cache)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        return -self.ops[0]._fwd_grad(wrt, valuation, cache)

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        self.ops[0]._rev_grad(valuation, -adjoint, gradient, cache)


class Max(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Max, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        pass

    def _fwd_grad(self, wrt, valuation, cache):
        pass

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        pass


class Dot(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Dot, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = linalg.dot(q, e1(valuation, cache), e2(valuation, cache))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        return linalg.dot(q, self.ops[0]._fwd_grad(wrt, valuation, cache), rhs) + \
               linalg.dot(q, lhs, self.ops[1]._fwd_grad(wrt, valuation, cache))

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        # lhs = cache[id(self.ops[0])]
        # rhs = cache[id(self.ops[1])]
        # self.ops[0]._rev_grad(valuation, linalg.dot(adjoint, rhs), gradient, cache)
        # self.ops[1]._rev_grad(valuation, linalg.dot(lhs, adjoint), gradient, cache)
        # TODO fale transponovanja ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        pass


class Conv2d(expr.Expression):
    def __init__(self, img, filters, bias, strides=(0,0), zero_padding=(0,0), cover_all=False, parents=None):
        super(Conv2d, self).__init__(parents)
        self.ops = [img, filters, bias]
        self.sy, self.sx = strides
        self.ph, self.pw = zero_padding
        self.cover_all = cover_all

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            X = self.ops[0]._evaluate(valuation, cache)
            W = self.ops[1]._evaluate(valuation, cache)
            b = self.ops[2]._evaluate(valuation, cache)
            out_c, _, kh, kw = W.shape
            n, c, h, w = X.shape
            out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph, cover_all=self.cover_all)
            out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw, cover_all=self.cover_all)
            y = clarray.empty(q, (n, out_c, out_h, out_w), dtype=X.dtype)
            self.col = conv.im2col(q, X, kh, kw, self.sy, self.sx, self.ph, self.pw,
                                   self.cover_all)
            W_mat = W.reshape(out_c, -1)
            col_mats = self.col.reshape(n, -1, out_h*out_w)
            y_mats = y.reshape(n, out_c, -1)
            for i in xrange(n):
                y_mats[i] = linalg.dot(q, W_mat, col_mats[i])
            if b is not None:
                # y += b[:, None, None]
                conv.bcast_add(q, y, b)
            cache[id(self)] = y
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        pass

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        X = cache[id(self.ops[0])]
        W = cache[id(self.ops[1])]
        gy = adjoint
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = X.shape
        kh, kw = W.shape[2:]

        gW = clarray.zeros_like(W)
        gW_mat = gW.reshape(out_c, c * kh * kw)
        col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
        gy_mats = gy.reshape(n, out_c, out_h * out_w)

        for i in xrange(n):
            gW_mat += linalg.dot(q, gy_mats[i], col_mats[i].T)

        W_mat = W.reshape(out_c, -1)
        gcol = clarray.empty_like(self.col)
        gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)
        for i in xrange(n):
            gcol_mats[i] = linalg.dot(q, W_mat.T, gy_mats[i])

        gx = conv.col2im(q, gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        #TODO bias... sum along multiple axes of gy?
        #TODO set gW, gx and gb in gradient dict
        self.ops[0]._rev_grad(valuation, gx, gradient, cache)
        self.ops[1]._rev_grad(valuation, gW, gradient, cache)
