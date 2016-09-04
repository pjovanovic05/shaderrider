"""Operator definitions."""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clmath
from pyopencl import clrandom
from shaderrider import expr
from shaderrider import linalg
from shaderrider import conv
from shaderrider import nnet
from shaderrider import bcast
from shaderrider import clplatf as pl


class Add(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(Add, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            op1 = self.ops[0]._evaluate(valuation, cache)
            op2 = self.ops[1]._evaluate(valuation, cache)
            if op1.shape == op2.shape:
                cache[id(self)] = op1 + op2
            else:
                cache[id(self)], evt = bcast.bcast_add(op1, op2)
                evt.wait()
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


class Log(expr.Expression):
    def __init__(self, op, parents=None):
        super(Log, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            cache[id(self)] = clmath.log(self.ops[0]._evaluate(valuation, cache))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        op = cache[id(self.ops[0])]
        dop = self.ops[0]._fwd_grad(wrt, valuation, cache)
        return 1/op * dop

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        op = cache[id(self.ops[0])]
        self.ops[0]._rev_grad(valuation, 1/op*adjoint, gradient, cache)


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
            x = self.ops[0]._evaluate(valuation, cache)
            cache[id(self)] = 1.0/(1 + clmath.exp(-x))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        ev = cache[id(self)]
        dop = self.ops[0]._fwd_grad(wrt, valuation, cache)
        return ev*(1.0-ev)*dop

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        ev = cache[id(self)]
        # op = cache[id(self.ops[0])]
        adj = adjoint*ev*(1-ev)
        # sig = 1/(1+clmath.exp(-op))
        # adj = adjoint * sig*(1-sig)
        self.ops[0]._rev_grad(valuation, adj, gradient, cache)


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
    def __init__(self, A, B, transA=False, transB=False, parents=None):
        super(Dot, self).__init__(parents)
        self.ops = [A, B]
        self.transA = transA
        self.transB = transB

    def _evaluate(self, valuation, cache):
        q = pl.qs[0]
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = linalg.dot(q, e1(valuation, cache), e2(valuation, cache))
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        q = pl.qs[0]
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        a = linalg.dot(q, self.ops[0]._fwd_grad(wrt, valuation, cache), rhs)
        b = linalg.dot(q, lhs, self.ops[1]._fwd_grad(wrt, valuation, cache))
        return a + b

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        q = pl.qs[0]
        lhs = cache[id(self.ops[0])]
        rhs = cache[id(self.ops[1])]
        adj1 = linalg.dot(q, adjoint, rhs, transB=not self.transB)
        adj2 = linalg.dot(q, lhs, adjoint, transA=not self.transA)
        self.ops[0]._rev_grad(valuation, adj1, gradient, cache)
        self.ops[1]._rev_grad(valuation, adj2, gradient, cache)


class Conv2d(expr.Expression):
    def __init__(self, img, filters, bias, strides=(0, 0), zero_padding=(0, 0),
                 cover_all=False, parents=None):
        super(Conv2d, self).__init__(parents)
        self.ops = [img, filters, bias]
        self.sy, self.sx = strides
        self.ph, self.pw = zero_padding
        self.cover_all = cover_all

    def _evaluate(self, valuation, cache):
        q = pl.qs[0]
        if id(self) not in cache:
            X = self.ops[0]._evaluate(valuation, cache)
            W = self.ops[1]._evaluate(valuation, cache)
            b = self.ops[2]._evaluate(valuation, cache)
            out_c, _, kh, kw = W.shape
            n, c, h, w = X.shape
            out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                          cover_all=self.cover_all)
            out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                          cover_all=self.cover_all)
            y = clarray.empty(q, (n, out_c, out_h, out_w), dtype=X.dtype)
            self.col, ev1 = conv.im2col(q, X, kh, kw, self.sy, self.sx,
                                        self.ph, self.pw, self.cover_all)
            W_mat = W.reshape(out_c, -1)
            ev1.wait()  # TODO asynchronize
            col_mats = self.col.reshape(n, -1, out_h*out_w)
            y_mats = y.reshape(n, out_c, -1)
            for i in xrange(n):
                y_mats[i] = linalg.dot(q, W_mat, col_mats[i])
            if b is not None:
                # y += b[:, None, None]
                _, ev3 = conv.bcast_add(q, y, b, y)
                ev3.wait()  # TODO asynchronize
            cache[id(self)] = y
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        pass

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        q = pl.qs[0]
        X = cache[id(self.ops[0])]
        W = cache[id(self.ops[1])]
        b = cache[id(self.ops[2])]
        gy = adjoint
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = X.shape
        kh, kw = W.shape[2:]

        gW = clarray.zeros_like(W)
        gW_mat = gW.reshape(out_c, c * kh * kw)
        col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
        gy_mats = gy.reshape(n, out_c, out_h * out_w)

        for i in xrange(n):
            gwmat = linalg.dot(q, gy_mats[i], col_mats[i].T)
            gW_mat += gwmat

        W_mat = W.reshape(out_c, -1)
        gcol = clarray.empty_like(self.col)
        gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)
        for i in xrange(n):
            gcol_mats[i] = linalg.dot(q, W_mat.T, gy_mats[i])

        gx, ev = conv.col2im(q, gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        ev.wait()
        gb = None
        if b is not None:
            gb, ev = conv.bgrads_sum(gy)
            ev.wait()
        # TODO bias... sum along multiple axes of gy?
        # TODO set gW, gx and gb in gradient dict
        self.ops[0]._rev_grad(valuation, gx, gradient, cache)
        self.ops[1]._rev_grad(valuation, gW, gradient, cache)
        if gb is not None:
            self.ops[2]._rev_grad(valuation, gb, gradient, cache)


class Dropout(expr.Expression):
    def __init__(self, op, ratio=0.5, parents=None):
        super(Dropout, self).__init__(parents)
        self.ops = [op]
        self.ratio = ratio

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            q = pl.qs[0]
            op = self.ops[0]._evaluate(valuation, cache)
            self.mask = clrandom.rand(q, op.shape, op.dtype) >= self.ratio
            cache[id(self)] = op * self.mask
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        pass

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        self.ops[0]._rev_grad(valuation, adjoint*self.mask, gradient, cache)


class Softmax(expr.Expression):
    pass


class Mean(expr.Expression):
    def __init__(self, op, parents=None):
        super(Mean, self).__init__(parents)
        self.ops = [op]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            op = self.ops[0]._evaluate(valuation, cache)
            cache[id(self)] = clarray.sum(op) / np.float32(op.size)
        return cache[id(self)]

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        op = cache[id(self.ops[0])]
        self.ops[0]._rev_grad(valuation, adjoint / op.size, gradient, cache)


class MeanSquaredErr(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(MeanSquaredErr, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            q = pl.qs[0]
            o1 = self.ops[0]._evaluate(valuation, cache)
            o2 = self.ops[1]._evaluate(valuation, cache)
            self.diff = o1 - o2
            self.diffr = self.diff.ravel()
            dop = linalg.dot(q, self.diffr, self.diffr)
            cache[id(self)] = dop/(2.0*self.diff.size)
        return cache[id(self)]

    def _fwd_grad(self, wrt, valuation, cache):
        pass

    def _rev_grad(self, valuation, adjoint, gradient, cache):
        # coeff = adjoint * 2/self.diff.size
        # df = coeff * self.diff
        # df, ev = bcast.bcast_mul(coeff, self.diff)
        df = bcast.bcast_mul(adjoint, self.diff)
        # ev.wait()

        self.ops[0]._rev_grad(valuation, df, gradient, cache)
        self.ops[1]._rev_grad(valuation, -df, gradient, cache)


class NotEq(expr.Expression):
    def __init__(self, op1, op2, parents=None):
        super(NotEq, self).__init__(parents)
        self.ops = [op1, op2]

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            e1, e2 = self.ops[0]._evaluate, self.ops[1]._evaluate
            cache[id(self)] = e1(valuation, cache) != e2(valuation, cache)
        return cache[id(self)]


class Argmax(expr.Expression):
    def __init__(self, op, axis, parents=None):
        super(Argmax, self).__init__(parents)
        self.ops = [op]
        self.axis = axis

    def _evaluate(self, valuation, cache):
        if id(self) not in cache:
            q = pl.qs[0]
            A = self.ops[0]._evaluate(valuation, cache)
            cache[id(self)], ev = nnet.argmax(q, A, self.axis)
            ev.wait()
        return cache[id(self)]
