"""
Operators for the PyOpenCL platform

WRITEME
"""

import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clmath
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators
from shaderrider.platform.pyocl.aux import clblaswrap

# TOC
#  - tensor ops
#  - arithmetic ops
#  - comparison ops
#  - elementwise op
#  - scan ops
#  - blas ops
#  - convolution ops






# OPERATOR FACTORIES                TODO move into operator module - each after the op it creates ##########################################################################################

# ARRAY MANIPULATION
def create_reshape(a, newshape):
    pass


def create_ravel(a):
    pass


def create_concatenate(a1, a2):
    pass


def create_stack(xs, axis):
    pass


def create_split(a, indicies):
    pass


def create_repeat(a, repeats, axis):
    pass


# BINARY OPERATIONS

def create_bitwise_and(x1, x2):
    pass


def create_bitwise_or(x1, x2):
    pass


def create_bitwise_xor(x1, x2):
    pass


def create_invert(x1, x2):
    pass


def create_left_shift(x1, x2):
    pass


def create_right_shift(x1, x2):
    pass


# INDEXING OPS
# TODO

# LINEAR ALGEBRA

def create_dot(a, b):
    pass


def create_vdot(a, b):
    pass


def create_inner(a, b):
    pass


def create_outer(a, b):
    pass


def create_matmul(a, b):
    pass


def create_eig(a):
    pass


def create_eigvals(a):
    pass


# LOGIC OPS

def create_all(a):
    pass


def create_any(a):
    pass


def create_and(a, b):
    pass


def create_or(a, b):
    pass


def create_not(a):
    pass


def create_xor(a, b):
    pass


def create_greater(a, b):
    pass


def create_less(a, b):
    pass


def create_greater_equal(a, b):
    pass


def create_less_equal(a, b):
    pass


def create_equal(a, b):
    pass


def create_not_equal(a, b):
    pass


# MATHEMATICAL OPS

def create_sin(x):
    pass


def create_cos(x):
    pass


def create_tan(x):
    pass


def create_arcsin(x):
    pass


def create_arccos(x):
    pass


def create_arctan(x):
    pass


def create_sinh(x):
    pass


def create_cosh(x):
    pass


def create_tanh(x):
    pass


def create_arcsinh(x):
    pass


def create_arccosh(x):
    pass


def create_arctanh(x):
    pass


def create_round(a, decimal=None, out=None):
    pass


def create_floor(x, out=None):
    pass


def create_ceil(x, out=None):
    pass


def create_prod(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_sum(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_nansum(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_cumprod(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_cumsum(a, axis, dtype, out, keepdims):
    pass


def create_exp(x):
    pass


def create_exp2(x, out=None):
    pass


def create_log(x, out=None):
    pass


def create_log10(x, out=None):
    pass


def create_log1p(x, out=None):
    pass


def create_add(x1, x2, out=None):
    pass


def create_reciprocal(x, out=None):
    pass


def create_negative(x, out=None):
    pass


def create_multiply(x1, x2, out=None):
    pass


def create_divide(x1, x2, out=None):
    pass


def create_power(x1, x2, out=None):
    pass


def create_subtract(x1, x2, out=None):
    pass


def create_true_divide(x1, x2, out=None):
    pass


def create_floor_divide(x1, x2, out=None):
    pass


def create_mod(x1, x2, out=None):
    pass


# STATISTICS OPS

def create_median(a, axis=None, out=None, overwrite_input=False, keepdims=None):
    pass


def create_average(a, axis=None, weights=None, returned=None):          # TODO sta je returned?
    pass


def create_mean(a, axis=None, out=None, keepdims=None):
    pass


def create_std(a, axis=None, out=None, ddof=None, keepdims=None):       # TODO sta je ddof?
    pass


def create_var(a, axis=None, out=None, ddof=None, keepdims=None):
    pass


def create_correlate(a, v, mode=None):
    pass


def create_cov(m, y, rowvar, bias, ddof, fweights):                     #TODO ima jos nepoznatih parametara
    pass













# TENSOR OPS ##########################################################

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation):
        """

        :type valuation: PyOCLValuation
        """
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clarray.reshape(param, self._shape))
        return None


# TODO indexing in pyopencl apears primitive... maybe clarray needs to return?
class IndexOP(operators.IndexOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, param[self._key])
        return None


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clarray.transpose(param, self._axes))
        return None


class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation):
        # TODO no dimshuffle in PyOpencl
        raise NotImplementedError


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, param.ravel())
        return None


class DiagonalOP(operators.DiagonalOP):
    def evaluate(self, valuation):
        raise NotImplementedError


class TraceOP(operators.TraceOP):
    def evaluate(self, valuation):
        raise NotImplementedError


class NormOP(operators.NormOP):
    def evaluate(self, valuation):
        raise NotImplementedError


# ARITHMETIC OPS ######################################################

class AbsOP(operators.AbsOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, abs(param))
        return None


class NegOP(operators.NegOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, -param)
        return None


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.exp(param))
        return None


class LogOP(operators.LogOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.log(param))
        return None


class SinOP(operators.SinOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.sin(param))
        return None


class CosOP(operators.CosOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.cos(param))
        return None


class CoshOP(operators.CoshOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.cosh(param))
        return None


class TanOP(operators.TanOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.tan(param))
        return None


class SignOP(operators.SignOP):
    def evaluate(self, valuation):
        raise NotImplementedError


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.ceil(param))
        return None


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.floor(param))
        return None


class RoundOP(operators.RoundOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.round(param))
        return None


class SqrOP(operators.SqrOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        # valuation[self.fid] = TODO
        raise NotImplementedError


class SqrtOP(operators.SqrtOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.sqrt(param))
        return None


class MaximumOP(operators.MaximumOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        out = valuation.read(self.fid) if self.fid in valuation else None
        valuation.add(self.fid, clarray.maximum(a, b, out))                     # TODO maybe set?
        return None


class MinimumOP(operators.MinimumOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        out = valuation.read(self.fid) if self.fid in valuation else None
        valuation.add(self.fid, clarray.minimum(a, b, out))                     # TODO maybe set?
        return None


class AddOP(operators.AddOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a + b)
        return None


class SubOP(operators.SubOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a - b)
        return None


class MulOP(operators.MulOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a * b)
        return None


class DivOP(operators.DivOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a / b)
        return None


class PowOP(operators.PowOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a ** b)
        return None


# COMPARISON OPS ######################################################

class AnyOP(operators.AnyOP):
    def evaluate(self, valuation):
        # TODO
        pass


class AllOP(operators.AllOP):
    def evaluate(self, valuation):
        # TODO
        pass


class EqOP(operators.EqOP):
    def evaluate(self, valuation):
        # TODO
        pass


class GtOP(operators.GtOP):
    def evaluate(self, valuation):
        # TODO
        pass


class LtOP(operators.LtOP):
    def evaluate(self, valuation):
        # TODO
        pass


class GeOP(operators.GeOP):
    def evaluate(self, valuation):
        # TODO
        pass


class LeOP(operators.LeOP):
    def evaluate(self, valuation):
        # TODO
        pass


class NeOP(operators.NeOP):
    def evaluate(self, valuation):
        # TODO
        pass


# ELEMENTWISE OP ######################################################

class ElementwiseOP(operators.ElementwiseOP):
    def __init__(self, expr, ops, ctx=None, device=0, parent=None):              # TODO ops should be expr.get_variables()?
        super(ElementwiseOP, self).__init__(expr, ops, parent)
        self._ctx = ctx
        self._device = device
        self.evaluate = self.generate_eval()

    def generate_eval(self):
        atoms = self._expr.get_variables()
        args = []
        for a in atoms:
            if a.is_array():
                args.append('%s *%s' % (a.dtype, a.fid))
            else:
                args.append('%s *%s' % (a.dtype, a.fid))
        args.append('%s *%s' % (self.dtype, self.fid))
        argstr = ', '.join(args)
        cexpr = _c_expr(self._expr)
        ewk = ElementwiseKernel(self._ctx, argstr, cexpr, name=self.fid + '_knl')

        def evaluatefn(self, valuation, events=None, device=0):
            params = []
            waits = []
            for a in atoms:
                if a.fid in valuation:
                    params.append(valuation[a.fid].data if a.is_array() else valuation[a.fid])
                else:
                    raise ValueError('Valuation missing a parameter: ' + a.fid)
                if a.fid in events:
                    waits.append(events[a.fid])
            if self.fid not in valuation:
                valuation[self.fid] = clarray.empty(self._ctx, self.get_shape(), self.dtype)
            out = valuation[self.fid]
            params.append(out)
            return ewk(wait_for=waits, *params)

        return evaluatefn


def _c_expr(formula):
    if isinstance(formula, exprgraph.Variable):
        if formula.is_array():
            return formula.name + '[i]'
        return formula.name                                             # TODO da li .name ili .fid?
    if isinstance(formula, exprgraph.Constant):
        if formula.is_array():  # TODO check this
            return formula.fid + '[i]'
        return formula.fid
    if isinstance(formula, operators.NegOP):
        return '-(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.ExpOP):
        return 'exp(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.LogOP):
        return 'log(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.SinOP):
        return 'sin(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.CosOP):
        return 'cos(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.TanOP):
        return 'tan(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, operators.AddOP):
        return '(%s + %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.SubOP):
        return '(%s - %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.MulOP):
        return '(%s * %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.DivOP):
        return '(%s / %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.PowOP):
        return 'pow(%s, %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.EqOP):
        return '(%s == %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.GtOP):
        return '(%s > %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.LtOP):
        return '(%s < %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.GeOP):
        return '(%s >= %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.LeOP):
        return '(%s <= %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, operators.NeOP):
        return '(%s != %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))

    # TODO handle blas and other more complex functions which behave as atoms in this context

    raise ValueError('Unable to convert formula to c expression: %s' % formula)


# SCAN OPS ############################################################

class ReduceOP(operators.ReduceOP):
    def generate_eval(self):
        if self._map_expr is None:
            if len(self.operands) != 1:
                raise ValueError            # TODO better error

        args = []
        for op in self.operands:
            if op.is_array():
                args.append('__global %s *%s' % (op.dtype, op.fid))
            else:
                args.append('%s %s' % (op.dtype, op.fid))
        cargs = ', '.join(args)

        reduck = ReductionKernel(self._ctx, self.dtype, neutral=self._neutral,
                                 reduce_expr=self._reduce_expr, map_expr=self._map_expr,
                                 arguments=cargs, name=self.fid+'_rknl')

        def evaluatefn(self, valuation, events=None, device=0):
            params = []
            waits = []
            for op in self.operands:
                if op.fid in valuation:
                    params.append(valuation[op.fid].data if op.is_array() else valuation[op.fid])
                else:
                    raise ValueError('Valuation missing a parameter: ' + op.fid)
                if op.fid in events:
                    waits.append(events[op.fid])
            out, event = reduck(wait_for=waits, *params)
            valuation[self.fid] = out                               # TODO Create an ATOM out of this.
            return event
        return evaluatefn


class ScanOP(operators.ScanOP):
    def generate_eval(self):
        args = []
        for op in self.operands:
            if op.is_array():
                args.append('__global %s *%s' % (op.dtype, op.fid))
            else:
                args.append('%s %s' % (op.dtype, op.fid))
        cargs = ', '.join(args)

        def evaluatefn(self, valuation, events=None, device=0):
            params = []
            waits = []
            for op in self.operands:
                if op.fid in valuation:
                    params.append(valuation[op.fid].data if op.is_array() else valuation[op.fid])
                else:
                    raise ValueError('Valuation missing a parameter: ' + op.fid)
                if op.fid in events:
                    waits.append(events[op.fid])
            pass
        return evaluatefn


# BLAS OPS ############################################################

class GemmOP(operators.GemmOP):
    def evaluate(self, valuation, events=None, device=0):
        queue = self._ctx.queue[device]         # TODO oklen kontekst? iz configa?
        # A, B, C, alpha, beta, transA, transB
        A = valuation[self.operands[0].fid]
        B = valuation[self.operands[1].fid]
        C = valuation[self.operands[2].fid]
        alpha = valuation[self.operands[3].fid]
        beta = valuation[self.operands[4].fid]
        transA = valuation[self.operands[5].fid]
        transB = valuation[self.operands[6].fid]

        waits = [events[op.fid] for op in operators if op.fid in events]

        # TODO blas setup and teardown in global context!!!
        return clblaswrap.gemm(queue, A, B, C, transA, transB, alpha, beta, wait_for=waits)


class GemvOP(operators.GemvOP):
    def evaluate(self, valuation, events=None, device=0):
        queue = None    # TODO
        # A, X, Y, alpha, beta, transA
        A = valuation[self.operands[0].fid]
        X = valuation[self.operands[1].fid]
        Y = valuation[self.operands[2].fid]
        alpha = valuation[self.operands[3].fid]
        beta = valuation[self.operands[4].fid]
        transA = valuation[self.operands[5].fid]

        waits = [events[op.fid] for op in operators if op.fid in events]
        # TODO


class GerOP(operators.GerOP):
    def evaluate(self, valuation, evalute=None, device=0):
        pass


# CONVOLUTION OPS #####################################################

class ConvOP(operators.ConvOP):
    # see https://www.cs.umd.edu/~djacobs/CMSC426/Convolution.pdf
    # see http://arxiv.org/pdf/1509.09308v2.pdf
    # TODO za sada samo obicna konvolucija, posle cu se baviti Winogradom.
    # TODO check the parameter shapes and generate either 2d or 3d convolution eval.
    def generate_eval(self):
        pass


class PoolOP(operators.PoolOP):
    def generate_eval(self):
        pass


class DownsampleOP(operators.DownsampleOP):
    def generate_eval(self):
        pass
