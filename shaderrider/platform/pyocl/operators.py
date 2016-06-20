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


# ARRAY MANIPULATION ##################################################

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation):
        """

        :type valuation: PyOCLValuation
        """
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clarray.reshape(param, self._shape))
        return None

def create_reshape(operands, parameters):              # a, newshape
    raise NotImplementedError


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, param.ravel())
        return None

def create_ravel(operands, parameters):                 # a
    raise NotImplementedError


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clarray.transpose(param, self._axes))
        return None

def create_transpose(operands, parameters):    # a
    raise NotImplementedError


class ConcatenateOP(operators.ConcatenateOP):
    # TODO
    pass

def create_concatenate(operands, parameters):   # a1, a2
    raise NotImplementedError


class StackOP(operators.StackOP):
    pass

def create_stack(operands, parameters):         # xs, axis
    raise NotImplementedError


class SplitOP(operators.SplitOP):
    pass

def create_split(operands, parameters):         # a, indicies
    raise NotImplementedError


class RepeatOP(operators.RepeatOP):
    pass

def create_repeat(operands, parameters):        # a, repeats, axis
    raise NotImplementedError


class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation):
        pass    # TODO what does this do anyway?

def create_dimshuffle(operands, parameters):
    raise NotImplementedError


class DiagonalOP(operators.DiagonalOP):
    def evaluate(self, valuation):
        pass

def create_diagonal(operands, parameters):
    raise NotImplementedError


class TraceOP(operators.TraceOP):
    pass

def create_trace(operands, parameters):
    raise NotImplementedError


# BINARY OPERATIONS ###################################################

class BitwiseAndOP(operators.BitwiseAndOP):
    # TODO
    pass

def create_bitwise_and(operands, parameters):   # x1, x2
    raise NotImplementedError


class BitwiseOrOP(operators.BitwiseOrOP):
    pass

def create_bitwise_or(operands, parameters):            # x1, x2):
    raise NotImplementedError


class BitwiseXorOP(operators.BitwiseXorOP):
    pass

def create_bitwise_xor(operands, parameters):           # x1, x2):
    raise NotImplementedError


class InvertOP(operators.InvertOP):
    pass

def create_invert(operands, parameters):            # x1, x2):
    raise NotImplementedError


class LeftShiftOP(operators.LeftShiftOP):
    pass

def create_left_shift(operands, parameters):            # x1, x2):
    raise NotImplementedError


class RightShiftOP(operators.RightShiftOP):
    pass

def create_right_shift(operands, parameters):           # x1, x2):
    raise NotImplementedError


# INDEXING OPS ########################################################
# TODO indexing in pyopencl apears primitive... maybe clarray needs to return?

class IndexOP(operators.IndexOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, param[self._key])
        return None

def create_index(operands, parameters):
    raise NotImplementedError


# LINEAR ALGEBRA ######################################################

class DotOP(operators.DotOP):
    pass

def create_dot(operands, parameters):           # a, b):
    raise NotImplementedError


class VdotOP(operators.VdotOP):
    pass

def create_vdot(operands, parameters):          # a, b):
    raise NotImplementedError


class InnerOP(operators.InnerOP):
    pass

def create_inner(operands, parameters):         # a, b):
    raise NotImplementedError


class OuterOP(operators.OuterOP):
    pass

def create_outer(operands, parameters):         # a, b):
    raise NotImplementedError


class MatmulOP(operators.MatmulOP):
    pass

def create_matmul(operands, parameters):            # a, b):
    raise NotImplementedError


class EigOP(operators.EigOP):
    pass

def create_eig(operands, parameters):           # a):
    raise NotImplementedError


class EigvalsOP(operators.EigvalsOP):
    pass

def create_eigvals(operands, parameters):           # a):
    raise NotImplementedError


class NormOP(operators.NormOP):
    def evaluate(self, valuation):
        raise NotImplementedError

def create_norm(operands, parameters):
    raise NotImplementedError


# LOGIC OPS ###########################################################


class AllOP(operators.AllOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_all(operands, parameters):           # a):
    raise NotImplementedError


class AnyOP(operators.AnyOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_any(operands, parameters):           # a):
    raise NotImplementedError


class AndOP(operators.AndOP):
    pass

def create_and(operands, parameters):           # a, b):
    raise NotImplementedError


class OrOP(operators.OrOP):
    pass

def create_or(operands, parameters):            # a, b):
    raise NotImplementedError


class NotOP(operators.NotOP):
    pass

def create_not(operands, parameters):           # a):
    raise NotImplementedError


class XorOP(operators.XorOP):
    pass

def create_xor(operands, parameters):           # a, b):
    raise NotImplementedError


class GtOP(operators.GtOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_greater(operands, parameters):           # a, b):
    raise NotImplementedError


class LtOP(operators.LtOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_less(operands, parameters):          # a, b):
    raise NotImplementedError


class GeOP(operators.GeOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_greater_equal(operands, parameters):         # a, b):
    raise NotImplementedError


class LeOP(operators.LeOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_less_equal(operands, parameters):            # a, b):
    raise NotImplementedError


class EqOP(operators.EqOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_equal(operands, parameters):         # a, b):
    raise NotImplementedError


class NeOP(operators.NeOP):
    def evaluate(self, valuation):
        # TODO
        pass

def create_not_equal(operands, parameters):         # a, b):
    raise NotImplementedError


# MATHEMATICAL OPS ####################################################

class SinOP(operators.SinOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.sin(param))
        return None

def create_sin(operands, parameters):           # x):
    raise NotImplementedError


class CosOP(operators.CosOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.cos(param))
        return None

def create_cos(operands, parameters):           # x):
    raise NotImplementedError


class TanOP(operators.TanOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.tan(param))
        return None

def create_tan(operands, parameters):           # x):
    raise NotImplementedError


class ArcsinOP(operators.ArcsinOP):
    pass

def create_arcsin(operands, parameters):            # x):
    raise NotImplementedError


class ArccosOP(operators.ArccosOP):
    pass

def create_arccos(operands, parameters):            # x):
    raise NotImplementedError


class ArctanOP(operators.ArctanOP):
    pass

def create_arctan(operands, parameters):            # x):
    raise NotImplementedError


class SinhOP(operators.SinhOP):
    pass

def create_sinh(operands, parameters):          # x):
    raise NotImplementedError


class CoshOP(operators.CoshOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.cosh(param))
        return None

def create_cosh(operands, parameters):          # x):
    raise NotImplementedError


class TanhOP(operators.TanhOP):
    pass

def create_tanh(operands, parameters):          # x):
    raise NotImplementedError


class ArcsinhOP(operators.ArcsinhOP):
    pass

def create_arcsinh(operands, parameters):           # x):
    raise NotImplementedError


class ArccoshOP(operators.ArccoshOP):
    pass

def create_arccosh(operands, parameters):           # x):
    raise NotImplementedError


class ArctanhOP(operators.ArctanhOP):
    pass

def create_arctanh(operands, parameters):           # x):
    raise NotImplementedError


class RoundOP(operators.RoundOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.round(param))
        return None

def create_round(operands, parameters):         # a, decimal=None, out=None):
    raise NotImplementedError


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.floor(param))
        return None

def create_floor(operands, parameters):         # x, out=None):
    raise NotImplementedError


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.ceil(param))
        return None

def create_ceil(operands, parameters):          # x, out=None):
    raise NotImplementedError


class ProdOP(operators.ProdOP):
    pass

def create_prod(operands, parameters):          # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


class SumOP(operators.SumOP):
    pass

def create_sum(operands, parameters):           # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


class NansumOP(operators.NansumOP):
    pass

def create_nansum(operands, parameters):            # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


class CumprodOP(operators.CumprodOP):
    pass

def create_cumprod(operands, parameters):           # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


class CumsumOP(operators.CumsumOP):
    pass

def create_cumsum(operands, parameters):            # a, axis, dtype, out, keepdims):
    raise NotImplementedError


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.exp(param))
        return None

def create_exp(operands, parameters):           # x):
    raise NotImplementedError


class Exp2OP(operators.Exp2OP):
    pass

def create_exp2(operands, parameters):          # x, out=None):
    raise NotImplementedError


class LogOP(operators.LogOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.log(param))
        return None

def create_log(operands, parameters):           # x, out=None):
    raise NotImplementedError


class Log10OP(operators.Log10OP):
    pass

def create_log10(operands, parameters):         # x, out=None):
    raise NotImplementedError


class Log1pOP(operators.Log1pOP):
    pass

def create_log1p(operands, parameters):         # x, out=None):
    raise NotImplementedError


class AddOP(operators.AddOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a + b)
        return None

def create_add(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


class ReciprocalOP(operators.ReciprocalOP):
    pass

def create_reciprocal(operands, parameters):            # x, out=None):
    raise NotImplementedError


class NegOP(operators.NegOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, -param)
        return None

def create_negative(operands, parameters):          # x, out=None):
    raise NotImplementedError


class MulOP(operators.MulOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a * b)
        return None

def create_multiply(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


class DivOP(operators.DivOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a / b)
        return None

def create_divide(operands, parameters):            # x1, x2, out=None):
    raise NotImplementedError


class PowOP(operators.PowOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a ** b)
        return None

def create_power(operands, parameters):         # x1, x2, out=None):
    raise NotImplementedError


class SubOP(operators.SubOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        valuation.add(self.fid, a - b)
        return None

def create_subtract(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


class TrueDivideOP(operators.TrueDivideOP):
    pass

def create_true_divide(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


class FloorDivideOP(operators.FloorDivideOP):
    pass

def create_floor_divide(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


class ModOP(operators.ModOP):
    pass

def create_mod(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


class AbsOP(operators.AbsOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, abs(param))
        return None

def create_absolute(operands, parameters):
    raise NotImplementedError


class SignOP(operators.SignOP):
    def evaluate(self, valuation):
        raise NotImplementedError

def create_sign(operands, parameters):
    raise NotImplementedError


class SqrOP(operators.SqrOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        # valuation[self.fid] = TODO
        raise NotImplementedError

def create_sqr(operands, parameters):
    raise NotImplementedError


class SqrtOP(operators.SqrtOP):
    def evaluate(self, valuation):
        param = valuation.read(self.operands[0].fid)
        valuation.add(self.fid, clmath.sqrt(param))
        return None

def create_sqrt(operands, parameters):
    pass


class MaximumOP(operators.MaximumOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        out = valuation.read(self.fid) if self.fid in valuation else None
        valuation.add(self.fid, clarray.maximum(a, b, out))                     # TODO maybe set?
        return None

def create_maximum(operands, parameters):
    pass


class MinimumOP(operators.MinimumOP):
    def evaluate(self, valuation):
        a = valuation.read(self.operands[0].fid)
        b = valuation.read(self.operands[1].fid)
        out = valuation.read(self.fid) if self.fid in valuation else None
        valuation.add(self.fid, clarray.minimum(a, b, out))                     # TODO maybe set?
        return None

def create_minimum(operands, parameters):
    raise NotImplementedError


# STATISTICS OPS ######################################################

class MedianOP(operators.MedianOP):
    pass

def create_median(operands, parameters):            # a, axis=None, out=None, overwrite_input=False, keepdims=None):
    raise NotImplementedError


class AverageOP(operators.AverageOP):
    pass

def create_average(operands, parameters):           # a, axis=None, weights=None, returned=None):          # TODO sta je returned?
    raise NotImplementedError


class MeanOP(operators.MeanOP):
    pass

def create_mean(operands, parameters):          # a, axis=None, out=None, keepdims=None):
    raise NotImplementedError


class StdOP(operators.StdOP):
    pass

def create_std(operands, parameters):           # a, axis=None, out=None, ddof=None, keepdims=None):       # TODO sta je ddof?
    raise NotImplementedError


class VarOP(operators.VarOP):
    pass

def create_var(operands, parameters):           # a, axis=None, out=None, ddof=None, keepdims=None):
    raise NotImplementedError


class CorrelateOP(operators.CorrelateOP):
    pass

def create_correlate(operands, parameters):         # a, v, mode=None):
    raise NotImplementedError


class CovOP(operators.CovOP):
    pass

def create_cov(operands, parameters):           # m, y, rowvar, bias, ddof, fweights):                     #TODO ima jos nepoznatih parametara
    raise NotImplementedError


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
