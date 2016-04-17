"""
Operators for the PyOpenCL platform

WRITEME
"""

import pyopencl as cl
from pyopencl import array
from pyopencl import clmath

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators

# TOC
#  - tensor ops
#  - arithmetic ops
#  - elementwise op
#  - scan ops
#  - blas ops
#  - convolution ops

# TENSOR OPS ##########################################################

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = array.reshape(param, self._shape)
        return None

# TODO indexing in pyopencl apears primitive... maybe clarray needs to return?
class IndexOP(operators.IndexOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = param[self._key]
        return None


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = array.transpose(param, self._axes)
        return None


class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation=None):
        # TODO no dimshuffle in PyOpencl
        raise NotImplementedError


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = param.ravel()
        return None


class DiagonalOP(operators.DiagonalOP):
    def evaluate(self, valuation=None):
        raise NotImplementedError


class TraceOP(operators.TraceOP):
    def evaluate(self, valuation=None):
        raise NotImplementedError


class NormOP(operators.NormOP):
    def evaluate(self, valuation=None):
        raise NotImplementedError


# ARITHMETIC OPS ######################################################

class AbsOP(operators.AbsOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = abs(param)
        return None


class NegOP(operators.NegOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = -param
        return None


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation = clmath.exp(param)
        return None


class LogOP(operators.LogOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.log(param)
        return None


class SinOP(operators.SinOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.sin(param)
        return None


class CosOP(operators.CosOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.cos(param)
        return None


class CoshOP(operators.CoshOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.cosh(param)
        return None


class TanOP(operators.TanOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.tan(param)
        return None


class SignOP(operators.SignOP):
    def evaluate(self, valuation=None):
        raise NotImplementedError


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.ceil(param)
        return None


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.floor(param)
        return None


class RoundOP(operators.RoundOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.round(param)
        return None


class SqrOP(operators.SqrOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        # valuation[self.fid] = TODO
        raise NotImplementedError


class SqrtOP(operators.SqrtOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = clmath.sqrt(param)
        return None


class MaximumOP(operators.MaximumOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        out = valuation[self.fid] if self.fid in valuation else None
        valuation[self.fid] = array.maximum(a, b, out)
        return None


class MinimumOP(operators.MinimumOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        out = valuation[self.fid] if self.fid in valuation else None
        valuation[self.fid] = array.minimum(a, b, out)
        return None


class AddOP(operators.AddOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        valuation[self.fid] = a + b
        return None


class SubOP(operators.SubOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        valuation[self.fid] = a - b
        return None


class MulOP(operators.MulOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        valuation[self.fid] = a * b
        return None


class DivOP(operators.DivOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        valuation[self.fid] = a / b
        return None


class PowOP(operators.PowOP):
    def evaluate(self, valuation=None):
        a = valuation[self.operands[0].fid]
        b = valuation[self.operands[1].fid]
        valuation[self.fid] = a ** b
        return None


# ELEMENTWISE OP ######################################################

class ElementwiseOP(operators.ElementwiseOP):
    def __init__(self, expr, ops, ctx=None, device=0, parent=None):
        super(ElementwiseOP, self).__init__(expr, ops, parent)
        self._ctx = ctx
        self._device = device
        self.evaluate = self.generate_eval()

    def generate_eval(self):
        atoms = self._expr.get_atoms()

        def evaluatefn(self, valuation, events=None, device=0):
            pass

        return evaluatefn


def _c_expr(formula):
    if isinstance(formula, exprgraph.Atom):
        if formula.is_array():
            return formula.name + '[i]'
        return formula.name
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
    pass


class ScanOP(operators.ScanOP):
    pass


# BLAS OPS ############################################################

class GemmOP(operators.GemmOP):
    pass


class GemvOP(operators.GemvOP):
    pass


class GerOP(operators.GerOP):
    pass


# CONVOLUTION OPS #####################################################

class ConvOP(operators.ConvOP):
    pass

class PoolOP(operators.PoolOP):
    pass

class DownsampleOP(operators.DownsampleOP):
    pass
