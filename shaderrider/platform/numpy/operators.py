"""
Operators for the numpy platform

WRITEME
"""

import numpy as np

from shaderrider.symbolic import exprgraph, operators


# TOC
#  - tensor ops
#  - arithmetic ops
#  - comparison ops
#  - elementwise op
#  - scan ops
#  - blas ops
#  - convolution ops


# ARRAY MANIPULATION

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.reshape(valuation.get(self.operands[0].fid), self._shape))

def create_reshape(a, newshape):
    raise NotImplementedError


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ravel(valuation.get(self.operands[0].fid)))

def create_ravel(a):
    raise NotImplementedError


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.transpose(valuation.get(self.operands[0].fid), self._axes))

def create_transpose(a):
    raise NotImplementedError


class ConcatenateOP(operators.ConcatenateOP):
    pass

def create_concatenate(a1, a2):
    raise NotImplementedError


class StackOP(operators.StackOP):
    pass

def create_stack(xs, axis):
    raise NotImplementedError


class SplitOP(operators.SplitOP):
    pass

def create_split(a, indicies):
    raise NotImplementedError


class RepeatOP(operators.RepeatOP):
    pass

def create_repeat(a, repeats, axis):
    raise NotImplementedError


# BINARY OPERATIONS

class BitwiseAndOP(operators.BitwiseAndOP):
    pass

def create_bitwise_and(x1, x2):
    raise NotImplementedError


class BitwiseOrOP(operators.BitwiseOrOP):
    pass

def create_bitwise_or(x1, x2):
    raise NotImplementedError


class BitwiseXorOP(operators.BitwiseXorOP):
    pass

def create_bitwise_xor(x1, x2):
    raise NotImplementedError


class InvertOP(operators.InvertOP):
    pass

def create_invert(x1, x2):
    raise NotImplementedError


class LeftShiftOP(operators.LeftShiftOP):
    pass

def create_left_shift(x1, x2):
    raise NotImplementedError


class RightShiftOP(operators.RightShiftOP):
    pass

def create_right_shift(x1, x2):
    raise NotImplementedError


# INDEXING OPS
# TODO

class IndexOP(operators.IndexOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, valuation.get(self.operands[0].fid)[self._key])

def create_index(a, key):
    raise NotImplementedError

# LINEAR ALGEBRA

class DotOP(operators.DotOP):
    pass

def create_dot(a, b):
    raise NotImplementedError


class VdotOP(operators.VdotOP):
    pass

def create_vdot(a, b):
    raise NotImplementedError


class InnerOP(operators.InnerOP):
    pass

def create_inner(a, b):
    raise NotImplementedError


class OuterOP(operators.OuterOP):
    pass

def create_outer(a, b):
    raise NotImplementedError


class MatmulOP(operators.MatmulOP):
    pass

def create_matmul(a, b):
    raise NotImplementedError


class EigOP(operators.EigOP):
    pass

def create_eig(a):
    raise NotImplementedError


class EigvalsOP(operators.EigvalsOP):
    pass

def create_eigvals(a):
    raise NotImplementedError


# LOGIC OPS

class AllOP(operators.AllOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.all(valuation.get(self.operands[0].fid)))

def create_all(a):
    raise NotImplementedError


class AnyOP(operators.AnyOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.any(valuation.get(self.operands[0].fid)))

def create_any(a):
    raise NotImplementedError


class AndOP(operators.AndOP):
    pass

def create_and(a, b):
    raise NotImplementedError


class OrOP(operators.OrOP):
    pass

def create_or(a, b):
    raise NotImplementedError


class NotOP(operators.NotOP):
    pass

def create_not(a):
    raise NotImplementedError


class XorOP(operators.XorOP):
    pass

def create_xor(a, b):
    raise NotImplementedError


class GtOP(operators.GtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.greater(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_greater(a, b):
    raise NotImplementedError


class LtOP(operators.LtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_less(a, b):
    raise NotImplementedError


class GeOP(operators.GeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.greater_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_greater_equal(a, b):
    raise NotImplementedError


class LeOP(operators.LeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_less_equal(a, b):
    raise NotImplementedError


class EqOP(operators.EqOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_equal(a, b):
    raise NotImplementedError


class NeOP(operators.NeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.not_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_not_equal(a, b):
    raise NotImplementedError


# MATHEMATICAL OPS

class SinOP(operators.SinOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sin(valuation.get(self.operands[0].fid)))

def create_sin(x):
    raise NotImplementedError


class CosOP(operators.CosOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.cos(valuation.get(self.operands[0].fid)))

def create_cos(x):
    raise NotImplementedError


class TanOP(operators.TanOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.tan(valuation.get(self.operands[0].fid)))

def create_tan(x):
    raise NotImplementedError


class ArcsinOP(operators.ArcsinOP):
    pass

def create_arcsin(x):
    raise NotImplementedError


class ArccosOP(operators.ArccosOP):
    pass

def create_arccos(x):
    raise NotImplementedError


class ArctanOP(operators.ArctanOP):
    pass

def create_arctan(x):
    raise NotImplementedError


class SinhOP(operators.SinhOP):
    pass

def create_sinh(x):
    raise NotImplementedError


class CoshOP(operators.CoshOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.cosh(valuation.get(self.operands[0].fid)))

def create_cosh(x):
    raise NotImplementedError


class TanhOP(operators.TanhOP):
    pass

def create_tanh(x):
    raise NotImplementedError


class ArcsinOP(operators.ArcsinOP):
    pass

def create_arcsinh(x):
    raise NotImplementedError


class ArccoshOP(operators.ArccoshOP):
    pass

def create_arccosh(x):
    raise NotImplementedError


class ArctanhOP(operators.ArctanhOP):
    pass

def create_arctanh(x):
    raise NotImplementedError


class RoundOP(operators.RoundOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.round(valuation.get(self.operands[0].fid)))

def create_round(a, decimal=None, out=None):
    raise NotImplementedError


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.floor(valuation.get(self.operands[0].fid)))

def create_floor(x, out=None):
    raise NotImplementedError


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ceil(valuation.get(self.operands[0].fid)))

def create_ceil(x, out=None):
    raise NotImplementedError


def create_prod(a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_sum(a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_nansum(a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_cumprod(a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_cumsum(a, axis, dtype, out, keepdims):
    raise NotImplementedError


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.exp(valuation.get(self.operands[0].fid)))

def create_exp(x):
    raise NotImplementedError


def create_exp2(x, out=None):
    raise NotImplementedError


class LogOP(operators.LogOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.log(valuation.get(self.operands[0].fid)))

def create_log(x, out=None):
    raise NotImplementedError


def create_log10(x, out=None):
    raise NotImplementedError


def create_log1p(x, out=None):
    raise NotImplementedError


class AddOP(operators.AddOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.add(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_add(x1, x2, out=None):
    raise NotImplementedError


def create_reciprocal(x, out=None):
    raise NotImplementedError


class NegOP(operators.NegOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, -valuation.get(self.operands[0].fid))

def create_negative(x, out=None):
    raise NotImplementedError


class MulOP(operators.MulOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.multiply(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_multiply(x1, x2, out=None):
    raise NotImplementedError


class DivOP(operators.DivOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.divide(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_divide(x1, x2, out=None):
    raise NotImplementedError


class PowOP(operators.PowOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.power(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_power(x1, x2, out=None):
    raise NotImplementedError


class SubOP(operators.SubOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.subtract(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_subtract(x1, x2, out=None):
    raise NotImplementedError


def create_true_divide(x1, x2, out=None):
    raise NotImplementedError


def create_floor_divide(x1, x2, out=None):
    raise NotImplementedError


def create_mod(x1, x2, out=None):
    raise NotImplementedError


# STATISTICS OPS

def create_median(a, axis=None, out=None, overwrite_input=False, keepdims=None):
    raise NotImplementedError


def create_average(a, axis=None, weights=None, returned=None):          # TODO sta je returned?
    raise NotImplementedError


def create_mean(a, axis=None, out=None, keepdims=None):
    raise NotImplementedError


def create_std(a, axis=None, out=None, ddof=None, keepdims=None):       # TODO sta je ddof?
    raise NotImplementedError


def create_var(a, axis=None, out=None, ddof=None, keepdims=None):
    raise NotImplementedError


def create_correlate(a, v, mode=None):
    raise NotImplementedError


def create_cov(m, y, rowvar, bias, ddof, fweights):                     #TODO ima jos nepoznatih parametara
    raise NotImplementedError








# TENSOR OPS ##########################################################




class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation):
        pass    # TODO what does this do anyway?


class DiagonalOP(operators.DiagonalOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.diagonal(valuation.get(self.operands[0].fid)))


class TraceOP(operators.TraceOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.trace(valuation.get(self.operands[0].fid)))


class NormOP(operators.NormOP):
    def evaluate(self, valuation):
        pass    # TODO is this like numpy.linalg.norm?


# ARITHMETIC OPS ######################################################

class AbsOP(operators.AbsOP):
    def evaluate(self, valuation):                                                  # TODO is this faster?
        if self.fid not in valuation:
            valuation.add(self.fid, np.absolute(valuation.get(self.operands[0].fid)))
        else:
            outvar = valuation.get(self.fid)
            np.absolute(valuation.get(self.operands[0].fid), out=outvar)


















class SignOP(operators.SignOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sign(valuation.get(self.operands[0].fid)))







class SqrOP(operators.SqrOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.square(valuation.get(self.operands[0].fid)))


class SqrtOP(operators.SqrtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sqrt(valuation.get(self.operands[0].fid)))


class MaximumOP(operators.MaximumOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.maximum(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class MinimumOP(operators.MinimumOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.minimum(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))



# COMPARISON OPS ######################################################







# ELEMENTWISE OP ######################################################

class ElementwiseOP(operators.ElementwiseOP):
    def __init__(self, expr, ctx=None, device=0, parent=None):
        pass

    def generate_eval(self):
        pass


# SCAN OPS ############################################################

class ReduceOP(operators.ReduceOP):
    def generate_eval(self):
        pass


class ScanOP(operators.ScanOP):
    def generate_eval(self):
        pass


# BLAS OPS ############################################################

class GemmOP(operators.GemmOP):
    def evaluate(self, valuation):
        pass


class GemvOP(operators.GemvOP):
    def evaluate(self, valuation):
        pass


class GerOP(operators.GerOP):
    def evaluate(self, valuation):
        pass


# Convolution OPS #####################################################

class ConvOP(operators.ConvOP):
    def generate_eval(self):
        pass


class PoolOP(operators.PoolOP):
    def generate_eval(self):
        pass


class DownsampleOP(operators.DownsampleOP):
    def generate_eval(self):
        pass
