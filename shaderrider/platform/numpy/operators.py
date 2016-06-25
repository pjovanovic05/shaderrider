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


# ARRAY MANIPULATION ##################################################

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.reshape(valuation.get(self.operands[0].fid), self._shape))


def create_reshape(operands, parameters):           # a, newshape):
    raise NotImplementedError


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ravel(valuation.get(self.operands[0].fid)))


def create_ravel(operands, parameters):         # a):
    raise NotImplementedError


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.transpose(valuation.get(self.operands[0].fid), self._axes))

def create_transpose(operands, parameters):         # a):
    raise NotImplementedError


class ConcatenateOP(operators.ConcatenateOP):
    pass

def create_concatenate(operands, parameters):           # a1, a2):
    raise NotImplementedError


class StackOP(operators.StackOP):
    pass

def create_stack(operands, parameters):         # xs, axis):
    raise NotImplementedError


class SplitOP(operators.SplitOP):
    pass

def create_split(operands, parameters):         # a, indicies):
    raise NotImplementedError


class RepeatOP(operators.RepeatOP):
    pass

def create_repeat(operands, parameters):            # a, repeats, axis):
    raise NotImplementedError


class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation):
        pass    # TODO what does this do anyway?


class DiagonalOP(operators.DiagonalOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.diagonal(valuation.get(self.operands[0].fid)))

def create_diagonal(operands, parameters):
    raise NotImplementedError


class TraceOP(operators.TraceOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.trace(valuation.get(self.operands[0].fid)))

def create_trace(operands, parameters):
    raise NotImplementedError


# BINARY OPERATIONS ###################################################

class BitwiseAndOP(operators.BitwiseAndOP):
    pass

def create_bitwise_and(operands, parameters):           # x1, x2):
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
# TODO

class IndexOP(operators.IndexOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, valuation.get(self.operands[0].fid)[self._key])

def create_index(operands, parameters):         # a, key):
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
        pass    # TODO is this like numpy.linalg.norm?

def create_norm(operands, parameters):
    raise NotImplementedError


# LOGIC OPS ###########################################################

class AllOP(operators.AllOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.all(valuation.get(self.operands[0].fid)))

def create_all(operands, parameters):           # a):
    raise NotImplementedError


class AnyOP(operators.AnyOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.any(valuation.get(self.operands[0].fid)))

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
        valuation.add(self.fid, np.greater(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_greater(operands, parameters):           # a, b):
    raise NotImplementedError


class LtOP(operators.LtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_less(operands, parameters):          # a, b):
    raise NotImplementedError


class GeOP(operators.GeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.greater_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_greater_equal(operands, parameters):         # a, b):
    raise NotImplementedError


class LeOP(operators.LeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_less_equal(operands, parameters):            # a, b):
    raise NotImplementedError


class EqOP(operators.EqOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_equal(operands, parameters):         # a, b):
    raise NotImplementedError


class NeOP(operators.NeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.not_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_not_equal(operands, parameters):         # a, b):
    raise NotImplementedError


# MATHEMATICAL OPS ####################################################

class SinOP(operators.SinOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sin(valuation.get(self.operands[0].fid)))

def create_sin(operands, parameters):           # x):
    raise NotImplementedError


class CosOP(operators.CosOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.cos(valuation.get(self.operands[0].fid)))

def create_cos(operands, parameters):           # x):
    raise NotImplementedError


class TanOP(operators.TanOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.tan(valuation.get(self.operands[0].fid)))

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
        valuation.add(self.fid, np.cosh(valuation.get(self.operands[0].fid)))

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
        valuation.add(self.fid, np.round(valuation.get(self.operands[0].fid)))

def create_round(operands, parameters):         # a, decimal=None, out=None):
    raise NotImplementedError


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.floor(valuation.get(self.operands[0].fid)))

def create_floor(operands, parameters):         # x, out=None):
    raise NotImplementedError


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ceil(valuation.get(self.operands[0].fid)))

def create_ceil(operands, parameters):          # x, out=None):
    raise NotImplementedError


def create_prod(operands, parameters):          # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_sum(operands, parameters):           # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_nansum(operands, parameters):            # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_cumprod(operands, parameters):           # a, axis=None, dtype=None, out=None, keepdims=None):
    raise NotImplementedError


def create_cumsum(operands, parameters):            # a, axis, dtype, out, keepdims):
    raise NotImplementedError


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.exp(valuation.get(self.operands[0].fid)))

def create_exp(operands, parameters):           # x):
    raise NotImplementedError


def create_exp2(operands, parameters):          # x, out=None):
    raise NotImplementedError


class LogOP(operators.LogOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.log(valuation.get(self.operands[0].fid)))

def create_log(operands, parameters):           # x, out=None):
    raise NotImplementedError


def create_log10(operands, parameters):         # x, out=None):
    raise NotImplementedError


def create_log1p(operands, parameters):         # x, out=None):
    raise NotImplementedError


class AddOP(operators.AddOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.add(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_add(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


def create_reciprocal(operands, parameters):            # x, out=None):
    raise NotImplementedError


class NegOP(operators.NegOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, -valuation.get(self.operands[0].fid))

def create_negative(operands, parameters):          # x, out=None):
    raise NotImplementedError


class MulOP(operators.MulOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.multiply(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_multiply(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


class DivOP(operators.DivOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.divide(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_divide(operands, parameters):            # x1, x2, out=None):
    raise NotImplementedError


class PowOP(operators.PowOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.power(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_power(operands, parameters):         # x1, x2, out=None):
    raise NotImplementedError


class SubOP(operators.SubOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.subtract(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_subtract(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


def create_true_divide(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


def create_floor_divide(operands, parameters):          # x1, x2, out=None):
    raise NotImplementedError


def create_mod(operands, parameters):           # x1, x2, out=None):
    raise NotImplementedError


class AbsOP(operators.AbsOP):
    def evaluate(self, valuation):                                                  # TODO is this faster?
        if self.fid not in valuation:
            valuation.add(self.fid, np.absolute(valuation.get(self.operands[0].fid)))
        else:
            outvar = valuation.get(self.fid)
            np.absolute(valuation.get(self.operands[0].fid), out=outvar)

def create_abs(operands, parameters):
    raise NotImplementedError


class SignOP(operators.SignOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sign(valuation.get(self.operands[0].fid)))

def create_sign(operands, parameters):
    raise NotImplementedError


class SqrOP(operators.SqrOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.square(valuation.get(self.operands[0].fid)))

def create_sqr(operands, parameters):
    raise NotImplementedError


class SqrtOP(operators.SqrtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sqrt(valuation.get(self.operands[0].fid)))

def create_sqrt(operands, parameters):
    raise NotImplementedError


class MaximumOP(operators.MaximumOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.maximum(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_maximum(operands, parameters):
    raise NotImplementedError


class MinimumOP(operators.MinimumOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.minimum(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))

def create_minimum(operands, parameters):
    raise NotImplementedError


# STATISTICS OPS ######################################################

def create_median(operands, parameters):            # a, axis=None, out=None, overwrite_input=False, keepdims=None):
    raise NotImplementedError


def create_average(operands, parameters):           # a, axis=None, weights=None, returned=None):          # TODO sta je returned?
    raise NotImplementedError


def create_mean(operands, parameters):          # a, axis=None, out=None, keepdims=None):
    raise NotImplementedError


def create_std(operands, parameters):           # a, axis=None, out=None, ddof=None, keepdims=None):       # TODO sta je ddof?
    raise NotImplementedError


def create_var(operands, parameters):           # a, axis=None, out=None, ddof=None, keepdims=None):
    raise NotImplementedError


def create_correlate(operands, parameters):         # a, v, mode=None):
    raise NotImplementedError


def create_cov(operands, parameters):           # m, y, rowvar, bias, ddof, fweights):                     #TODO ima jos nepoznatih parametara
    raise NotImplementedError


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
