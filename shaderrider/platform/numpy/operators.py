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


# TENSOR OPS ##########################################################

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.reshape(valuation.get(self.operands[0].fid), self._shape))


class IndexOP(operators.IndexOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, valuation.get(self.operands[0].fid)[self._key])


class TransposeOP(operators.TransposeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.transpose(valuation.get(self.operands[0].fid), self._axes))


class DimshuffleOP(operators.DimshuffleOP):
    def evaluate(self, valuation):
        pass    # TODO what does this do anyway?


class RavelOP(operators.RavelOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ravel(valuation.get(self.operands[0].fid)))


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
    def evaluate(self, valuation):
        valuation.add(self.fid, np.absolute(valuation.get(self.operands[0].fid)))


class NegOP(operators.NegOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, -valuation.get(self.operands[0].fid))


class ExpOP(operators.ExpOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.exp(valuation.get(self.operands[0].fid)))


class LogOP(operators.LogOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.log(valuation.get(self.operands[0].fid)))


class SinOP(operators.SinOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sin(valuation.get(self.operands[0].fid)))


class CosOP(operators.CosOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.cos(valuation.get(self.operands[0].fid)))


class CoshOP(operators.CoshOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.cosh(valuation.get(self.operands[0].fid)))


class TanOP(operators.TanOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.tan(valuation.get(self.operands[0].fid)))


class SignOP(operators.SignOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.sign(valuation.get(self.operands[0].fid)))


class CeilOP(operators.CeilOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.ceil(valuation.get(self.operands[0].fid)))


class FloorOP(operators.FloorOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.floor(valuation.get(self.operands[0].fid)))


class RoundOP(operators.RoundOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.round(valuation.get(self.operands[0].fid)))

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


class AddOP(operators.AddOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.add(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class SubOP(operators.SubOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.subtract(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class MulOP(operators.MulOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.multiply(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class DivOP(operators.DivOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.divide(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class PowOP(operators.PowOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.power(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


# COMPARISON OPS ######################################################

class AnyOP(operators.AnyOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.any(valuation.get(self.operands[0].fid)))


class AllOP(operators.AllOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.all(valuation.get(self.operands[0].fid)))


class EqOP(operators.EqOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class GtOP(operators.GtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.greater(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class LtOP(operators.LtOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class GeOP(operators.GeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.greater_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class LeOP(operators.LeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.less_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


class NeOP(operators.NeOP):
    def evaluate(self, valuation):
        valuation.add(self.fid, np.not_equal(valuation.get(self.operands[0].fid), valuation.get(self.operands[1].fid)))


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
