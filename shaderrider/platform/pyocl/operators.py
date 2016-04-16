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
