"""
Operators for the PyOpenCL platform

WRITEME
"""

import pyopencl as cl
from pyopencl import array

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators

# TOC
#  - tensor ops
#  - arithmetic ops
#  - elementwise op
#  - scan ops
#  - blas ops
#  - convolution ops

class ReshapeOP(operators.ReshapeOP):
    def evaluate(self, valuation=None):
        param = valuation[self.operands[0].fid]
        valuation[self.fid] = array.reshape(param, self._shape)
        return None

class IndexOP(operators.IndexOP):
    def evaluate(self, valuation=None):
        pass
