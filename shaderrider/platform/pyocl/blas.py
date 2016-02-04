"""
WRITEME
"""

from shaderrider.symbolic import blas
from shaderrider.generator import codegen
from shaderrider.platform.pyocl.aux import clblaswrap as clblas


class GemmOP(blas.GemmOP):
    def evaluate(self, valuation=None, events=None, device=0):
        A = valuation[self.operands[0].fid]
        B = valuation[self.operands[1].fid]
        C = valuation[self.fid]     # TODO if has_key?
        alpha = valuation[self.operands[3].fid]
        beta = valuation[self.operands[4].fid]
        transA = valuation[self.operands[5].fid]
        transB = valuation[self.operands[6].fid]
        return clblas.gemm()


class GemvOP(blas.GemvOP):
    pass


class GerOP(blas.GerOP):
    pass
