"""
WRITEME
"""

from shaderrider.symbolic import blas
from shaderrider.platform.pyocl.aux import clblaswrap as clblas
from shaderrider.platform.pyocl import platformdef


class GemmOP(blas.GemmOP):
    def evaluate(self, valuation=None, events=None, device=0):
        A = valuation[self.operands[0].fid].data
        B = valuation[self.operands[1].fid].data
        C = valuation[self.fid].data     # TODO if has_key?
        alpha = valuation[self.operands[3].fid]
        beta = valuation[self.operands[4].fid]
        transA = valuation[self.operands[5].fid]
        transB = valuation[self.operands[6].fid]

        # TODO check array orders

        waits = []
        for op in self.operands:
            if op.fid in events:
                waits.append(events[op.fid])

        # return clblas.gemm(platformdef.queues[device], A, B, C, transA, transB, alpha, beta, wait_for=waits)


class GemvOP(blas.GemvOP):
    def evaluate(self, valuation=None, events=None, device=0):
        A = valuation[self.operands[0].fid].data
        x = valuation[self.operands[1].fid].data
        y = valuation[self.operands[2].fid].data
        transA = valuation[self.operands[3].fid]
        alpha = valuation[self.operands[4].fid]
        beta = valuation[self.operands[5].fid]

        waits = []
        for op in self.operands:
            if op.fid in events:
                waits.append(events[op.fid])

        # return clblas.gemv(platformdef.queues[device], A, x, y, transA, alpha, beta, wait_for=waits)


class GerOP(blas.GerOP):
    def evaluate(self, valuation=None, events=None, device=0):
        A = valuation[self.operands[0].fid].data
        x = valuation[self.operands[1].fid].data
        y = valuation[self.operands[2].fid].data
        alpha = valuation[self.operands[3].fid]

        waits = []
        for op in self.operands:
            if op.fid in events:
                waits.append(events[op.fid])

        # return clblas.ger(platformdef.queues[device], A, x, y, alpha, wait_for=waits)
