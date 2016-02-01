"""
WRITEME
"""

import numpy as np
import pyopencl as cl
from pyopencl.array import Array

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import basic
from shaderrider.generator import codegen

from shaderrider.platform.pyocl import platformdef as platform


class NegOP(basic.NegOP):

    def evaluate(self, valuation=None, events=None):
        arg = valuation[op.operands[0].fid]
        out = valuation[op.fid]
        waits = []
        if events is not None and self.operands[0].fid in events:
            waits.append(events[self.operands[0].fid])
        myevt = self.prog.k_neg(arg.data, arg.size, out.data, wait_for=waits)
        if events is not None:
            events[self.fid] = myevt
        return myevt

    def generate_eval(self):
        dtype = self.operands[0].dtype
        typestr = 'float'
        if dtype == np.float32:
            typestr = 'float'
        elif dtype == np.float64:
            typestr = 'double'
        else:
            raise NotImplementedError('Unsupported data type on PyOpenCL platform: %s' % str(dtype))

        prog = cl.Program(platform.default_ctx, '''
            __kernel void k_neg(__global %(typestr)s *in, size_t n, __global %(typestr)s *out) {
                    int idx = get_global_id(0);
                    if (idx < n)
                        out[idx] = -in[idx];
            }''' % locals()).build()

        def evaluator(valuation, events=None, device=0):
            param = valuation[self.operands[0].fid]
            out = valuation[self.fid]
            waits = [events[ev.fid] for ev in self.operands] if events is not None else None

            if out is None or out.shape != param.shape:
                out = cl.array.empty(platform.queues[device], param.shape, param.dtype)   #TODO
            evt = prog.k_neg(platform.queues[device], param.shape, None, param.data, out.data, wait_for=waits)
            return evt

        return evaluator


class ExpEval(codegen.OpEvaluator):
    pass


class LogEval(codegen.OpEvaluator):
    pass


class SinEval(codegen.OpEvaluator):
    pass


class CosEval(codegen.OpEvaluator):
    pass


class TanEval(codegen.OpEvaluator):
    pass


# binary

class AddEval(codegen.OpEvaluator):
    pass


class SubEval(codegen.OpEvaluator):
    pass


class MulEval(codegen.OpEvaluator):
    pass


class DivEval(codegen.OpEvaluator):
    pass


class PowEval(codegen.OpEvaluator):
    pass


# comparisons

class EqEval(codegen.OpEvaluator):
    pass


class GtEval(codegen.OpEvaluator):
    pass


class LtEval(codegen.OpEvaluator):
    pass


class GeEval(codegen.OpEvaluator):
    pass


class LeEval(codegen.OpEvaluator):
    pass

class NeEval(codegen.OpEvaluator):
    pass