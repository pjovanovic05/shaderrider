"""
WRITEME
"""

import numpy as np
import pyopencl as cl
from pyopencl.array import Array

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import basic

from shaderrider.platform.pyocl import platformdef as platform


class NegOP(basic.NegOP):

    def __init__(self, operand, ctx=None, device=0):
        super(NegOP, self).__init__(operand)
        self._ctx = platform.default_ctx if ctx is None else ctx
        self._fn = self.generate_eval()

    def evaluate(self, valuation=None, events=None):
        return self._fn(valuation, events)

    def generate_eval(self):
        dtype = self.operands[0].dtype
        typestr = 'float'
        if dtype == np.float32:
            typestr = 'float'
        elif dtype == np.float64:
            typestr = 'double'
        else:
            raise NotImplementedError('Unsupported data type on PyOpenCL platform: %s' % str(dtype))

        prog = cl.Program(self._ctx, '''
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


class ExpOP(basic.ExpOP):
    def __init__(self, operand, ctx=None, device=0):
        super(ExpOP, self).__init__(operand)
        self._ctx = platform.default_ctx if ctx is None else ctx

    def evaluate(self, valuation=None, events=None):
        pass

    def generate_eval(self):
        pass


class LogOP(basic.LogOP):
    pass


class SinOP(basic.SinOP):
    pass


class CosOP(basic.CosOP):
    pass


class TanOP(basic.TanOP):
    pass


# binary

class AddOP(basic.AddOP):
    pass


class SubOP(basic.SubOP):
    pass


class MulOP(basic.MulOP):
    pass


class DivOP(basic.DivOP):
    pass


class PowOP(basic.PowOP):
    pass


# comparisons

class EqOP(basic.EqOP):
    pass


class GtOP(basic.GtOP):
    pass


class LtOP(basic.LtOP):
    pass


class GeOP(basic.GeOP):
    pass


class LeOP(basic.LeOP):
    pass

class NeOP(basic.NeOP):
    pass
