"""
WRITEME
"""

import numpy as np
import pyopencl as cl
from pyopencl.array import Array

from shaderrider.symbolic import vast
from shaderrider.symbolic import basic
from shaderrider.generator import codegen


class NegEval(codegen.OpEvaluator):
    def __init__(self, ctx):
        self._ctx = ctx
        self.prog = None

    def setup(self, dtype=np.float32):
        typestr = 'float'
        if dtype == np.float32:
            typestr = 'float'
        elif dtype == np.float64:
            typestr = 'double'
        else:
            raise NotImplementedError('Unsupported data type on PyOpenCL platform: %s' % str(dtype))

        self.prog = cl.Program(self._ctx, '''
            __kernel void k_neg(__global %(typestr)s *in, size_t n, __global %(typestr)s *out) {
                int idx = get_global_id(0);
                if (idx < n)
                    out[idx] = -in[idx];
            }
            __kernel void k_neg_inplace(__global %(typestr)s *a, size_t n) {
                int idx = get_global_id(0);
                if (idx < n)
                    a[i] = -a[i];
            }''' % locals()).build()

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None, events=None):
        arg = valuation[op.operands[0].fid]
        out = valuation[op.fid]
        waits = []
        if events is not None and op.operands[0].fid in events:
            waits.append(events[op.operands[0].fid])
        myevt = self.prog.k_neg(arg.data, arg.size, out.data, wait_for=waits)
        if events is not None:
            events[op.fid] = myevt
        return myevt

    def after(self, op, valuation=None):
        pass


class ExpEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class LogEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class SinEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class CosEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class TanEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


# binary

class AddEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class SubEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class MulEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class DivEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class PowEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


# comparisons

class EqEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class GtEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class LtEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class GeEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class LeEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class NeEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass
