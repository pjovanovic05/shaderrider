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

    def init(self, dtype=np.float32):
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

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        arg = valuation[op.operands[0].fid]
        out = valuation[op.fid]
        # TODO wait for operand events?
        return self.prog.k_neg(arg.data, arg.size, out.data)

    def after(self, op, valueation=None):
        pass


class ExpEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class LogEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class SinEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class CosEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class TanEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


# binary

class AddEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class SubEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class MulEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class DivEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class PowEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


# comparisons

class EqEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class GtEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class LtEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class GeEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class LeEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass


class NeEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valueation=None):
        pass
