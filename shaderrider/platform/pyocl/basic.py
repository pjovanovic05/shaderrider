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

    def init(self):
        self.prog = cl.Program(self._ctx, '''
            __kernel void k_neg_f32(__global float *in, size_t n, __global float *out) {
                int idx = get_global_id(0);
                if (idx < n)
                    out[idx] = !in[idx];
            }

            __kernel void k_neg_f64(__global double *in, size_t n, __global double *out) {
                int idx = get_global_id(0);
                if (idx < n)
                    out[idx] = !in[idx];
            }
        ''')
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

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
