"""
WRITEME
"""

import pyopencl as cl

from shaderrider.generator import codegen


class ElementwiseEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def eval(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class ElementwiseGenerator(codegen.OpEvalGenerator):
    def generate(self, op, valuation=None):
        pass
