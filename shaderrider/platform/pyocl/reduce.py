"""
WRITEME
"""

from pyopencl.reduction import ReductionKernel

from shaderrider.generator import codegen


class ReduceGenerator(codegen.OpEvalGenerator):
    def generate(self, op, ctx):
        pass
