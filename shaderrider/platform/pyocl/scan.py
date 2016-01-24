"""
WRITEME
"""
import numpy as np
from pyopencl.scan import GenericScanKernel

from shaderrider.generator import codegen


class ScanEval(codegen.OpEvaluator):
    # def __init__(self, ctx, dtype, arguments, input_expr, scan_expr, neutral, output_statement,
    #              is_segment_start_expr=None, input_fetch_exprs=[], index_dtype=np.int32, name_prefix='scan',
    #              options=[], preamble='', devices=None):
    #     self._scankernel = GenericScanKernel(ctx, dtype, arguments, input_expr, scan_expr, neutral, output_statement,
    #                                          is_segment_start_expr, input_fetch_exprs, index_dtype, name_prefix,
    #                                          options, preamble, devices)

    def __init__(self, fn, sequences):
        # TODO this will have to generate input and scan expressions from symbolic fn and sequences
        pass

    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass
