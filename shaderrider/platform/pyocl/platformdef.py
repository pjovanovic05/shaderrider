"""
Defines PyOpenCL platform.
"""

import pyopencl as cl

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import basic as sbo
from shaderrider.symbolic import blas as sblas
from shaderrider.symbolic import elementwise as sew

from shaderrider.generator.function import Function, topsort_formula

from shaderrider.platform.pyocl import basic as bo
from shaderrider.platform.pyocl import blas
from shaderrider.platform.pyocl import elementwise


def setup_context(ngpus=0):
    """

    :param ngpus:
    :return:
    """
    ctx = None
    qs = None
    if ngpus > 0:
        ps = cl.get_platforms()
        for p in ps:
            ds = p.get_devices(device_type=cl.device_type.GPU)
            if len(ds) < ngpus:
                continue    # insufficient number of gpus on the platform
            ctx = cl.Context(ds[:ngpus])
            qs = [cl.CommandQueue(ctx, device=d) for d in ds]
            break   # found our platform (probably)
    else:
        ctx = cl.create_some_context()
        qs = [cl.CommandQueue(ctx)]
    return ctx, qs


default_ctx, queues = setup_context(1)
default_queue = queues[0]


class PyOCLFunction(Function):
    def __init__(self, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(expressions, updates, name)
        self._expr_evals = []
        self._update_evals = []     # list of tuples (var, update_expr)
        self._inputs = []           # TODO collect inputs from exprs

    def __call__(self, *args, **kwargs):
        valuation = {}
        events = {}
        for ee in self._expr_evals:
            evt = ee.evaluate(valuation, events)
            # TODO update events dict
            # TODO save evaluation output

        for (upvar, upexpr) in self._update_evals:
            evt = upexpr.evaluate(valuation, events)

    def _create_evaluation_path(self):
        for expr in self._expressions:
            ts = topsort_formula(expr)
            outname = ts[-1].fid


# TODO where do i put the optimizations?
