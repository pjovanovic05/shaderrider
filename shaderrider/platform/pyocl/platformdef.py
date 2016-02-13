"""
Defines PyOpenCL platform.
"""

import pyopencl as cl
import pyopencl.array as clarray

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

        if expressions is None and updates is None:
            raise ValueError(
                "Can't create a function for doing nothing. Provide some expressions or updates to execute.")

        self._collect_inputs()
        self._create_evaluation_path()

    def __call__(self, *args, **kwargs):
        valuation = {}
        events = {}

        for arg, i in enumerate(args):
            if isinstance(arg, clarray.Array):
                valuation[self._inputs[i].fid] = arg
            else:
                valuation[self._inputs[i].fid] = clarray.to_device(default_queue, arg)
        # handle kwargs
        for arg in kwargs:
            pass    # TODO

        for ee in self._expr_evals:
            evt = ee.evaluate(valuation, events)
            # TODO update events dict
            # TODO save evaluation output

        for (upvar, upexpr) in self._update_evals:
            evt = upexpr.evaluate(valuation, events)

        # TODO transfer outputs?

    def _collect_inputs(self):
        for expr in self._expressions:
            for a in expr.get_atoms():
                if a not in self._inputs:
                    self._inputs.append(a)
        for var, update in self._updates:
            for a in update.get_atoms():
                if a not in self._inputs:
                    self._inputs.append(a)

    def _create_evaluation_path(self, expr=0):
        ts = topsort_formula(expr)
        outname = ts[-1].fid
        ops = [op for op in ts if isinstance(op, exprgraph.Operator)]


# TODO where do i put the optimizations?
