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
from shaderrider.generator import optimization as opt

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
                continue  # insufficient number of gpus on the platform
            ctx = cl.Context(ds[:ngpus])
            qs = [cl.CommandQueue(ctx, device=d) for d in ds]
            break  # found our platform (probably)
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
        self._update_evals = []  # list of tuples (var, update_expr)
        self._inputs = []  # TODO collect inputs from exprs

        if expressions is None and updates is None:
            raise ValueError(
                "Can't create a function for doing nothing. Provide some expressions or updates to execute.")

        self._collect_inputs()

        # create evaluation paths
        for expr in self._expressions:
            self._expr_evals.extend(_compile_expression(expr))

        for var,update in self._updates:
            self._update_evals.append((var, _compile_expression(update)))

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
            # TODO treba proveriti da li je arg u inputima i proveriti tip ako treba.
            if isinstance(kwargs[arg], clarray.Array):
                valuation[arg] = kwargs[arg]
            else:
                valuation[arg] = clarray.to_device(default_queue, kwargs[arg])

        for ee in self._expr_evals:
            evt = ee.evaluate(valuation, events)
            events[ee.fid] = evt

        for (upvar, upexpr) in self._update_evals:
            evt = upexpr.evaluate(valuation, events)
            events[upvar.fid] = evt

        # collect outputs
        outs = []
        for ex in self._expressions:
            outs.append(valuation[ex.fid])
        # TODO transfer outputs?
        return outs

    def _collect_inputs(self):
        for expr in self._expressions:
            for a in expr.get_atoms():
                if a not in self._inputs:
                    self._inputs.append(a)
        for var, update in self._updates:
            for a in update.get_atoms():
                if a not in self._inputs:
                    self._inputs.append(a)


def _compile_expression(expr):
    """
    Creates a list of evaluators to be called in order, which represents the execution of the expression.

    :type expr: Formula
    :rtype: list of evaluators
    """
    # run graph checks
    # run optimizations
    #   * simplify
    #   * fold constants
    #   * replace blas?
    #   * elementwise contraction
    # top sort
    # return exec path array

    # optimizations
    sexpr = expr.simplify()
    # more opts ...

    # top sort
    ts = topsort_formula(sexpr)
    ops = [op for op in ts if isinstance(op, exprgraph.Operator)]

    return ops
