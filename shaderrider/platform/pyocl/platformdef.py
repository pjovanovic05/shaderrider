"""
Defines PyOpenCL platform.

WRITEME
"""
from numbers import Number

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray

from shaderrider.core import PlatformNotInitializedError
from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators

from shaderrider.generator.pdefs import Function, Valuation, PlatformFactory
from shaderrider.generator.util import topsort_formula

from shaderrider.platform.pyocl import operators as ops


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


class PyOCLFunction(Function):

    def __init__(self, inputs=None, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(expressions, updates, name)
        self._expressions = []
        self._updates = []
        self._epath = []
        self._upath = []
        self._inputs = set()
        self._uinputs = set()

        if expressions is None:
            expressions = []
        if updates is None:
            updates = []

        for expr in expressions:
            vs = expr.get_variables()
            self._inputs.update(v.fid for v in vs)
            pexpr = _get_platform_expression(expr)
            self._expressions.append(pexpr)
            self._epath.append(filter(lambda x: isinstance(x, exprgraph.Operator), topsort_formula(pexpr)))

        for (fid, expr) in updates:
            vs = expr.get_variables()
            self._uinputs.update(v.fid for v in vs)
            pexpr = _get_platform_expression(expr)
            self._updates.append((fid, pexpr))
            self._updates.append((fid, filter(lambda x: isinstance(x, exprgraph.Operator), topsort_formula(pexpr))))

    def evaluate(self, valuation):
        # check inputs?
        for invar in self._inputs:
            if invar not in valuation:
                raise ValueError('Missing argument: ' + invar + ' not found in valuation')

        for ee in self._epath:
            for expr in ee:         # TODO evaluation should set their own events in valuation.events dict
                expr.evaluate(valuation)

        for invar in self._uinputs:
            if invar not in valuation:
                raise ValueError('Missing argument: ' + invar + ' not found in valuation')

        for (upvar, upexpr) in self._upath:
            for expr in upexpr:     # TODO evaluation should set their own events in valuation.events dict
                expr.evaluate(valuation)

        # collect outputs   TODO move to host memory?


def _get_platform_expression(expr):
    """
    Recursively replaces operators in an expr graph with platform specific
    op implementations.
    """
    if isinstance(expr, exprgraph.Operator):
        ops = [_get_platform_expression(op) for op in expr.operands]
        params = {} # TODO
        platform_op = factories[expr.get_type_name()](ops, params)
        platform_op.fid = expr.fid
        return platform_op
    return expr         # TODO jel treba jos nesto kada je atom?


class PyOCLValuation(Valuation):
    def __init__(self, queue=None):
        super(PyOCLValuation, self).__init__()
        self._queue = queue

    def add(self, name, value, const=False, async=False):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')

        if isinstance(value, np.ndarray):
            val = exprgraph.Variable(name, array=value) if not const else exprgraph.Constant(value)
            val._gpu_array = clarray.to_device(self._queue, val.value, async=async)
            self._vars[name] = val
        elif isinstance(value, Number):
            val = exprgraph.Constant(value)
            self._vars[name] = val
        elif isinstance(value, (exprgraph.Variable, exprgraph.Constant)):
            # ovde pomeram value iz variablinog ndarray-a u cl_array!!
            value._gpu_array = clarray.to_device(self._queue, value.value, async=async)
            self._vars[name] = value
        elif isinstance(value, clarray.Array):
            val = exprgraph.Variable(name=name)
            val._gpu_array = value
            self._vars[name] = val
        else:
            raise ValueError        # TODO better throwable needed

    def add_shared(self, name, value, async=False):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')

        if isinstance(value, np.ndarray):
            val = exprgraph.Variable(name=name, array=value)
            val._gpu_array = clarray.to_device(self._queue, value, async=async)
            self._shared[name] = val
        elif isinstance(value, (exprgraph.Variable, exprgraph.Constant)):
            value._gpu_array = clarray.to_device(self._queue, value.value)
            self._shared[name] = value
        elif isinstance(value, clarray.Array):
            val = exprgraph.Variable(name=name)
            val._gpu_array = value
            self._shared[name] = val
        else:
            raise ValueError        # TODO raise better exception

    # def get(self, name, async=False):
    #     if name in self._vars:
    #         # do the movement with clarray.get()
    #         val = self._vars[name]
    #         if val.value is None:
    #             val.value = val._gpu_array.get(async=async) # TODO can this be async at all???????????????????????
    #         else:
    #             val._gpu_array.get(ary=val.value, async=async)
    #         return val
    #     elif name in self._shared:
    #         val = self._shared[name]
    #         if val.value is None:
    #             val.value = val._gpu_array.get(async=async)
    #         else:
    #             val._gpu_array.get(ary=val.value, async=async)  # TODO check if asynchronicity returns events
    #     else:
    #         raise KeyError('Variable "' + name + '" not found in valuation.')

    def read(self, name):
        if name in self._vars:
            # just get the _gpu_arrays
            return self._vars[name]._gpu_array
        elif name in self._shared:
            return self._shared[name]._gpu_array
        else:
            raise KeyError('Variable "' + name + '" not found in valuation.')

    def set(self, name, value, async=False):
        if name in self._shared:
            # TODO check value type (needs to be ndarray of the same type and size)
            self._shared[name]._gpu_array.set(value, async=async)
        elif name in self._vars:
            self._vars[name]._gpu_array.set(value, async=async)
        else:
            raise KeyError('Variable "' + name + '" not found in this valuation.')

    def clear(self, async=False):
        # TODO clear _vars, leave _shared
        # TODO add remove method to remove specific vars?
        # for var in self._vars:
        #     var._gpu_array.data.release();        # risky!! what if a var is used as a function output?
        self._vars.clear()


class PyOCLValuation1(dict):
    def __init__(self, queue,  *args, **kwargs):
        self._queue = queue
        self.update(*args, **kwargs)

    # def __getitem__(self, key):
    #     val = dict.__getitem__(self, key)
    #     return val

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            val = clarray.to_device(self._queue, value)
        elif isinstance(value, clarray.Array):
            val = value
        else:
            raise TypeError('Unsupported value type')
        dict.__setitem__(self, key, val)

    def set_async(self, key, value):
        if isinstance(value, np.ndarray):
            val = clarray.to_device(self._queue, value, async=True)
            dict.__setitem__(self, key, val)

    def get_async(self, key):
        return self[key].get(async=True)

    def read_value(self, key):
        return self[key].get()

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k,v in dict(*args, **kwargs).iteritems():
            self[k] = v


#######################################################################

factories = {
    # operators.ReshapeOP.get_type_name() : ops.create_reshape,
    # operators.RavelOP.get_type_name(): ops.create_ravel,
    # operators.ConcatenateOP.get_type_name(): ops.create_concatenate,
    # operators.StackOP.get_type_name(): ops.create_stack,
    # operators.SplitOP.get_type_name(): ops.create_split,
    # operators.RepeatOP.get_type_name(): ops.create_repeat,
    # operators.BitwiseAndOP.get_type_name(): ops.create_bitwise_and,
    # operators.BitwiseOrOP.get_type_name(): ops.create_bitwise_or,
    # operators.BitwiseXorOP.get_type_name(): ops.create_bitwise_xor,
    # operators.InvertOP.get_type_name(): ops.create_invert,
    # operators.LeftShiftOP.get_type_name(): ops.create_left_shift,
    # operators.RightShiftOP.get_type_name(): ops.create_right_shift,
    # operators.DotOP.get_type_name(): ops.create_dot,
    # operators.VdotOP.get_type_name(): ops.create_vdot,
    # operators.InnerOP.get_type_name(): ops.create_inner,
    # operators.OuterOP.get_type_name(): ops.create_outer,
    # operators.MatmulOP.get_type_name(): ops.create_matmul,
    # operators.EigOP.get_type_name(): ops.create_eig,
    # operators.EigvalsOP.get_type_name(): ops.create_eigvals,
    # operators.AllOP.get_type_name(): ops.create_all,
    # operators.AnyOP.get_type_name(): ops.create_any,
    # operators.AndOP.get_type_name(): ops.create_and,
    # operators.OrOP.get_type_name(): ops.create_or,
    # operators.NotOP.get_type_name(): ops.create_not,
    # operators.XorOP.get_type_name(): ops.create_xor,
    # operators.GtOP.get_type_name(): ops.create_greater,
    # operators.LtOP.get_type_name(): ops.create_less,
    # operators.GeOP.get_type_name(): ops.create_greater_equal,
    # operators.LeOP.get_type_name(): ops.create_less_equal,
    # operators.EqOP.get_type_name(): ops.create_equal,
    # operators.NeOP.get_type_name(): ops.create_not_equal,
    # operators.SinOP.get_type_name(): ops.create_sin,
    # operators.CosOP.get_type_name(): ops.create_cos,
    # operators.TanOP.get_type_name(): ops.create_tan,
    # operators.ArcsinOP.get_type_name(): ops.create_arcsin,
    # operators.ArccosOP.get_type_name(): ops.create_arccos,
    # operators.ArctanOP.get_type_name(): ops.create_arctan,
    # operators.SinhOP.get_type_name(): ops.create_sinh,
    # operators.CoshOP.get_type_name(): ops.create_cosh,
    # operators.TanhOP.get_type_name(): ops.create_tanh,
    # operators.ArcsinhOP.get_type_name(): ops.create_arcsinh,
    # operators.ArccoshOP.get_type_name(): ops.create_arccosh,
    # operators.ArctanhOP.get_type_name(): ops.create_arctanh,
    # operators.RoundOP.get_type_name(): ops.create_round,
    # operators.FloorOP.get_type_name(): ops.create_floor,
    # operators.CeilOP.get_type_name(): ops.create_ceil,
    # operators.ProdOP.get_type_name(): ops.create_prod,
    # operators.SumOP.get_type_name(): ops.create_sum,
    # operators.NansumOP.get_type_name(): ops.create_nansum,
    # operators.CumprodOP.get_type_name(): ops.create_cumprod,
    # operators.CumsumOP.get_type_name(): ops.create_cumsum,
    # operators.ExpOP.get_type_name(): ops.create_exp,
    # operators.Exp2OP.get_type_name(): ops.create_exp2,
    # operators.LogOP.get_type_name(): ops.create_log,
    # operators.Log10OP.get_type_name(): ops.create_log10,
    # operators.Log1pOP.get_type_name(): ops.create_log1p,
    # operators.AddOP.get_type_name(): ops.create_add,
    # operators.ReciprocalOP.get_type_name(): ops.create_reciprocal,
    # operators.NegOP.get_type_name(): ops.create_negative,
    # operators.MulOP.get_type_name(): ops.create_multiply,
    # operators.DivOP.get_type_name(): ops.create_divide,
    # operators.PowOP.get_type_name(): ops.create_power,
    # operators.SubOP.get_type_name(): ops.create_subtract,
    # # operators.TrueDivideOP.get_type_name(): ops.create_true_divide,
    # # operators.FloorDivideOP.get_type_name(): ops.create_floor_divide,
    # operators.ModOP.get_type_name(): ops.create_mod,
    # operators.MedianOP.get_type_name(): ops.create_median,
    # operators.AverageOP.get_type_name(): ops.create_average,
    # operators.MeanOP.get_type_name(): ops.create_mean,
    # operators.StdOP.get_type_name(): ops.create_std,
    # operators.VarOP.get_type_name(): ops.create_var,
    # operators.CorrelateOP.get_type_name(): ops.create_correlate,
    # operators.CovOP.get_type_name(): ops.create_cov
}


class PyOCLFactory(PlatformFactory):
    def __init__(self):
        self._context = None
        self._queues = []
        self._default_queue = -1
        self._initialized = False

    @property
    def default_queue(self):
        return self._queues[self._default_queue]

    @default_queue.setter
    def default_queue(self, value):
        self._default_queue = value

    def init_platform(self, ngpus=0):
        self._context, self._queues = setup_context(ngpus)
        self._default_queue = 0
        self._initialized = True

    def finalize_platform(self):
        self._initialized = False

    def create_valuation(self, device=None):
        if not self._initialized:
            raise PlatformNotInitializedError
        q = self._queues[device] if device is not None else self.default_queue
        return PyOCLValuation(queue=q)

    def create_function(self, expressions=None, updates=None, name=None, skip_platform_opts=False):
        if not self._initialized:
            raise PlatformNotInitializedError
        return PyOCLFunction(expressions=expressions, updates=updates, name=name)

    def create_op(self, type_name, operands, params):
        return factories[type_name](operands, params)
                                                            # XXX mozda su i operatori i funkcije jer nekad su u izrazu a nekad se pozivaju eksterno
    # ARRAY CREATION                                        TODO da li su ovo zapravo samo operatori bez operanada?
    def empty(self, shape, dtype=None, order='C', name=None):
        pass

    def empty_like(self, a, dtype=None, order='C', name=None):
        pass

    def eye(self, N, M=0, k=0, dtype=None, const=False, name=None):
        pass

    def identity(self, N, dtype=None, const=False, name=None):
        pass

    def ones(self, shape, dtype=None, order='C'):
        pass

    def ones_like(self, a, dtype=None, order='C'):
        pass

    def from_data(self):
        pass

    def arange(self):
        pass

    def linspace(self):
        pass

    def logspace(self):
        pass
