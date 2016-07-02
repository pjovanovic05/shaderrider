"""
Defines PyOpenCL platform.

WRITEME
"""
from numbers import Number

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators

from shaderrider.generator.function import Function, topsort_formula, Valuation, PlatformFactory
from shaderrider.generator import optimization as opt

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


default_ctx, queues = setup_context(1)
default_queue = queues[0]


def platform_init():
    pass                    # TODO


class PyOCLFunction(Function):

    def __init__(self, inputs=None, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(inputs, expressions, updates, name)
        self._expr_evals = []
        self._update_evals = []

        for expr in self._expressions:
            # TODO create platform expression from abstract expression (topsorted and everything else)
            self._expr_evals.append(_get_platform_expression(expr))
        for (v, e) in updates:
            # TODO create platform expr from e
            # TODO save pair (v,platform_e) in update evals or something
            pass

        # TODO da li ovde idu optimizacije?

    def evaluate(self, valuation):
        # check inputs?

        for ee in self._expr_evals:
            evt = ee.evaluate(valuation, valuation.events)
            valuation.events[ee.fid] = evt

        for (upvar, upexpr) in self._update_evals:
            evt = upexpr.evaluate(valuation, valuation.events)
            valuation.events[upvar.fid] = evt

        # collect outputs   TODO move to host memory?


        # TODO WAIT FOR OUTPUT EVENT TO finish calculating???


        outs = []
        for ex in self._expressions:
            outs.append(valuation[ex.fid])
        # TODO transfer outputs?
        return outs


def _get_platform_expression(expr):
    """
    Recursively replaces operators in an expr graph with platform specific
    op implementations.
    """
    # TODO ipak moram da imam operatnds i params argumente zbog rekurzivnih poziva.             <<<<<<< STAO OVDE
    if isinstance(expr, exprgraph.Operator):
        ops = [_get_platform_expression(op) for op in expr.operands]
        params = {} #TODO
        return factories[expr.get_type_name()](ops, params)
    return expr         # TODO jel treba jos nesto kada je atom?


def _compile_expression(expr):              # TODO da li se ovde topsortira? ne treba praviti f-je sa milion odgovornosti!!!!!!!1!!!!!
    """
    Creates platform specific expression graph and performs platform optimizations (optionally).

    :type expr: Formula
    :rtype: list of evaluators
    """

    # TODO expr graph translation?

    # optimizations
    sexpr = expr.simplify()
    # for optimizer in optimizers:
    #     sexpr = optimizer.optimize(sexpr)
    # more opts ...

    # top sort
    ts = topsort_formula(sexpr)
    ops = [op for op in ts if isinstance(op, exprgraph.Operator)]

    return ops


class PyOCLValuation(Valuation):
    def add(self, name, value, const=False, async=False):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')

        if isinstance(value, np.ndarray):
            val = exprgraph.Variable(name, array=value) if not const else exprgraph.Constant(value)
            val._gpu_array = clarray.to_device(default_queue, val.value, async=async)
            self._vars[name] = val
        elif isinstance(value, Number):
            val = exprgraph.Literal(value)
            self._vars[name] = val
        elif isinstance(value, (exprgraph.Variable, exprgraph.Constant)):
            # ovde pomeram value iz variablinog ndarray-a u cl_array!!
            value._gpu_array = clarray.to_device(default_queue, value.value, async=async)
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
            val._gpu_array = clarray.to_device(default_queue, value, async=async)
            self._shared[name] = val
        elif isinstance(value, (exprgraph.Variable, exprgraph.Constant)):
            value._gpu_array = clarray.to_device(default_queue, value.value)
            self._shared[name] = value
        elif isinstance(value, clarray.Array):
            val = exprgraph.Variable(name=name)
            val._gpu_array = value
            self._shared[name] = val
        else:
            raise ValueError        # TODO raise better exception

    def get(self, name, async=False):
        if name in self._vars:
            # do the movement with clarray.get()
            val = self._vars[name]
            if val.value is None:
                val.value = val._gpu_array.get(async=async) # TODO can this be async at all???????????????????????
            else:
                val._gpu_array.get(ary=val.value, async=async)
            return val
        elif name in self._shared:
            val = self._shared[name]
            if val.value is None:
                val.value = val._gpu_array.get(async=async)
            else:
                val._gpu_array.get(ary=val.value, async=async)  # TODO check if asynchronicity returns events
        else:
            raise KeyError('Variable "' + name + '" not found in valuation.')

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




#######################################################################

factories = {
    operators.ReshapeOP.get_type_name() : ops.create_reshape,
    operators.RavelOP.get_type_name(): ops.create_ravel,
    operators.ConcatenateOP.get_type_name(): ops.create_concatenate,
    operators.StackOP.get_type_name(): ops.create_stack,
    operators.SplitOP.get_type_name(): ops.create_split,
    operators.RepeatOP.get_type_name(): ops.create_repeat,
    operators.BitwiseAndOP.get_type_name(): ops.create_bitwise_and,
    operators.BitwiseOrOP.get_type_name(): ops.create_bitwise_or,
    operators.BitwiseXorOP.get_type_name(): ops.create_bitwise_xor,
    operators.InvertOP.get_type_name(): ops.create_invert,
    operators.LeftShiftOP.get_type_name(): ops.create_left_shift,
    operators.RightShiftOP.get_type_name(): ops.create_right_shift,
    operators.DotOP.get_type_name(): ops.create_dot,
    operators.VdotOP.get_type_name(): ops.create_vdot,
    operators.InnerOP.get_type_name(): ops.create_inner,
    operators.OuterOP.get_type_name(): ops.create_outer,
    operators.MatmulOP.get_type_name(): ops.create_matmul,
    operators.EigOP.get_type_name(): ops.create_eig,
    operators.EigvalsOP.get_type_name(): ops.create_eigvals,
    operators.AllOP.get_type_name(): ops.create_all,
    operators.AnyOP.get_type_name(): ops.create_any,
    operators.AndOP.get_type_name(): ops.create_and,
    operators.OrOP.get_type_name(): ops.create_or,
    operators.NotOP.get_type_name(): ops.create_not,
    operators.XorOP.get_type_name(): ops.create_xor,
    operators.GtOP.get_type_name(): ops.create_greater,
    operators.LtOP.get_type_name(): ops.create_less,
    operators.GeOP.get_type_name(): ops.create_greater_equal,
    operators.LeOP.get_type_name(): ops.create_less_equal,
    operators.EqOP.get_type_name(): ops.create_equal,
    operators.NeOP.get_type_name(): ops.create_not_equal,
    operators.SinOP.get_type_name(): ops.create_sin,
    operators.CosOP.get_type_name(): ops.create_cos,
    operators.TanOP.get_type_name(): ops.create_tan,
    operators.ArcsinOP.get_type_name(): ops.create_arcsin,
    operators.ArccosOP.get_type_name(): ops.create_arccos,
    operators.ArctanOP.get_type_name(): ops.create_arctan,
    operators.SinhOP.get_type_name(): ops.create_sinh,
    operators.CoshOP.get_type_name(): ops.create_cosh,
    operators.TanhOP.get_type_name(): ops.create_tanh,
    operators.ArcsinhOP.get_type_name(): ops.create_arcsinh,
    operators.ArccoshOP.get_type_name(): ops.create_arccosh,
    operators.ArctanhOP.get_type_name(): ops.create_arctanh,
    operators.RoundOP.get_type_name(): ops.create_round,
    operators.FloorOP.get_type_name(): ops.create_floor,
    operators.CeilOP.get_type_name(): ops.create_ceil,
    operators.ProdOP.get_type_name(): ops.create_prod,
    operators.SumOP.get_type_name(): ops.create_sum,
    operators.NansumOP.get_type_name(): ops.create_nansum,
    operators.CumprodOP.get_type_name(): ops.create_cumprod,
    operators.CumsumOP.get_type_name(): ops.create_cumsum,
    operators.ExpOP.get_type_name(): ops.create_exp,
    operators.Exp2OP.get_type_name(): ops.create_exp2,
    operators.LogOP.get_type_name(): ops.create_log,
    operators.Log10OP.get_type_name(): ops.create_log10,
    operators.Log1pOP.get_type_naem(): ops.create_log1p,
    operators.AddOP.get_type_name(): ops.create_add,
    operators.ReciprocalOP.get_type_name(): ops.create_reciprocal,
    operators.NegOP.get_type_name(): ops.create_negative,
    operators.MulOP.get_type_name(): ops.create_multiply,
    operators.DivOP.get_type_name(): ops.create_divide,
    operators.PowOP.get_type_name(): ops.create_power,
    operators.SubOP.get_type_name(): ops.create_subtract(),
    operators.TrueDivideOP.get_type_name(): ops.create_true_divide,
    operators.FloorDivideOP.get_type_name(): ops.create_floor_divide,
    operators.ModOP.get_type_name(): ops.create_mod,
    operators.MedianOP.get_type_name(): ops.create_median,
    operators.AverageOP.get_type_name(): ops.create_average,
    operators.MeanOP.get_type_name(): ops.create_mean,
    operators.StdOP.get_type_name(): ops.create_std,
    operators.VarOP.get_type_name(): ops.create_var,
    operators.CorrelateOP.get_type_name(): ops.create_correlate,
    operators.CovOP.get_type_name(): ops.create_cov
}


class PyOCLFactory(PlatformFactory):
    def init_platform(self):
        pass

    def finalize_platform(self):
        pass

    def create_valuation(self):
        pass

    def create_function(self, expressions=None, updates=None, name=None, skip_platform_opts=False):
        pass

    def create_op(self, type_name, operands, params):
        return factories[type_name](*operands)          # FIXME raspakivanje parametara ovde nece raditi
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
