"""
Defines PyOpenCL platform.
"""
from numbers import Number

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import operators

from shaderrider.generator.function import Function, topsort_formula, Valuation, PlatformFactory
from shaderrider.generator import optimization as opt


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

    #           (self, expression, outvarnames, name)                                                           <- TODO
    def __init__(self, inputs=None, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(inputs, expressions, updates, name)
        self._expr_evals = []
        self._update_evals = []

        for expr in self._expressions:
            # TODO create platform expression from abstract expression (topsorted and everything else)
            pass
        for (v, e) in updates:
            # TODO create platform expr from e
            # TODO save pair (v,platform_e) in update evals or something
            pass
        # bice samo jedan expression po funkciji
        # primice apstraktni simbolicki graf
        # graf ce vec proci (opciono) kroz generalne optimizacije
        #   treba da prodje i kroz platformske
        # onda se topsort_formula pozove
        # i napravi se niz evaluatora koji ce se pozivati   (evaluator == operator za platformu)
        # outvarnames je niz varijabli u valuaciji koje ce biti napravljene ili updateovane kao rezultati funkcije
        # evaluacija treba da vrati samo event koji se ceka? ili da ga samo podesi u svojim rezultatima u valuaciji!

    def evaluate(self, valuation):
        # check inputs?

        for ee in self._expr_evals:
            evt = ee.evaluate(valuation, valuation.events)
            valuation.events[ee.fid] = evt

        for (upvar, upexpr) in self._update_evals:
            evt = upexpr.evaluate(valuation, valuation.events)
            valuation.events[upvar.fid] = evt

        # collect outputs   TODO move to host memory?
        outs = []
        for ex in self._expressions:
            outs.append(valuation[ex.fid])
        # TODO transfer outputs?
        return outs

    def _collect_inputs(self):
        for expr in self._expressions:
            for a in expr.get_variables():
                if a not in self._inputs:
                    self._inputs.append(a)
        for var, update in self._updates:
            for a in update.get_variables():
                if a not in self._inputs:
                    self._inputs.append(a)


optimizers = [opt.ElementwiseOpt()]


def _compile_expression(expr):
    """
    Creates a list of evaluators to be called in order, which represents the execution
    of the expression.

    :type expr: Formula
    :rtype: list of evaluators
    """

    # optimizations
    sexpr = expr.simplify()
    for optimizer in optimizers:
        sexpr = optimizer.optimize(sexpr)
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


# OPERATOR FACTORIES                TODO move into operator module - each after the op it creates ##########################################################################################

# ARRAY MANIPULATION
def create_reshape(a, newshape):
    pass


def create_ravel(a):
    pass


def create_concatenate(a1, a2):
    pass


def create_stack(xs, axis):
    pass


def create_split(a, indicies):
    pass


def create_repeat(a, repeats, axis):
    pass


# BINARY OPERATIONS

def create_bitwise_and(x1, x2):
    pass


def create_bitwise_or(x1, x2):
    pass


def create_bitwise_xor(x1, x2):
    pass


def create_invert(x1, x2):
    pass


def create_left_shift(x1, x2):
    pass


def create_right_shift(x1, x2):
    pass


# INDEXING OPS
# TODO

# LINEAR ALGEBRA

def create_dot(a, b):
    pass


def create_vdot(a, b):
    pass


def create_inner(a, b):
    pass


def create_outer(a, b):
    pass


def create_matmul(a, b):
    pass


def create_eig(a):
    pass


def create_eigvals(a):
    pass


# LOGIC OPS

def create_all(a):
    pass


def create_any(a):
    pass


def create_and(a, b):
    pass


def create_or(a, b):
    pass


def create_not(a):
    pass


def create_xor(a, b):
    pass


def create_greater(a, b):
    pass


def create_less(a, b):
    pass


def create_greater_equal(a, b):
    pass


def create_less_equal(a, b):
    pass


def create_equal(a, b):
    pass


def create_not_equal(a, b):
    pass


# MATHEMATICAL OPS

def create_sin(x):
    pass


def create_cos(x):
    pass


def create_tan(x):
    pass


def create_arcsin(x):
    pass


def create_arccos(x):
    pass


def create_arctan(x):
    pass


def create_sinh(x):
    pass


def create_cosh(x):
    pass


def create_tanh(x):
    pass


def create_arcsinh(x):
    pass


def create_arccosh(x):
    pass


def create_arctanh(x):
    pass


def create_round(a, decimal=None, out=None):
    pass


def create_floor(x, out=None):
    pass


def create_ceil(x, out=None):
    pass


def create_prod(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_sum(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_nansum(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_cumprod(a, axis=None, dtype=None, out=None, keepdims=None):
    pass


def create_cumsum(a, axis, dtype, out, keepdims):
    pass


def create_exp(x):
    pass


def create_exp2(x, out=None):
    pass


def create_log(x, out=None):
    pass


def create_log10(x, out=None):
    pass


def create_log1p(x, out=None):
    pass


def create_add(x1, x2, out=None):
    pass


#######################################################################

factories = {
    operators.ReshapeOP.get_type_name() : create_reshape,
    operators.RavelOP.get_type_name(): create_ravel,
    operators.ConcatenateOP.get_type_name(): create_concatenate,
    'stack': create_stack
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

    def create_op(self, type_name, operands):
        return factories[type_name](*operands)

    # ARRAY CREATION                                        TODO da li su ovo zapravo samo operatori bez operanada?
    def empty(self, shape, dtype=None, order='C'):
        pass

    def empty_like(self, a, dtype=None, order='C'):
        pass

    def eye(self, N, M=0, k=0, dtype=None):
        pass

    def identity(self, N, dtype=None):
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
