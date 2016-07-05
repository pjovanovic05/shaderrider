"""
Numpy based execution platform
"""
from numbers import Number

import numpy as np

import shaderrider.generator.pdefs as pdefs
from shaderrider.generator.util import topsort_formula
from shaderrider.symbolic import exprgraph
from shaderrider.platform.numpy import operators as ops


class NPValuation(pdefs.Valuation):
    def add(self, name, value):
        if isinstance(value, np.ndarray):
            self._vars[name] = value
        elif isinstance(value, Number):
            self._vars[name] = value
        elif isinstance(value, exprgraph.Atom):
            self._vars[name] = value.value          # TODO check value type?
        else:
            raise TypeError        # TODO raise unsupported type or something

    def add_shared(self, name, value):
        if isinstance(value, np.ndarray):
            self._shared[name] = value
        elif isinstance(value, Number):
            self._shared[name] = value
        elif isinstance(value, exprgraph.Atom):
            self._shared[name] = value.value
        else:
            raise TypeError  # TODO raise unsupported type or something

    # get stays the same...
    # def get(self, name):
    #     pass
    #
    # def set(self, name, value):
    #     pass


class NPFunction(pdefs.Function):
    def __init__(self, expressions=None, updates=None, name=None):
        super(NPFunction, self).__init__(expressions, updates, name)

        self._expressions = []
        self._epath = []
        self._updates = []
        self._upath = []
        self._inputs = set()
        self._uinputs = set()

        for expr in expressions:
            # TODO platform optimizations?
            vs = expr.get_variables()
            self._inputs.update(v.fid for v in vs)
            pexpr = _get_platform_expression(expr)
            self._expressions.append(pexpr)
            self._epath.append(topsort_formula(pexpr))          # TODO get ops only!!!

        for (fid, expr) in updates:
            vs = expr.get_variables()
            self._uinputs.update(v.fid for v in vars)
            pexpr = _get_platform_expression(expr)
            self._upath.append((fid, topsort_formula(pexpr)))          # TODO extract operators!!!

    def evaluate(self, valuation):
        """

        :type valuation: NPValuation
        """
        # check that inputs are in valuation
        for invar in self._inputs:
            if invar not in valuation:
                raise ValueError('Missing argument: ' + invar + ' not found in valuation')

        # evaluate the evaluation paths
        for ep in self._epath:
            for expr in ep:
                expr.evaluate(valuation)

        # TODO collect (rename?) output
        # TODO free valuation from temps that will not be reused?

        # check if inputs for updates are present
        for invar in self._uinputs:
            if invar not in valuation:
                raise ValueError('Missing argument: ' + invar + ' not found in valuation')

        # evaluate update paths
        for v, up in self._upath:
            for expr in up:
                expr.evaluate(valuation)
            valuation.set(v, valuation.get(up[-1].fid))

        # TODO return ouputs from _expressions??


def _get_platform_expression(expr):
    if isinstance(expr, exprgraph.Operator):
        ops = [_get_platform_expression(op) for op in expr.operands]
        params = expr.params
        platform_op = factories[expr.get_type_name()](ops, params)
        for op in ops:
            op.parents.push(platform_op)
        return platform_op
    elif isinstance(expr, exprgraph.Atom):
        return expr

    raise NotImplementedError           # TODO


factories = {
    ops.ReshapeOP.get_type_name() : ops.create_reshape,
    ops.RavelOP.get_type_name(): ops.create_ravel,
    ops.ConcatenateOP.get_type_name(): ops.create_concatenate,
    ops.StackOP.get_type_name(): ops.create_stack,
    ops.SplitOP.get_type_name(): ops.create_split,
    ops.RepeatOP.get_type_name(): ops.create_repeat,
    ops.BitwiseAndOP.get_type_name(): ops.create_bitwise_and,
    ops.BitwiseOrOP.get_type_name(): ops.create_bitwise_or,
    ops.BitwiseXorOP.get_type_name(): ops.create_bitwise_xor,
    ops.InvertOP.get_type_name(): ops.create_invert,
    ops.LeftShiftOP.get_type_name(): ops.create_left_shift,
    ops.RightShiftOP.get_type_name(): ops.create_right_shift,
    ops.DotOP.get_type_name(): ops.create_dot,
    ops.VdotOP.get_type_name(): ops.create_vdot,
    ops.InnerOP.get_type_name(): ops.create_inner,
    ops.OuterOP.get_type_name(): ops.create_outer,
    ops.MatmulOP.get_type_name(): ops.create_matmul,
    ops.EigOP.get_type_name(): ops.create_eig,
    ops.EigvalsOP.get_type_name(): ops.create_eigvals,
    ops.AllOP.get_type_name(): ops.create_all,
    ops.AnyOP.get_type_name(): ops.create_any,
    ops.AndOP.get_type_name(): ops.create_and,
    ops.OrOP.get_type_name(): ops.create_or,
    ops.NotOP.get_type_name(): ops.create_not,
    ops.XorOP.get_type_name(): ops.create_xor,
    ops.GtOP.get_type_name(): ops.create_greater,
    ops.LtOP.get_type_name(): ops.create_less,
    ops.GeOP.get_type_name(): ops.create_greater_equal,
    ops.LeOP.get_type_name(): ops.create_less_equal,
    ops.EqOP.get_type_name(): ops.create_equal,
    ops.NeOP.get_type_name(): ops.create_not_equal,
    ops.SinOP.get_type_name(): ops.create_sin,
    ops.CosOP.get_type_name(): ops.create_cos,
    ops.TanOP.get_type_name(): ops.create_tan,
    ops.ArcsinOP.get_type_name(): ops.create_arcsin,
    ops.ArccosOP.get_type_name(): ops.create_arccos,
    ops.ArctanOP.get_type_name(): ops.create_arctan,
    ops.SinhOP.get_type_name(): ops.create_sinh,
    ops.CoshOP.get_type_name(): ops.create_cosh,
    ops.TanhOP.get_type_name(): ops.create_tanh,
    ops.ArcsinhOP.get_type_name(): ops.create_arcsinh,
    ops.ArccoshOP.get_type_name(): ops.create_arccosh,
    ops.ArctanhOP.get_type_name(): ops.create_arctanh,
    ops.RoundOP.get_type_name(): ops.create_round,
    ops.FloorOP.get_type_name(): ops.create_floor,
    ops.CeilOP.get_type_name(): ops.create_ceil,
    ops.ProdOP.get_type_name(): ops.create_prod,
    ops.SumOP.get_type_name(): ops.create_sum,
    ops.NansumOP.get_type_name(): ops.create_nansum,
    ops.CumprodOP.get_type_name(): ops.create_cumprod,
    ops.CumsumOP.get_type_name(): ops.create_cumsum,
    ops.ExpOP.get_type_name(): ops.create_exp,
    ops.Exp2OP.get_type_name(): ops.create_exp2,
    ops.LogOP.get_type_name(): ops.create_log,
    ops.Log10OP.get_type_name(): ops.create_log10,
    ops.Log1pOP.get_type_name(): ops.create_log1p,
    ops.AddOP.get_type_name(): ops.create_add,
    ops.ReciprocalOP.get_type_name(): ops.create_reciprocal,
    ops.NegOP.get_type_name(): ops.create_negative,
    ops.MulOP.get_type_name(): ops.create_multiply,
    ops.DivOP.get_type_name(): ops.create_divide,
    ops.PowOP.get_type_name(): ops.create_power,
    ops.SubOP.get_type_name(): ops.create_subtract,
    #ops.TrueDivideOP.get_type_name(): ops.create_true_divide,
    #ops.FloorDivideOP.get_type_name(): ops.create_floor_divide,
    ops.ModOP.get_type_name(): ops.create_mod,
    ops.MedianOP.get_type_name(): ops.create_median,
    ops.AverageOP.get_type_name(): ops.create_average,
    ops.MeanOP.get_type_name(): ops.create_mean,
    ops.StdOP.get_type_name(): ops.create_std,
    ops.VarOP.get_type_name(): ops.create_var,
    ops.CorrelateOP.get_type_name(): ops.create_correlate,
    ops.CovOP.get_type_name(): ops.create_cov
}


class NPFactory(pdefs.PlatformFactory):
    def init_platform(self):
        pass

    def finalize_platform(self):
        pass

    def create_valuation(self):
        return NPValuation()

    def create_function(self, expressions=None, updates=None, name=None, skip_platform_opts=False):
        pass

    def create_op(self, type_name, operands, params):
        return factories[type_name](operands, params)

    def empty(self, shape, dtype=None, order='C', name=None):
        ary = np.empty(shape, dtype, order)
        return exprgraph.Variable(name=name, array=ary)

    def empty_like(self, a, dtype=None, order='C', name=None):
        ary = np.empty_like(a)
        return exprgraph.Variable(name=name, array=ary)

    def eye(self, N, M=0, k=0, dtype=None, const=False, name=None):
        ary = np.eye(N, M, k, dtype=dtype)
        if const:
            return exprgraph.Constant(ary, name=name)
        return exprgraph.Variable(name=name, array=ary)

    def identity(self, N, dtype=None, const=False, name=None):
        pass        # FIXME same as eye

    def ones(self, shape, dtype=None, order='C', const=False, name=None):
        ary = np.ones(shape. dtype, order)
        if const:
            return exprgraph.Constant(ary, name)
        return exprgraph.Variable(name=name, array=ary)

    def ones_like(self, a, dtype=None, order='C', const=False, name=None):
        ary = np.ones_like(a)
        if const:
            return exprgraph.Constant(ary, name)
        return exprgraph.Variable(name=name, array=ary)

    def from_data(self):
        pass

    def arange(self, start, stop, step=None, dtype=None, const=False, name=None):
        ary = np.arange(start, stop, step, dtype)
        if const:
            return exprgraph.Constant(ary, name)
        return exprgraph.Variable(name=name, array=ary)

    def linspace(self, start, stop, num=50, endpoint=True, const=False, name=None):
        ary = np.linspace(start, stop, num, endpoint)
        if const:
            return exprgraph.Constant(ary, name)
        return exprgraph.Variable(name=name, array=ary)

    def logspace(self, start, stop, num, endpoint, base, const=False, name=None):
        raise NotImplementedError
