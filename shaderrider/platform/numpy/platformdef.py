"""
Numpy based execution platform
"""
from numbers import Number

import numpy as np

from shaderrider.symbolic import exprgraph, operators
from shaderrider.generator.function import Function, Valuation, PlatformFactory, topsort_formula
from shaderrider.platform.numpy import operators as ops


class NPValuation(Valuation):
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


class NPFunction(Function):
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


class NPFactory(PlatformFactory):
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
