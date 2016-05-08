"""
Defines PyOpenCL platform.
"""

import pyopencl as cl
import pyopencl.array as clarray

from shaderrider.generator.codegen import FormulaFactory
from shaderrider.symbolic import exprgraph

from shaderrider.generator.function import Function, topsort_formula, Valuation
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


# TODO ovo je pogresno, razmisljaj o expressionima koje dobijes kao vec proverenim i optimizovanim
# TODO ovaj function treba samo da zameni expressione instancama svoje implementacije!
# TODO ovo takodje treba da resi i pomeranje vrednosti u valuaciji u PyOpencl array
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

        for var, update in self._updates:
            self._update_evals.append((var, _compile_expression(update)))

    def novi_konstruktor(self, inputs=None, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(inputs, expressions, updates, name)
        self._expr_evals = []
        self._update_evals = []

        # TODO svaki expression zameni operatorima platforme
        for expr in expressions:
            self._expr_evals.append(_replace_ops(expr))
        # TODO svaki update zameni operatorima platforme


    def evaluate(self, valuation):
        # valuation = {}
        # events = {}

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


optimizers = [opt.ElementwiseOpt()]


def _compile_expression(expr):
    """
    Creates a list of evaluators to be called in order, which represents the execution of the expression.

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
    def add(self, name, value):
        pass

    def add_shared(self, name, value):
        pass

    def get(self, name):
        pass


class PyOCLPlatform(object):
    @classmethod
    def get_validations(cls):
        """gets validation objects which check the validity of an expression graph"""
        return []

    def get_optimizations(self):
        """gets optimization objects which implement platform specific optimizations on an expression graph"""
        return []

    def write_value(self):
        """puts something into the platform valuation (host2device)"""
        pass

    def read_value(self):
        """reads something from the platform valuation (device2host)"""
        pass

    def create_function(self, inputs, expressions, updates, name):
        """create appropriate function instance for this platform"""
        # treba da zamenim sve operatore u grafu operatorima platforme
        return PyOCLFunction()


class PyOCLFFactory(FormulaFactory):
    def create_valuation(vdict):
        pass

    def create_neg(self, operand):
        return bo.NegOP(operand)

    def create_exp(self, operand):
        return bo.ExpOP(operand)

    def create_log(self, operand):
        return bo.LogOP(operand)

    def create_sin(self, operand):
        return bo.SinOP(operand)

    def create_cos(self, operand):
        return bo.CosOP(operand)

    def create_tan(self, operand):
        return bo.TanOP(operand)

    def create_add(self, op1, op2):
        return bo.AddOP(op1, op2)

    def create_sub(self, op1, op2):
        return bo.SubOP(op1, op2)

    def create_mul(self, op1, op2):
        return bo.MulOP(op1, op2)

    def create_div(self, op1, op2):
        return bo.DivOP(op1, op2)

    def create_pow(self, op1, op2):
        return bo.PowOP(op1, op2)

    def create_eq(self, op1, op2):
        return bo.EqOP(op1, op2)

    def create_gt(self, op1, op2):
        return bo.GtOP(op1, op2)

    def create_lt(self, op1, op2):
        return bo.LtOP(op1, op2)

    def create_ge(self, op1, op2):
        return bo.GeOP(op1, op2)

    def create_le(self, op1, op2):
        return bo.LeOP(op1, op2)

    def create_ne(self, op1, op2):
        return bo.NeOP(op1, op2)

    def create_elementwise(self, formula):
        return elementwise.ElementwiseOP(formula)

    def create_gemm(self, A, B, C,
                    alpha=exprgraph.Constant(1.0),
                    beta=exprgraph.Constant(0.0),
                    transA=exprgraph.Constant(False),
                    transB=exprgraph.Constant(False),
                    parent=None):
        return blas.GemmOP(A, B, C, alpha, beta, transA, transB, parent)

    def create_gemv(self, A, X, Y, alpha=exprgraph.Constant(1.0), beta=exprgraph.Constant(0.0),
                    transA=exprgraph.Constant(False), parent=None):
        return blas.GemvOP(A, X, Y, alpha, beta, transA, parent)

    def create_ger(self, alpha, X, Y, A, parent=None):
        return blas.GerOP(alpha, X, Y, A, parent)
