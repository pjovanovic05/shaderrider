import abc

from shaderrider.symbolic import exprgraph


class OpGenerator(object):
    """
    Abstract code generator for operators.

    Intended to be implemented in platforms which rely on template code generation.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        self._name = name

    @abc.abstractmethod
    def generate_support_code(self):
        pass

    @abc.abstractmethod
    def generate_before_code(self, op):
        pass

    @abc.abstractmethod
    def generate_after_code(self, op):
        pass

    @abc.abstractmethod
    def generate_eval_code(self, op):
        pass

    @abc.abstractmethod
    def generate_init_code(self):
        pass

    @abc.abstractmethod
    def generate_finalize_code(self):
        pass


class COpGenerator(OpGenerator):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def c_headers(self):
        """Returns the list of headers to be included for this formula.

        If the header name does not begin with '<' it is assumed to be
        locally referenced (i.e. include "header.h").
        """
        return []

    @abc.abstractmethod
    def c_header_dirs(self):
        """Returns the list of include dirs where required headers are.
        Optional.

        """
        return []

    @abc.abstractmethod
    def c_libraries(self):
        return []

    @abc.abstractmethod
    def c_lib_dirs(self):
        return []

    @abc.abstractmethod
    def c_compile_args(self):
        return []


class OpEvaluator(object):
    """
    Evaluates an operator.

    Intended for platforms which execute expression graphs directly in python context.
    This is the abstract class that all generated op evaluators should implement.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def teardown(self):
        pass

    @abc.abstractmethod
    def before(self, op, valuation=None):
        pass

    @abc.abstractmethod
    def after(self, op, valuation=None):
        pass

    @abc.abstractmethod
    def evaluate(self, op, valuation=None):
        pass


class OpEvalGenerator(object):
    def generate(self, op, ctx):
        raise NotImplementedError


class FormulaFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_neg(self, operand):
        pass

    @abc.abstractmethod
    def create_exp(self, operand):
        pass

    @abc.abstractmethod
    def create_log(self, operand):
        pass

    @abc.abstractmethod
    def create_sin(self, operand):
        pass

    @abc.abstractmethod
    def create_cos(self, operand):
        pass

    @abc.abstractmethod
    def create_tan(self, operand):
        pass

    @abc.abstractmethod
    def create_add(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_sub(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_mul(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_div(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_pow(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_eq(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_gt(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_lt(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_ge(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_le(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_ne(self, op1, op2):
        pass

    @abc.abstractmethod
    def create_elementwise(self, formula):
        pass

    @abc.abstractmethod
    def create_gemm(self, A, B, C,
                    alpha=exprgraph.Constant(1.0),
                    beta=exprgraph.Constant(0.0),
                    transA=exprgraph.Constant(False),
                    transB=exprgraph.Constant(False),
                    parent=None):
        pass

    @abc.abstractmethod
    def create_gemv(self, A, X, Y, alpha=exprgraph.Constant(1.0), beta=exprgraph.Constant(0.0),
                    transA=exprgraph.Constant(False), parent=None):
        pass

    @abc.abstractmethod
    def create_ger(self, alpha, X, Y, A, parent=None):
        pass
