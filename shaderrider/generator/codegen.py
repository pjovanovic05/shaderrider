import abc


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
    def init(self):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass

    @abc.abstractmethod
    def before(self, op, valuation=None):
        pass

    @abc.abstractmethod
    def after(self, op, valuation=None):
        pass

    @abc.abstractmethod
    def eval(self, op, valuation=None):
        pass


class OpEvalGenerator(object):
    def generate(self, op, ctx):
        raise NotImplementedError
