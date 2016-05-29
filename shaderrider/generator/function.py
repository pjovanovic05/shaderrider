"""
WRITEME

"""

from abc import ABCMeta, abstractmethod
from shaderrider.util import OrderedSet
from shaderrider.symbolic import exprgraph
from shaderrider import configuration as config


class PlatformFactory(object):
    """Abstract factory that defines a platform."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_platform(self):
        """
        Performs platform initialization, like context and queue creation.
        """
        pass

    @abstractmethod
    def finalize_platform(self):
        """
        Finalizes the platform, closing the queues and contexts.
        """
        pass

    @abstractmethod
    def create_valuation(self):
        pass

    @abstractmethod
    def create_function(self):
        pass

    @abstractmethod
    def create_op(self, type_name, operands):
        pass

    # TODO factory methods for each op type
    # ARRAY CREATION
    # ??? TODO do i need this? valuation creates my arrays, but what about initialization (zeros, ones, random...)?

    # ARRAY MANIPULATION

    @abstractmethod
    def create_reshape(self, a, newshape):
        pass

    @abstractmethod
    def create_ravel(self, a):
        pass

    @abstractmethod
    def create_concatenate(self, a1, a2):
        pass    # TODO think this through

    @abstractmethod
    def create_stack(self, xs, axis):
        pass    # TODO think this through

    @abstractmethod
    def create_split(self, a, indicies):
        pass    # TODO think this through

    @abstractmethod
    def create_repeat(self, a, repeats, axis):
        pass    # TODO think this through

    # BINARY OPERATIONS

    @abstractmethod
    def create_bitwise_and(self, x1, x2):
        pass

    @abstractmethod
    def create_bitwise_or(self, x1, x2):
        pass

    @abstractmethod
    def create_bitwise_xor(self, x1, x2):
        pass

    @abstractmethod
    def create_invert(self, x):
        pass

    @abstractmethod
    def create_left_shift(self, x1, x2):
        pass

    @abstractmethod
    def create_right_shift(self, x1, x2):
        pass

    # INDEXING OPS
    # TODO

    # LINEAR ALGEBRA

    @abstractmethod
    def create_dot(self, a, b):
        pass

    @abstractmethod
    def create_vdot(self, a, b):
        pass

    @abstractmethod
    def create_inner(self, a, b):
        pass

    @abstractmethod
    def create_outer(self, a, b):
        pass

    @abstractmethod
    def create_matmul(self, a, b):
        pass

    @abstractmethod
    def create_eig(self, a):
        pass

    @abstractmethod
    def create_eigvals(self, a):
        pass

    # LOGIC OPS

    @abstractmethod
    def create_all(self, a):
        pass

    @abstractmethod
    def create_any(self, a):
        pass

    @abstractmethod
    def create_and(self, a, b):
        pass

    @abstractmethod
    def create_or(self, a, b):
        pass

    @abstractmethod
    def create_not(self, a):
        pass

    @abstractmethod
    def create_xor(self, a, b):
        pass

    @abstractmethod
    def create_greater(self, a, b):
        pass

    @abstractmethod
    def create_less(self, a, b):
        pass

    @abstractmethod
    def create_greater_equal(self, a, b):
        pass

    @abstractmethod
    def create_less_equal(self, a, b):
        pass

    @abstractmethod
    def create_equal(self, a, b):
        pass

    @abstractmethod
    def create_not_equal(self, a, b):
        pass

    # MATHEMATICAL OPS

    @abstractmethod
    def create_sin(self, x):
        pass

    @abstractmethod
    def create_cos(self, x):
        pass

    @abstractmethod
    def create_tan(self, x):
        pass

    @abstractmethod
    def create_arcsin(self, x):
        pass

    @abstractmethod
    def create_arccos(self, x):
        pass

    @abstractmethod
    def create_arctan(self, x):
        pass

    @abstractmethod
    def create_sinh(self, x):
        pass

    @abstractmethod
    def create_cosh(self, x):
        pass

    @abstractmethod
    def create_tanh(self, x):
        pass

    @abstractmethod
    def create_arcsinh(self, x):
        pass

    @abstractmethod
    def create_arccosh(self, x):
        pass

    @abstractmethod
    def create_arctanh(self, x):
        pass

    @abstractmethod
    def create_round(self, a, decimals=None, out=None):
        pass

    @abstractmethod
    def create_floor(self, x, out=None):
        pass

    @abstractmethod
    def create_ceil(self, x, out=None):
        pass

    @abstractmethod
    def create_prod(self, a, axis=None, dtype=None, out=None, keepdims=None):
        pass

    @abstractmethod
    def create_sum(self, a, axis=None, dtype=None, out=None, keepdims=None):
        pass

    @abstractmethod
    def create_nansum(self, a, axis=None, dtype=None, out=None, keepdims=None):
        pass

    @abstractmethod
    def create_cumprod(self, a, axis, dtype, out, keepdims):
        pass

    @abstractmethod
    def create_cumsum(self, a, axis, dtype, out, keepdims):
        pass

    @abstractmethod
    def create_exp(self, x):
        pass

    @abstractmethod
    def create_exp2(self, x, out=None):
        pass

    @abstractmethod
    def create_log(self, x, out=None):
        pass

    @abstractmethod
    def create_log10(self, x, out=None):
        pass

    @abstractmethod
    def create_log2(self, x, out=None):
        pass

    @abstractmethod
    def create_log1p(self, x, out=None):
        pass

    @abstractmethod
    def create_add(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_reciprocal(self, x, out=None):
        pass

    @abstractmethod
    def create_negative(self, x, out=None):
        pass

    @abstractmethod
    def create_multiply(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_divide(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_power(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_subtract(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_true_divide(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_floor_divide(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_mod(self, x1, x2, out=None):
        pass

    # @abstractmethod
    # def create_convolve(self, a, v, mode=None):
    #     pass

    @abstractmethod
    def create_clip(self, a, a_min, a_max, out=None):
        pass

    @abstractmethod
    def create_sqrt(self, x, out=None):
        pass

    @abstractmethod
    def create_square(self, x, out=None):
        pass

    @abstractmethod
    def create_absolute(self, x, out=None):
        pass

    @abstractmethod
    def create_sign(self, x, out=None):
        pass

    @abstractmethod
    def create_maximum(self, x1, x2, out=None):
        pass

    @abstractmethod
    def create_minimum(self, x1, x2, out=None):
        pass

    # @abstractmethod
    # def create_nan_to_num(self, x):
    #     pass

    # STATISTICS OPS

    @abstractmethod
    def create_median(self, a): #[axis, out, overwrite_input, keepdims]
        pass

    @abstractmethod
    def create_average(self, a):    #[axis, weights, returned]
        pass

    @abstractmethod
    def create_mean(self, a):   #[axis, dtype, out, keepdims]
        pass

    @abstractmethod
    def create_std(self, a):    #[axis, dtype, out, ddof, keepdims]
        pass

    @abstractmethod
    def create_var(self, a):    #[axis, dtype, out, ddof, keepdims]
        pass

    @abstractmethod
    def create_correlate(self, a, v, mode=None):
        pass

    @abstractmethod
    def create_cov(self, m):    #[y, rowvar, bias, ddof, fweights, ...]
        pass


class Valuation(object):
    """
    Holds the computation state from inputs to outputs and intermediate results in between.
    """
    def __init__(self):
        self._shared = {}
        self._vars = {}
        self._events = {}

    @property
    def events(self):
        return self._events

    def add(self, name, value):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')
        self._vars[name] = value

    def add_shared(self, name, value):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')
        self._shared[name] = value

    def __contains__(self, item):
        return (item in self._vars) or (item in self._shared)

    def get(self, name):
        if name in self._shared:
            return self._shared[name]
        elif name in self._vars:
            return self._vars[name]
        else:
            raise KeyError('Variable "' + name + '" not found in this valuation.')

    def set(self, name, value):
        if name in self._shared:
            self._shared[name] = value
        elif name in self._vars:
            self._shared[name] = value
        else:
            raise KeyError('Variable "' + name + '" not found in this valuation.')

    def clear(self):
        self._vars.clear()

    def remove(self, name):
        del self._vars[name]
        del self._shared[name]


class Function(object):
    __metaclass__ = ABCMeta
    _ctr = 0

    def __init__(self, inputs=None, expressions=None, updates=None, name=None): # TODO name could be mandatory
        self._inputs = inputs
        self._expressions = expressions
        self._updates = updates
        self._name = name if name is not None else 'f'+str(Function._ctr)
        Function._ctr += 1

    def __call__(self, *args, **kwargs):
        """
        :param valuation:
        :type valuation: Valuation
        """
        assert 'valuation' in kwargs
        # TODO umotavanje parametara u valuation bi moglo ovde da se desi?

        return self.evaluate(kwargs['valuation'])

    @abstractmethod
    def evaluate(self, valuation):
        raise NotImplementedError


def function(expressions=None, updates=None, name=None, skip_opts=False,
             skip_symbolic_opts=False, skip_platform_opts=False):
    """
    TODO Creates callable object that performs calculation described by the
    computation graph(s) for the expression(s) to be calculated.

    The expressions are computed first, then the updates are performed, and then ????

    :param expressions: list of expression graphs to evaluate, whose outputs will be function outputs
    :type expressions: list

    :param updates: shared var as key, expression that updates it as value
    :type updates: dict

    :param name: debug name for the generated function object
    :type name: str

    :rtype: Function
    """

    # configure compilation
    platform = config.get_platform()

    # collect inputs
    inputs = _collect_inputs(expressions, updates)

    # for each expression
    for expr in expressions:
        pass

    # for each update
    for update in updates:
        # run checks, maybe opts?
        pass

    # create appropriate Function instance
    fn = platform.create_function(inputs, expressions, updates, name)
    return fn


def valuation(shared=None, temps=None, platform=None):
    factory = None
    if platform is not None:
        factory = config.platforms[platform]
    else:
        factory = config.get_factory()
    return factory.create_valuation(shared, temps)


def topsort_formula(formula):
    """Topological sorting of formula expression tree.

    Returns the list of operators (and atomics) in order they are applied,
    bottom up, left to right (i.e. post order traversal).

    :rtype: list of formula nodes in topological order (post order traversal).
    :param formula: Formula
    """
    outlist = []
    treestack = [formula]
    while len(treestack) > 0:
        node = treestack.pop()
        if isinstance(node, exprgraph.Atom):
            outlist.append(node)
        elif not hasattr(node, '_visited'):   # expand node & return to stack
            node._visited = True
            treestack.append(node)
            for op in node.operands:
                treestack.append(op)
        else:
            if hasattr(node, '_visited'):
                del node._visited   # TODO hackish...
            outlist.append(node)
    return outlist


def _validate_formula(formula):
    # TODO arity checks, type checks, shape checks, dtype checks?
    pass


def _collect_inputs(expressions=None, updates=None):        # TODO gde ide collect outputs?
    inputs = OrderedSet()
    for expr in expressions:
        for a in expr.get_variables():
            if a not in inputs:     # is this necessary?
                inputs.add(a)
    for var, update in updates:
        for a in update.get_variables():
            if a not in inputs:
                inputs.add(a)
    return list(inputs)
