"""
WRITEME

"""

from abc import ABCMeta, abstractmethod
from shaderrider.util import OrderedSet
from shaderrider.symbolic import exprgraph
from shaderrider import configuration


class PlatformFactory(object):
    """Abstract factory that defines a platform."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_platform(self):
        """
        Performs platform initialization, like context and queue creation.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_platform(self):
        """
        Finalizes the platform, closing the queues and contexts.
        """
        raise NotImplementedError

    @abstractmethod
    def create_valuation(self):
        raise NotImplementedError

    @abstractmethod
    def create_function(self, expressions=None, updates=None, name=None, skip_platform_opts=False):
        raise NotImplementedError

    @abstractmethod
    def create_op(self, type_name, operands, params):
        raise NotImplementedError

    # ARRAY CREATION
    @abstractmethod
    def empty(self, shape, dtype=None, order='C', name=None):
        raise NotImplementedError

    @abstractmethod
    def empty_like(self, a, dtype=None, order='C', name=None):
        raise NotImplementedError

    @abstractmethod
    def eye(self, N, M=0, k=0, dtype=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def identity(self, N, dtype=None, const=False, name=None):                      # TODO do we need this?
        raise NotImplementedError

    @abstractmethod
    def ones(self, shape, dtype=None, order='C', const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def ones_like(self, a, dtype=None, order='C', const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def from_data(self):                    # TODO parameters?
        raise NotImplementedError

    @abstractmethod
    def arange(self, start, stop, step=None, dtype=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def linspace(self, start, stop, num=None, endpoint=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def logspace(self, start, stop, num, endpoint, base, const=False, name=None):
        raise NotImplementedError


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

    def __init__(self, expressions=None, updates=None, name=None):
        self._name = name if name is not None else 'f' + str(Function._ctr)
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
    platform = configuration.get_platform_factory()

    # TODO optimizations?

    fn = platform.create_function(expressions, updates, name)
    return fn


def valuation(shared=None, temps=None, platform=None):
    factory = None
    if platform is not None:
        factory = configuration.platforms[platform]
    else:
        factory = configuration.get_factory()
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
