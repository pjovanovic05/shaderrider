import abc
from shaderrider.util import OrderedSet
from shaderrider.symbolic import exprgraph
from shaderrider import configuration


class Function(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, inputs=None, expressions=None, updates=None, name=None):
        self._inputs = inputs
        self._expressions = expressions
        self._updates = updates
        self._name = name

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        :param valuation:
        :type valuation: dict

        :param check_inputs:
        :type check_inputs: bool
        """
        raise NotImplementedError


def function(expressions=None, updates=None, name=None):
    """
    TODO Creates callable object that performs calculation described by the
    computation graph(s) for the expression(s) to be calculated.

    The expressions are computed first, then the updates are performed, and then

    :param expressions: list of expression graphs to evaluate, whose outputs will be function outputs
    :type expressions: list

    :param updates: shared var as key, expression that updates it as value
    :type updates: dict

    :param name: debug name for the generated function object
    :type name: str
    """

    # configure compilation
    platform = configuration.get_platform()
    checks = platform.get_validations()
    opts = platform.get_optimizations()

    # collect inputs
    inputs = _collect_inputs(expressions, updates)

    # for each expression
    for expr in expressions:
    #   run checks
        for check in checks:
            pass    #validate graph
    #   optimizations
        for opt in opts:
            pass    #optimize graph

    # for each update
    for update in updates:
    #   run checks, maybe opts?
        pass

    # create appropriate Function instance
    fn = platform.create_function(inputs, expressions, updates, name)
    return fn


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
        if isinstance(node, exprgraph.AtomicFormula):
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


def validate_formula(formula):
    # TODO arity checks, type checks, shape checks, dtype checks?
    pass


def _collect_inputs(expressions=None, updates=None):        # TODO gde ide collect outputs?
    inputs = OrderedSet()
    for expr in expressions:
        for a in expr.get_atoms():
            if a not in inputs:     # is this necessary?
                inputs.add(a)
    for var, update in updates:
        for a in update.get_atoms():
            if a not in inputs:
                inputs.add(a)
    return list(inputs)
