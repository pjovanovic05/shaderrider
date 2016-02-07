import abc
from shaderrider.symbolic import exprgraph


class Function(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, expressions=None, updates=None, name=None):
        self._expressions = expressions
        self._updates = updates
        self._name = name

    @abc.abstractmethod
    def __call__(self, valuation=None, check_inputs=True):
        """
        :param valuation:
        :type valuation: dict

        :param check_inputs:
        :type check_inputs: bool
        """
        raise NotImplementedError

    def _create_evaluation_path(self):
        self._expr_evals = []
        # for expressions
        for expr in self._expressions:
            pass
        # for updates


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

    # check graph
    # optimize (simplify +)
    # TODO get current platform generator
    # generate source
    # configure compiler
    # compile
    # import
    pass


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
