"""
WRITEME

"""

from shaderrider import configuration


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
        factory = configuration.get_platform_factory()
    return factory.create_valuation()
