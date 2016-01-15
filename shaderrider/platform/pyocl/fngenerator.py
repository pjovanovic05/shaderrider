"""
Generates callable Function from expression graphs.
"""

import shaderrider.symbolic.vast as ast
import shaderrider.generator.function as fn


class PyOCLFunction(fn.Function):
    def __init__(self, expressions=None, updates=None, name=None):
        super(PyOCLFunction, self).__init__(expressions, updates, name)
        self._ops = []
        self._tops = []
        self._evaluators = []
        # self.
        if expressions is None and updates is None:
            raise ValueError(
                "Can't create a function for doing nothing. Provide some expressions or updates to execute.")

        for expr in self._expressions:
            ts = fn.topsort_formula(expr)
            # generate evaluators
            #

        for update in self._updates:
            pass

    def __call__(self, valuation=None, check_inputs=True):
        pass


def make_function():
    pass


class PyOCLGenerator(object):
    def __init__(self):
        self._opevals = {}
        self._isEvaluator = True
