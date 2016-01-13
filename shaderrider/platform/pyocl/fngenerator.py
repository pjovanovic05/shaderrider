"""
Generates callable Function from expression graphs.
"""

import shaderrider.symbolic.vast as ast
import shaderrider.generator.function as fn


class PyOCLFunction(fn.Function):
    def __init__(self, expr, updates=None, name=None):
        super(PyOCLFunction, self).__init__(expr, updates, name)
        self._ops = []
        self._tops = []
        self._evaluators = []
        # self.

    def __call__(self, valuation=None, check_inputs=True):
        pass


def make_function():
    pass