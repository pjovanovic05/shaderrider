"""
Numpy based execution platform
"""
from numbers import Number

import numpy as np

from shaderrider.symbolic import exprgraph
from shaderrider.generator.function import Function, Valuation, PlatformFactory


class NPValuation(Valuation):
    def add(self, name, value):
        if isinstance(value, np.ndarray):
            self._vars[name] = value
        elif isinstance(value, Number):
            self._vars[name] = value
        elif isinstance(value, exprgraph.Atom):
            self._vars[name] = value.value
        else:
            raise TypeError        # TODO raise unsupported type or something

    def add_shared(self, name, value):
        if isinstance(value, np.ndarray):
            self._shared[name] = value
        elif isinstance(value, Number):
            self._shared[name] = value
        elif isinstance(value, exprgraph.Atom):
            self._shared[name] = value.value
        else:
            raise TypeError  # TODO raise unsupported type or something

    # get stays the same...
    # def get(self, name):
    #     pass

    def set(self, name, value):
        pass


class NPFunction(Function):
    def __init__(self, inputs=None, expressions=None, updates=None, name=None):
        pass

    def evaluate(self, valuation):
        pass


class NPFactory(PlatformFactory):
    pass
