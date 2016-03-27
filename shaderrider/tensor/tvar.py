"""
Variable that abstracts the gpu tensor implementation, and implements shallow embedding of the vast syntax.
"""

# TODO http://www.deeplearning.net/software/theano/library/tensor/index.html
# o da...
# U tensoru cu drzati numpy array, a funkcija koja se konstruise s njim neka brine o pomeranju
# izmedju host i device memorije.

import numbers

from shaderrider import configuration as config
from shaderrider.symbolic import exprgraph
import numpy as np


class Tensor(object):
    def __init__(self, data=None, name=None, dtype=None, shape=None, formula=None):
        if formula is not None:
            self._formula = formula
        elif data is not None:
            self._formula = exprgraph.Constant(data)
            if name is not None:
                self._formula.name = name
        else:
            self._formula = exprgraph.Atom(name=name, dtype=dtype, shape=shape)


    @property
    def formula(self):
        return self._formula

    @property
    def ndim(self):
        return len(self._formula.get_shape())   # TODO shape inference, could be slow

    @property
    def type(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._array.dtype            # TODO there's no good equivalent in exprgraph

    @property
    def T(self):
        raise NotImplemented

    def reshape(self, shape, ndim=None):
        pass

    def dimshuffle(self, *pattern):
        pass

    def flatten(self, ndim=1):
        pass

    def ravel(self):
        pass

    # reductions

    def any(self, axis=None, keepdims=False):
        pass

    def all(self, axis=None, keepdims=False):
        pass

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        pass

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        pass

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        pass

    def var(self, axis=None, keepdims=False):
        pass

    def std(self, axis=None, keepdims=False):
        pass

    def min(self, axis=None, keepdims=False):
        pass

    def max(self, axis=None, keepdims=False):
        pass

    def argmin(self, axis=None, keepdims=False):
        pass

    def argmax(self, axis=None, keepdims=False):
        pass

    def diagonal(self, offset=0, axis1=0, axis2=1):
        pass

    def take(selfindices, axis=None, mode='raise'):
        pass

    def copy(self):
        pass

    def norm(self, L, axis=None):
        pass

    def nonzero(self, return_matrix=False):
        pass

    def nonzero_values(self):
        pass

    def sort(self, axis=-1, kind='quicksort', order=None):
        pass

    def argsort(self, axis=-1, kind='quicksort', order=None):
        pass

    def clip(self, a_min, a_max):
        pass

    def conf(self):
        pass

    def repeat(self, repeats, axis=None):
        pass

    def round(self, mode='half_away_from_zero'):
        pass

    def trace(self):
        pass

    def get_scalar_constant_value(self):
        pass

    def zeros_like(self, model, dtype=None):
        pass

    # arithmetic

    def __add__(self, other):
        otherf = None
        if isinstance(other, exprgraph.Formula):
            otherf = other
        elif isinstance(other, np.ndarray) or isinstance(other, numbers.Number):
            otherf = exprgraph.Constant(other)
        else:
            raise NotImplementedError('unsupported operand type')
        # use factory to make AddOP
        newf = config.get_formula_factory().create_add(self, otherf)
        # return Tensor wrapping the AddOP
        return Tensor(formula=newf)

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __floordiv__(self, other):
        pass

    def __mod__(self, other):
        pass

    def __divmod__(self, other):
        pass

    def __pow__(self, power, modulo=None):
        pass

    def __lshift__(self, other):
        pass

    def __rshift__(self, other):
        pass

    def __and__(self, other):
        pass

    def __xor__(self, other):
        pass

    def __or__(self, other):
        pass

    def __div__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __rpow__(self, power, modulo=None):
        pass

    def __rlshift__(self, other):
        pass

    def __rrshift__(self, other):
        pass

    def __rand__(self, other):
        pass

    def __rxor__(self, other):
        pass

    def __ror__(self, other):
        pass

    def __invert__(self):
        pass

    def __pos__(self):
        pass

    def __neg__(self):
        pass

    def __abs__(self):
        pass

    # comparisons

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass


# constructors

def empty():
    pass


def zeros():
    pass


def scalar():
    pass


def vector():
    pass


def matrix():
    pass


# arithmetic functions

def neg(t1):
    pass


def exp(t1):
    pass


# linear algebra

def dot(a, b):
    pass

