"""
exprgraph.py: Abstract Syntax Tree.

Contains the base classes for specifying symbolic expression structure.
"""

import numpy as np
from abc import ABCMeta, abstractproperty, abstractmethod
import weakref


class Formula(object):
    """Abstract syntax tree node base class.

    Formulas are generic interfaces for representations of symbolic expressions
    that are to be compiled for the backend (opencl, cpu).
    """

    __metaclass__ = ABCMeta

    def __init__(self, parents=None):
        self._parent = weakref.ref(parents) if parents is not None else None
        self._parents = weakref.WeakSet(parents)
        self._fid = None

    @abstractmethod
    def get_variables(self):
        """Get all the variables under this formula."""
        raise NotImplementedError

    @abstractmethod
    def get_atoms(self):
        raise NotImplementedError

    @abstractmethod
    def complexity(self):
        raise NotImplementedError

    @abstractmethod
    def substitute(self, a, b):
        raise NotImplementedError

    def evaluate(self, valuation):
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, valuation, cache):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, wrt):
        raise NotImplementedError

    def forward_grad(self, wrt, valuation):
        # TODO
        cache = {}
        # self.eval(valuation, cache)...
        return self._forward_grad(wrt, valuation, cache)

    def _forward_grad(self, wrt, valuation, cache):
        pass

    def reverse_grad(self, adjoint, grad, cache):           # TODO da li mi ovde treba i valuation?
        raise NotImplementedError

    @abstractmethod
    def _reverse_grad(self, valuation, adjoint, grad, cache):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        raise NotImplementedError

    @property
    def fid(self):
        return self._fid

    @fid.setter
    def fid(self, value):
        self._fid = value

    @property
    def parents(self):
        return self._parents


class Atom(Formula):
    """docstring for Atom"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def value(self):
        pass

    def get_atoms(self):
        return [self]

    def complexity(self):
        return 0

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return self


class Constant(Atom):
    """docstring for Constant"""
    _ctr = 0

    def __init__(self, value, name=None, parents=None):
        super(Constant, self).__init__(parents)
        Constant._ctr += 1
        self._value = np.asarray(value)
        self._fid = 'C' + str(Constant._ctr)
        self._name = name if name is not None else self._fid

    @property
    def value(self):
        return self._value

    # @value.setter
    # def value(self, val):
    #     self._value = val

    def _evaluate(self, valuation, cache):
        cache[id(self)] = self._value
        return self._value

    def gradient(self, wrt):
        return Constant(0)

    def _forward_grad(self, wrt, valuation, cache):
        return 0

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        pass

    def get_variables(self):
        return []

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other.value

    def __str__(self):
        return str(self._value)

    def get_shape(self):
        return self._value.shape


class Variable(Atom):
    """docstring for Variable"""
    _ctr = 0

    def __init__(self, name=None, dtype=None, shape=None, array=None, grad_flag=False,
                 shared=False, parents=None):
        super(Variable, self).__init__(parents)
        Variable._ctr += 1
        self._fid = ('V' + str(Variable._ctr)) if name is None else name
        self._dtype = dtype
        self._shape = shape if shape is not None else ()
        self._shared = shared
        self._value = None
        if array is not None:
            self._dtype = array.dtype
            self._shape = array.shape
            self._value = array

    @property
    def name(self):
        return self._fid

    @name.setter
    def name(self, value):
        self._fid = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    def gradient(self, wrt):
        if self == wrt:
            return Constant(1)
        else:
            return Constant(0)  # TODO is it?

    def _reverse_grad(self, valuation, adjoint, grad, cache):
        grad[self.name] += adjoint

    def get_variables(self):
        return [self]

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name \
               and self._shape == other.get_shape() \
               and self._dtype == other.dtype

    def __str__(self):
        return self._fid

    def get_shape(self):
        return self._shape


class Operator(Formula):
    """docstring for Operator"""
    __metaclass__ = ABCMeta
    _ctr = 0
    _type_name = 'Op'

    def __init__(self, arity, operands, parents=None):
        super(Operator, self).__init__(parents)
        Operator._ctr += 1
        self._arity = arity
        self._operands = operands
        for op in self._operands:
            op.parents.add(self)
        self._fid = self._type_name + str(Operator._ctr)
        self._fn = None     # TODO generate_evaluator maybe?
        self._params = {}

    def complexity(self):
        c = 1
        for op in self._operands:
            c += op.complexity()
        return c

    def substitute(self, a, b):
        raise NotImplementedError

    def get_variables(self):
        atoms = []
        for op in self._operands:
            atoms.extend(op.get_variables())
        return atoms

    def get_atoms(self):
        atomics = []
        for op in self._operands:
            atomics.extend(op.get_atoms())
        return atomics

    def get_shape(self):
        raise NotImplementedError

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if len(self._operands) != len(other.operands):
            return False
        for i in xrange(len(self._operands)):
            if self._operands[i] != other.operands[i]:
                return False
        return True

    def __str__(self):
        return '%s(%s)' % (self._fid,
                           ', '.join([str(op) for op in self._operands]))

    @property
    def operands(self):
        return self._operands

    @property
    def params(self):
        return self._params

    @property
    def arity(self):
        return self._arity

    @classmethod
    def get_type_name(cls):
        return cls._type_name

    def evaluate(self, valuation):
        if self._fn is None:
            self._fn = self.generate_eval()
        return self._fn(valuation)

    def generate_eval(self):
        """generates a zero argument function which executes the computation
        steps represented by this operator."""
        raise NotImplementedError

