"""
vast.py: Vermilion Abstract Syntax Tree
Contains the base classes for specifying symbolic expression structure.
"""

import numpy as np
import abc


class Formula(object):
    """Abstract syntax tree node base class.

    Formulas are generic interfaces for representations of symbolic expressions
    that are to be compiled for the backend (opencl, cpu).
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getAtoms(self):
        """Gets all the atomic formulas under this formula.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getAtomics(self):
        raise NotImplementedError

    @abc.abstractmethod
    def complexity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def substitute(self, a, b):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, wrt):
        raise NotImplementedError

    @abc.abstractmethod
    def simplify(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def getShape(self):
        raise NotImplementedError

    @property
    def fid(self):
        raise NotImplementedError

    def isArray(self):
        raise NotImplementedError

    def isScalar(self):
        raise NotImplementedError


class AtomicFormula(Formula):
    """docstring for AtomicFormula"""

    __metaclass__ = abc.ABCMeta

    def getAtomics(self):
        return [self]

    def complexity(self):
        return 0

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return self

    def simplify(self):
        return self


class Constant(AtomicFormula):
    """docstring for Constant"""
    _ctr = 0

    def __init__(self, value):
        Constant._ctr += 1
        self._value = np.asarray(value)
        self._fid = 'C' + str(Constant._ctr)

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, val):
        self._value = val

    @property
    def fid(self):
        return self._fid

    def gradient(self, wrt):
        return Constant(0)

    def getAtoms(self):
        return []

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other.value

    def __str__(self):
        return str(self._value)

    def getShape(self):
        return self._value.shape

    def isArray(self):
        return self._value.ndim != 0

    def isScalar(self):
        return self._value.ndim == 0


class Atom(AtomicFormula):
    """docstring for Atom"""
    _ctr = 0

    def __init__(self, name=None, dtype=None, shape=(), shared=False):
        Atom._ctr += 1
        self._name = name if name!=None else 'A%d' % Atom._ctr
        self._dtype = dtype
        self._shape = shape
        self._shared = shared
        self._fid = 'A' + str(Atom._ctr)
        # atoms should not have values, they are assigned in valuations

    @property
    def fid(self):
        return self._fid

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

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
            return Constant(0)

    def getAtoms(self):
        return [self]

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name \
               and self._shape == other._shape and self._dtype == other._dtype

    def __str__(self):
        return self._name

    def getShape(self):
        return self._shape

    def isArray(self):
        return len(self._shape) != 0

    def isScalar(self):
        return len(self._shape) == 0


class Operator(Formula):
    """docstring for Operator"""
    __metaclass__ = abc.ABCMeta
    _ctr = 0
    _type_name = 'Op'

    def __init__(self, arity, operands):
        Operator._ctr += 1
        self._arity = arity
        self._operands = operands     # formulas, operands
        self._fid = self._type_name + str(Operator._ctr)

    def complexity(self):
        c = 1
        for op in self._operands:
            c += op.complexity()
        return c

    def substitute(self, a, b):
        raise NotImplementedError

    def getAtoms(self):
        atoms = []
        for op in self._operands:
            atoms.extend(op.getAtoms())
        return atoms

    def getAtomics(self):
        atomics = []
        for op in self._operands:
            atomics.extend(op.getAtomics())
        return atomics

    def getShape(self):
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
        return '%s(%s)' % (self._fid, ', '.join([str(op) for op in  self._operands]))

    @property
    def fid(self):
        return self._fid

    @property
    def operands(self):
        return self._operands

    @property
    def arity(self):
        return self._arity

    def isArray(self):
        return True

    def isScalar(self):
        return False

    @classmethod
    def getTypeName(cls):
        return cls._type_name
