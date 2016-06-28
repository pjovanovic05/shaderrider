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

    def __init__(self, parent=None):
        self._parent = weakref.ref(parent) if parent is not None else None
        for p in parent:
            self._parent.append(weakref.ref(p))

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

    @abstractmethod
    def gradient(self, wrt):
        raise NotImplementedError

    @abstractmethod
    def simplify(self):
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
        raise NotImplementedError

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val):
        self._parent = val

    def is_array(self):
        raise NotImplementedError

    def is_scalar(self):
        raise NotImplementedError


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

    def simplify(self):
        return self


class Literal(Atom):                                    # TODO add dtype?
    """docstring for Literal"""
    _ctr = 0

    def __init__(self, value, name=None, parent=None):
        super(Literal, self).__init__(parent)
        # TODO check type to be primitive (literal) (int, float, bool)
        Literal._ctr += 1
        self._value = value
        self._fid = 'L' + str(Literal._ctr)
        self._name = name if name is not None else self._fid

    @property
    def fid(self):
        return self._fid

    @property
    def value(self):
        return self._value

    def is_scalar(self):
        return True

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other.value

    def __str__(self):
        return str(self._value)

    def get_variables(self):
        return []       # TODO da li ovde treba da bude [self]? da li uopste trebaju atomi ili varijable?

    def is_array(self):
        return False

    def gradient(self, wrt):
        pass    # TODO

    def get_shape(self):
        return (1,)


class Constant(Atom):
    """docstring for Constant"""
    _ctr = 0

    def __init__(self, value, name=None, parent=None):
        super(Constant, self).__init__(parent)
        Constant._ctr += 1
        self._value = np.asarray(value)
        self._fid = 'C' + str(Constant._ctr)
        self._name = name if name is not None else self._fid

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

    def get_variables(self):
        return []

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other.value

    def __str__(self):
        return str(self._value)

    def get_shape(self):
        return self._value.shape

    def is_array(self):
        return self._value.ndim != 0

    def is_scalar(self):
        return self._value.ndim == 0


class Variable(Atom):
    """docstring for Variable"""
    _ctr = 0

    def __init__(self, name=None, dtype=None, shape=None, array=None,
                 shared=False, parent=None):
        super(Variable, self).__init__(parent)
        Variable._ctr += 1
        self._fid = ('V' + str(Variable._ctr)) if name is None else name
        self._dtype = dtype
        self._shape = shape if shape is not None else ()
        self._shared = shared
        if array is not None:
            self._dtype = array.dtype
            self._shape = array.shape
            self._value = array

    @property
    def fid(self):
        return self._fid

    @property
    def name(self):
        return self._fid

    @name.setter
    def name(self, value):
        self._fid = value

    @property
    def value(self):
        return self._value

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

    def is_array(self):
        return len(self._shape) != 0

    def is_scalar(self):
        return len(self._shape) == 0


class Operator(Formula):
    """docstring for Operator"""
    __metaclass__ = ABCMeta
    _ctr = 0
    _type_name = 'Op'

    def __init__(self, arity, operands, parent=None):
        super(Operator, self).__init__(parent)
        Operator._ctr += 1
        self._arity = arity
        self._operands = operands     # formulas, operands
        self._fid = self._type_name + str(Operator._ctr)
        self._fn = None     # TODO generate_evaluator maybe?

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
    def fid(self):
        return self._fid

    @property
    def operands(self):
        return self._operands

    @property
    def arity(self):
        return self._arity

    @classmethod
    def is_broadcastable(cls):
        return False

    def is_array(self):
        return True

    def is_scalar(self):
        return False

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
