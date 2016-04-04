"""
All the operator types.

WRITEME
"""
import abc

import shaderrider.configuration as config
from shaderrider.core import IncompatibleDimensionsError, NondifferentiableOpError
from shaderrider.generator.codegen import FormulaFactory            # TODO gradient and substitutions need to use this
from shaderrider.symbolic import exprgraph

# TOC
#  - abstract ops
#  - tensor ops
#  - arithmetic ops
#  - elementwise op
#  - scan ops
#  - blas ops
#  - convolution ops


# ABSTRACT OPS ########################################################

class UnaryOP(exprgraph.Operator):
    """docstring for UnaryOP"""
    __metaclass__ = abc.ABCMeta

    def __eq__(self, other):
        return type(self) == type(other) and self.operands[0] == other.operands[0]

    def get_shape(self):
        return self.operands[0].get_shape()


class BinaryOP(exprgraph.Operator):
    """docstring for BinaryOP"""
    __metaclass__ = abc.ABCMeta
    isCommutative = False
    isAssociative = False

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.operands[0] == other.operands[0]) and \
               (self.operands[1] == other.operands[1])

    def get_shape(self):  # TODO move this to broadcastable or elementwise
        ds = []
        for d1, d2 in zip(self.operands[0].get_shape(), self.operands[1].get_shape()):
            if d1 == 1 or d2 == 1 or d1 == d2:
                ds.append(max(d1, d2))
            else:
                # TODO print offending dimensions to exception message
                raise IncompatibleDimensionsError
        return tuple(ds)


# TENSOR OPS ##########################################################

class ReshapeOP(exprgraph.Operator):
    _type_name = 'Reshape'

    def __init__(self, arr, shape, parent=None):
        super(ReshapeOP, self).__init__(2, [arr, shape], parent)
        assert isinstance(shape, exprgraph.Constant) or isinstance(shape, tuple)
        # assert isinstance(arr, exprgraph.Atom)  # da li???  IPAK NE

        self._shape = shape if isinstance(shape, tuple) else None   # TODO extract shape
        # TODO check transformation compatibility
        # multiply shape components and see if the lengths match current length

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_reshape(self.operands[0].substitute(a,b), self._shape)

    def get_shape(self):
        return self._shape

    def simplify(self):
        ff = config.get_formula_factory()
        return ff.create_reshape(self.operands[0].simplify(), self._shape)


class IndexOP(exprgraph.Operator):
    _type_name = 'Index'
    # TODO proveri kako se radi ono advanced i basic indeksiranje u Theanou i sta od toga moze u pyopenclu.
    
    def __init__(self, op, key, parent=None):
        """
        WRITEME

        :param key an integer or slice object wrapped in a exprgraph.Constant
        """
        super(IndexOP, self).__init__(1, [op, key], parent)
        self._key = key

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_index()

    def simplify(self):
        ff = config.get_formula_factory()
        return ff.create_index(self.operands[0].simplify(), self._key, self._parent)

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def get_shape(self):
        pass    # TODO calculate size and shape of the result if possible?


class TransposeOP(UnaryOP):
    _type_name = 'Transpose'

    def __init__(self, op, axes=None, parent=None):
        super(TransposeOP, self).__init__(1, [op], parent)
        self._axes = axes

    def simplify(self):
        # some ideas:
        # if op is also a transpose on the same axes, cancel out
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_transpose(self.operands[0].substitute(a, b))


class DimshuffleOP(exprgraph.Operator):
    _type_name = 'Dimshuffle'

    def __init__(self, op, new_dims, parent=None):
        super(DimshuffleOP, self).__init__(2, [op, new_dims], parent)
        self._new_dims = new_dims

    def simplify(self):
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_dimshuffle(self.operands[0].substitute(a, b), self._new_dims)

    def get_shape(self):
        return self._new_dims


class RavelOP(UnaryOP):
    _type_name = 'Ravel'

    def __init__(self, op, parent=None):
        super(RavelOP, self).__init__(1, [op], parent)

    def simplify(self):
        # idea: if op is already a vector just omit this operator.
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_ravel(self.operands[0].substitute(a, b))

    def get_shape(self):
        return (sum(self.operands[0].get_shape()),)


class DiagonalOP(exprgraph.Operator):
    _type_name = 'Diagonal'

    def __init__(self, op, parent=None):
        super(DiagonalOP, self).__init__(1, [op], parent)

    def simplify(self):
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_diagonal(self.operands[0].substitute(a, b)) # TODO parent?

    def get_shape(self):
        pass


class TraceOP(exprgraph.Operator):
    _type_name = 'Trace'

    def __init__(self, op, parent=None):
        super(TraceOP, self).__init__(1, [op], parent)

    def simplify(self):
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_trace(self.operands[0].substitute(a, b))

    def get_shape(self):
        return (1,)


class NormOP(exprgraph.Operator):
    _type_name = 'Norm'

    def __init__(self, op, norm, parent=None):
        super(NormOP, self).__init__(2, [op, norm], parent)

    def simplify(self):
        pass

    def gradient(self, wrt):
        pass

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_norm(self.operands[0].substitute(a, b), self.operands[1])

    def get_shape(self):
        pass


# ARITHMETIC OPS ######################################################

class AbsOP(UnaryOP):
    _type_name = 'Abs'

    def __init__(self, op, parent):
        super(AbsOP, self).__init__(1, [op], parent)

    def simplify(self):
        # if operand is also abs, colapse it
        # if operand is a constant, colapse this into a constant
        pass

    def gradient(self, wrt):
        pass # <0 : -1, 0: 0, >0: 1

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_abs(self.operands[0].substitute(a, b)) # parent?

class NegOP(UnaryOP):
    _type_name = "Neg"

    def __init__(self, operand):
        super(NegOP, self).__init__(1, [operand])

    def __str__(self):
        return '-(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return NegOP(self.operands[0].substitute(a, b))         # FIXME use factory!!

    def gradient(self, wrt):
        return NegOP(self.operands[0].gradient(wrt))            # FIXME use factory!!

    def simplify(self):
        simp_op = self.operands[0].simplify()
        if type(simp_op) == type(self):
            return simp_op.operands[0]
        return NegOP(simp_op)                                   # FIXME use factory!!


class ExpOP(UnaryOP):
    _type_name = "Exp"

    def __init__(self, operand):
        super(ExpOP, self).__init__(1, [operand])

    def __str__(self):
        return 'exp(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return ExpOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        return ExpOP(self.operands[0])

    def simplify(self):
        simp_op = self.operands[0].simplify()
                                                                # TODO wtf?
        if isinstance(simp_op, ExpOP):
            return simp_op.operands[0]

        raise NotImplementedError


class LogOP(UnaryOP):
    _type_name = "Log"

    def __init__(self, operand):
        super(LogOP, self).__init__(1, [operand])

    def __str__(self):
        return 'log(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return LogOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class SinOP(UnaryOP):
    _type_name = "Sin"

    def __init__(self, operand):
        super(SinOP, self).__init__(1, [operand])

    def __str__(self):
        return 'sin(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return SinOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class CosOP(UnaryOP):
    _type_name = "Cos"

    def __init__(self, operand):
        super(CosOP, self).__init__(1, [operand])

    def __str__(self):
        return "cos(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return CosOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class CoshOP(UnaryOP):
    pass


class TanOP(UnaryOP):
    _type_name = "Tan"

    def __init__(self, operand):
        super(TanOP, self).__init__(1, [operand])

    def __str__(self):
        return "tan(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return TanOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class SignOP(UnaryOP):
    pass


class CeilOP(UnaryOP):
    pass


class FloorOP(UnaryOP):
    pass


class RoundOP(UnaryOP):
    pass


class SqrtOP(UnaryOP):
    pass


class MaximumOP(BinaryOP):
    pass


class MinimumOP(BinaryOP):
    pass


# ELEMENTWISE OP ######################################################

class ElementwiseOP(exprgraph.Operator):
    _type_name = 'Elwise'

    def __init__(self, expr, ops, parent=None):
        super(ElementwiseOP, self).__init__('Elwise', ops, parent=parent)

    def substitute(self, a, b):
        # TODO should this be supported?
        pass

    def get_shape(self):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


# SCAN OPS ############################################################

# sum, max, min, mean, std,


# BLAS OPS ############################################################

class GemmOP(exprgraph.Operator):
    _type_name = "Gemm"

    def __init__(self, A, B, C, alpha=exprgraph.Constant(1.0), beta=exprgraph.Constant(0.0),
                 transA=exprgraph.Constant(False), transB=exprgraph.Constant(False), parent=None):
        super(GemmOP, self).__init__(7, [A, B, C, alpha, beta, transA, transB], parent)

        if len(A.get_shape()) != 2 or len(B.get_shape()) != 2 or len(C.get_shape()) != 2:
            raise ValueError
            # TODO

    def substitute(self, a, b):
        if self == a:
            return b
        return GemmOP(self.operands[0].substitute(a, b),
                      self.operands[1].substitute(a, b),
                      self.operands[2].substitute(a, b),
                      self.operands[3],
                      self.operands[4],
                      self.operands[5],
                      self.operands[6])

    def __str__(self):
        fstr = 'GEMM(%f * %s'
        if self.operands[5].value:  # transA
            fstr += "'"
        fstr += " * %s"
        if self.operands[6].value:  # transB
            fstr += "'"
        fstr += " + %f * %s)"
        return fstr % map(str, (self.operands[3].value, self.operands[0].name,
                                self.operands[1].name, self.operands[4].value, self.operands[2].name))

    def get_shape(self):
        # TODO
        pass

    def gradient(self, wrt):
        raise NotImplementedError

    def simplify(self):
        return GemmOP(self.operands[0].simplify(),
                      self.operands[1].simplify(),
                      self.operands[2].simplify(),
                      self.operands[3].simplify(),
                      self.operands[4].simplify(),
                      self.operands[5].simplify(),
                      self.operands[6].simplify())


class GemvOP(exprgraph.Operator):
    _type_name = "Gemv"

    def __init__(self, A, X, Y, alpha=exprgraph.Constant(1.0), beta=exprgraph.Constant(0.0),
                 transA=exprgraph.Constant(False), parent=None):
        super(GemvOP, self).__init__(6, [A, X, Y, alpha, beta, transA], parent)
        # TODO check dimensions compatibility

    def substitute(self, a, b):
        if self == a:
            return b
        return GemvOP(self.operands[0].substitute(a, b),
                      self.operands[1].substitute(a, b),
                      self.operands[2].substitute(a, b),
                      self.operands[3],
                      self.operands[4],
                      self.operands[5])

    def __str__(self):
        fstr = 'GEMV(%f * %s'
        if self.operands[5].value:
            fstr += "'"
        fstr += ' * %s + %f * %s)'
        return fstr % map(str, (self.operands[3], self.operands[0], self.operands[1],
                                self.operands[4], self.operands[2]))

    def gradient(self, wrt):
        pass

    def get_shape(self):
        pass

    def simplify(self):
        pass


class GerOP(exprgraph.Operator):
    _type_name = "Ger"

    def __init__(self, alpha, X, Y, A, parent=None):
        super(GerOP, self).__init__(4, [alpha, X, Y, A], parent)
        # TODO check dimensions

        if len(X.get_shape()) != 1 or len(Y.get_shape()) != 1 or len(A.get_shape()) != 2:
            raise ValueError
        m, n = X.get_shape()[0], Y.get_shape()[0]
        if (m, n) != A.get_shape():
            raise IncompatibleDimensionsError

    def substitute(self, a, b):
        if self == a:
            return b
        return GerOP(self.operands[0],
                     self.operands[1].substitute(a, b),
                     self.operands[2].substitute(a, b),
                     self.operands[3].substitute(a, b))

    def __str__(self):
        return "GER(%f * %s * %s' + %s)" % map(str, self.operands)

    def get_shape(self):
        return self.operands[1].get_shape()[0], self.operands[1].get_shape()[0]

    def gradient(self, wrt):
        raise NotImplementedError

    def simplify(self):
        pass


# CONVOLUTION OPS #####################################################
