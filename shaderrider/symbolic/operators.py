"""
All the operator types.

WRITEME
"""
import abc

from shaderrider.core import IncompatibleDimensionsError
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
    pass


class IndexOP(exprgraph.Operator):
    pass


class TransposeOP(exprgraph.Operator):
    pass


class DimshuffleOP(exprgraph.Operator):
    pass


class RavelOP(exprgraph.Operator):
    pass


class DiagonalOP(exprgraph.Operator):
    pass


class TraceOP(exprgraph.Operator):
    pass


class NormOP(exprgraph.Operator):
    pass


# ARITHMETIC OPS ######################################################

class AbsOP(UnaryOP):
    pass

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


# BLAS OPS ############################################################


# CONVOLUTION OPS #####################################################
