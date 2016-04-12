"""
basic arithmetic operations for tensors
"""

import abc

from shaderrider.symbolic import exprgraph as ast
from shaderrider.core import IncompatibleDimensionsError


# Unary

class UnaryOP(ast.Operator):
    """docstring for UnaryOP"""
    __metaclass__ = abc.ABCMeta

    def __eq__(self, other):
        return type(self) == type(other) and self.operands[0] == other.operands[0]

    def get_shape(self):
        return self.operands[0].get_shape()


class NegOP(UnaryOP):
    _type_name = "Neg"

    def __init__(self, operand):
        super(NegOP, self).__init__(1, [operand])

    def __str__(self):
        return '-(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        return NegOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        return NegOP(self.operands[0].gradient(wrt))

    def simplify(self):
        simp_op = self.operands[0].simplify()
        if type(simp_op) == type(self):
            return simp_op.operands[0]
        return NegOP(simp_op)


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
        # TODO
        simp_op = self.operands[0].simplify()

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


# Binary ops

class BinaryOP(ast.Operator):
    """docstring for BinaryOP"""
    __metaclass__ = abc.ABCMeta
    isCommutative = False
    isAssociative = False

    def __eq__(self, other):
        return ((type(self) == type(other))
                and (self.operands[0] == other.operands[0])
                and (self.operands[1] == other.operands[1]))

    def get_shape(self):  # TODO move this to broadcastable or elementwise
        ds = []
        for d1, d2 in zip(self.operands[0].get_shape(), self.operands[1].get_shape()):
            if d1 == 1 or d2 == 1 or d1 == d2:
                ds.append(max(d1, d2))
            else:
                # TODO print offending dimensions to exception message
                raise IncompatibleDimensionsError
        return tuple(ds)


class AddOP(BinaryOP):
    _type_name = "Add"
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2):
        super(AddOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s + %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return AddOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        return AddOP(self.operands[0].gradient(wrt), self.operands[1].gradient(wrt))

    def simplify(self):
        # TODO
        raise NotImplementedError


class SubOP(BinaryOP):
    _type_name = "Sub"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(SubOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s - %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return SubOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        return SubOP(self.operands[0].gradient(wrt), self.operands[1].gradient(wrt))

    def simplify(self):
        # TODO
        raise NotImplementedError


class MulOP(BinaryOP):
    _type_name = "Mul"
    isCommutative = True
    isAssociative = False

    def __init__(self, op1, op2):
        super(MulOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s * %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return MulOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        return AddOP(MulOP(self.operands[0].gradient(wrt), self.operands[1]),
                     MulOP(self.operands[0], self.operands[1].gradient(wrt)))

    def simplify(self):
        # TODO
        raise NotImplementedError


class DivOP(BinaryOP):
    _type_name = "Div"
    isCommutative = False
    isAssociative = False   # rly?

    def __init__(self, op1, op2):
        super(DivOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s / %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return DivOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        return DivOP(SubOP(MulOP(self.operands[0].gradient(wrt), self.operands[1]),
                           MulOP(self.operands[0], self.operands[1].gradient(wrt))),
                     PowOP(self.operands[1], ast.Constant(2)))

    def simplify(self):
        # TODO
        raise NotImplementedError


class PowOP(BinaryOP):
    _type_name = "Pow"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(PowOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s ^ %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return PowOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO treba ln za generalnu verziju...
        pass

    def simplify(self):
        # TODO
        raise NotImplementedError


# Comparisons

class EqOP(BinaryOP):
    _type_name = "Eq"
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2):
        super(EqOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s == %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return EqOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return EqOP(simp_op1, simp_op2)


class GtOP(BinaryOP):
    _type_name = "Gt"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(GtOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s > %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return GtOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return GtOP(simp_op1, simp_op2)


class LtOP(BinaryOP):
    _type_name = "Lt"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(LtOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s < %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return LtOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return GtOP(simp_op1, simp_op2)


class GeOP(BinaryOP):
    _type_name = "Ge"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(GeOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s >= %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return GeOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return GeOP(simp_op1, simp_op2)


class LeOP(BinaryOP):
    _type_name = "Le"
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2):
        super(LeOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s <= %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return LeOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return LeOP(simp_op1, simp_op2)


class NeOP(BinaryOP):
    _type_name = "Ne"
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2):
        super(NeOP, self).__init__(2, [op1, op2])

    def __str__(self):
        return '(%s != %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        return NeOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        pass

    def simplify(self):
        simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        return NeOP(simp_op1, simp_op2)


# Tensor ops...

class ReshapeOP(ast.Operator):
    pass


class IndexOP(ast.Operator):
    pass


class TransposeOP(ast.Operator):
    pass
