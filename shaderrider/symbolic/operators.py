"""
All the operator types.

WRITEME
"""
import abc


from shaderrider.core import IncompatibleDimensionsError, NondifferentiableOpError
from shaderrider.symbolic import exprgraph


# ABSTRACT OPS ########################################################

class UnaryOP(exprgraph.Operator):
    """docstring for UnaryOP"""
    __metaclass__ = abc.ABCMeta

    def __eq__(self, other):
        return type(self) == type(other) and \
               self.operands[0] == other.operands[0]

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

    def __init__(self, arr, shape, parents=None):
        super(ReshapeOP, self).__init__(1, [arr], parents)

        assert isinstance(shape, exprgraph.Constant) or \
               isinstance(shape, tuple)

        # check transformation compatibility
        a1 = reduce(lambda x, y: x * y, shape, 1)
        a2 = reduce(lambda x, y: x * y, arr.get_shape(), 1)
        assert a1 == a2

        self._params['shape'] = shape if isinstance(shape, tuple) else None

    def gradient(self, wrt):
        # TODO or is there a gradient for ReshapeOP? maybe just neutral?
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return ReshapeOP(self.operands[0].substitute(a, b), self._params['shape'], self.parents)

    def get_shape(self):
        return self._params['shape']

    def simplify(self):
        # Ideje:
        # ako je operand skalar ili literal, nema sta da se shapeuje
        # ako je operand jos jedan reshape, njegova primena se moze preskociti
        # ako je operand

        # ff = config.get_formula_factory()
        # return ff.create_reshape(self.operands[0].simplify(), self._shape)
        raise NotImplementedError


class RavelOP(UnaryOP):
    _type_name = 'Ravel'

    def __init__(self, op, parents=None):
        super(RavelOP, self).__init__(1, [op], parents)

    def simplify(self):
        # idea: if op is already a vector just omit this operator.
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return RavelOP(self.operands[0].substitute(a, b), self.parents)

    def get_shape(self):
        return (reduce(lambda x, y: x * y, self.operands[0].get_shape(), 1),)


class TransposeOP(UnaryOP):
    _type_name = 'Transpose'

    def __init__(self, op, axes=None, parents=None):
        super(TransposeOP, self).__init__(1, [op], parents)
        # self._axes = axes
        self._params['axes'] = axes

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return TransposeOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return TransposeOP(self.operands[0].gradient(wrt))

    def simplify(self):
        # some ideas:
        # if op is also a transpose on the same axes, cancel out
        pass


class DimshuffleOP(UnaryOP):
    _type_name = 'Dimshuffle'

    def __init__(self, op, new_dims, parents=None):
        super(DimshuffleOP, self).__init__(1, [op], parents)
        self._params['new_dims'] = new_dims

    def simplify(self):
        pass

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_dimshuffle(self.operands[0].substitute(a, b), self._new_dims)

    def get_shape(self):
        return self._params['new_dims']


class DiagonalOP(UnaryOP):
    _type_name = 'Diagonal'

    def __init__(self, op, parents=None):
        super(DiagonalOP, self).__init__(1, [op], parents)

    def simplify(self):
        pass

    def gradient(self, wrt):
        # TODO ipak jeste
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_diagonal(self.operands[0].substitute(a, b))  # TODO parents?

    def get_shape(self):
        pass


class TraceOP(UnaryOP):
    _type_name = 'Trace'

    def __init__(self, op, parents=None):
        super(TraceOP, self).__init__(1, [op], parents)

    def simplify(self):
        pass

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return TraceOP(self.operands[0].gradient(wrt))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return TraceOP(self.operands[0].substitute(a, b))

    def get_shape(self):
        return (1,)


class ConcatenateOP(exprgraph.Operator):
    _type_name = 'Concatenate'


class StackOP(exprgraph.Operator):
    _type_name = 'Stack'
    # TODO what arity?


class SplitOP(exprgraph.Operator):
    pass


class RepeatOP(exprgraph.Operator):
    pass


# BITWISE OPS (binary operations) #####################################


class BitwiseAndOP(BinaryOP):
    _type_name = 'bAnd'


class BitwiseOrOP(BinaryOP):
    _type_name = 'bOr'


class BitwiseXorOP(exprgraph.Operator):
    _type_name = 'bXor'


class InvertOP(UnaryOP):
    _type_name = 'Invert'


class LeftShiftOP(exprgraph.Operator):
    pass


class RightShiftOP(exprgraph.Operator):
    pass


# INDEX OPS ###########################################################


class IndexOP(exprgraph.Operator):
    _type_name = 'Index'

    # TODO proveri kako se radi ono advanced i basic indeksiranje u Theanou i sta od toga moze u pyopenclu.

    def __init__(self, op, key, parents=None):
        """
        WRITEME

        :param key an integer or slice object wrapped in a exprgraph.Constant
        """
        super(IndexOP, self).__init__(1, [op], parents)
        self._params['key'] = key

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_index()

    def simplify(self):
        pass    # TODO
        # ff = config.get_formula_factory()
        # return ff.create_index(self.operands[0].simplify(), self._key, self.parents)

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def get_shape(self):
        pass  # TODO calculate size and shape of the result if possible?


# LINEAR ALGEBRA ######################################################


class DotOP(exprgraph.Operator):
    _type_name = 'Dot'
    pass


class VdotOP(exprgraph.Operator):
    _type_name = 'Vdot'
    pass


class InnerOP(exprgraph.Operator):
    _type_name = 'Inner'
    pass


class OuterOP(exprgraph.Operator):
    _type_name = 'Outer'
    pass


class MatmulOP(exprgraph.Operator):
    _type_name = 'Matmul'
    pass


class EigOP(exprgraph.Operator):
    _type_name = 'Eig'
    pass


class EigvalsOP(exprgraph.Operator):
    _type_name = 'Eigvals'
    pass


class NormOP(exprgraph.Operator):
    _type_name = 'Norm'

    def __init__(self, op, norm, parents=None):
        super(NormOP, self).__init__(1, [op], parents)
        self._params['norm'] = norm

    def simplify(self):
        pass

    def gradient(self, wrt):
        pass

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_norm(self.operands[0].substitute(a, b), self.operands[1])

    def get_shape(self):
        pass


# LOGIC OPS ###########################################################


class AllOP(UnaryOP):
    _type_name = 'All'

    def __init__(self, op, parents=None):
        super(AllOP, self).__init__(1, [op], parents)

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass

    def substitute(self, a, b):
        pass


class AnyOP(UnaryOP):
    _type_name = 'Any'

    def __init__(self, op, parents=None):
        super(AnyOP, self).__init__(1, [op], parents)

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass

    def substitute(self, a, b):
        pass


class AndOP(BinaryOP):
    pass


class OrOP(BinaryOP):
    pass


class NotOP(UnaryOP):
    pass


class XorOP(BinaryOP):
    pass


class GtOP(BinaryOP):
    pass


class LtOP(BinaryOP):
    pass


class GeOP(BinaryOP):
    pass


class LeOP(BinaryOP):
    pass


class EqOP(BinaryOP):
    _type_name = 'Eq'
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2, parents=None):
        super(EqOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s == %s)' % (str(self.operands[0]), str(self.operands[1]))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass    # TODO
        # ff = config.get_formula_factory()
        # simp_op1, simp_op2 = (op.simplify() for op in self.operands)
        # return ff.create_eq(simp_op1, simp_op2)

    def substitute(self, a, b):
        pass


class NeOP(BinaryOP):
    pass


# MATHEMATICAL OPS ######################################################


class SinOP(UnaryOP):
    _type_name = "Sin"

    def __init__(self, operand, parents=None):
        super(SinOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return 'sin(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return SinOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1)
        return MulOP(CosOP(self.operands[0]), self.operands[0].gradient(wrt))

    def simplify(self):
        # TODO
        raise NotImplementedError


class CosOP(UnaryOP):
    _type_name = "Cos"

    def __init__(self, operand, parents=None):
        super(CosOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return "cos(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return CosOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # -sin(x)*dx
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(NegOP(SinOP(self.operands[0])), self.operands[0].gradient(wrt))

    def simplify(self):
        # TODO
        raise NotImplementedError


class TanOP(UnaryOP):
    _type_name = "Tan"

    def __init__(self, operand, parents=None):
        super(TanOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return "tan(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return TanOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO 1/(cos(x)^2)  or 1+tan(x)^2 or sec(x)^2
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class ArcsinOP(UnaryOP):                                                        # TODO do i need this now?
    _type_name = 'Arcsin'

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        # 1/sqrt(1-x^2)*dx          -1 < x < 1
        return MulOP(ReciprocalOP(SqrtOP(SubOP(exprgraph.Constant(1), SqrOP(self.operands[0])))),
                     self.operands[0].gradient(wrt))


class ArccosOP(UnaryOP):                                                        # TODO do i need this now?
    _type_name = 'ArccosOP'

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        # -1/sqrt(1-x^2)*dx          -1 < x < 1
        return NegOP(MulOP(ReciprocalOP(SqrtOP(SubOP(exprgraph.Constant(1), SqrOP(self.operands[0])))),
                           self.operands[0].gradient(wrt)))


class ArctanOP(UnaryOP):                                                        # TODO do i need this now?
    _type_name = 'Arctan'

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        # 1/(1 + x^2)*dx
        return MulOP(ReciprocalOP(AddOP(exprgraph.Constant(1), SqrOP(self.operands[0]))),
                     self.operands[0].gradient(wrt))


class SinhOP(UnaryOP):
    _type_name = 'Sinh'

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        # https://en.wikipedia.org/wiki/Hyperbolic_function
        return MulOP(CoshOP(self.operands[0]), self.operands[0].gradient(wrt))


class CoshOP(UnaryOP):
    _type_name = 'Cosh'

    def __init__(self, operand, parents=None):
        super(CoshOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return CoshOP(self.operands[0].substitute(a, b))

    def simplify(self):
        pass

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(SinhOP(self.operands[0]), self.operands[0].gradient(wrt))


class TanhOP(UnaryOP):
    _type_name = 'Tanh'

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(SubOP(exprgraph.Constant(1), SqrOP(TanhOP(self.operands[0]))),
                     self.operands[0].gradient(wrt))


class ArcsinhOP(UnaryOP):
    _type_name = 'Arcsinh'

    def __init__(self, operand, parents=None):
        super(ArcsinhOP, self).__init__(1, [operand], parents)

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(ReciprocalOP(SqrtOP(AddOP(exprgraph.Constant(1), SqrOP(self.operands[0])))),
                     self.operands[0].gradient(wrt))


class ArccoshOP(UnaryOP):
    pass


class ArctanhOP(UnaryOP):
    pass


class RoundOP(UnaryOP):
    _type_name = 'Round'

    def __init__(self, operand, parents=None):
        super(RoundOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_round(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        # TODO raise NotDifferentiable????

    def simplify(self):
        pass


class FloorOP(UnaryOP):
    _type_name = 'Floor'

    def __init__(self, operand, parents=None):
        super(FloorOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_floor(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class CeilOP(UnaryOP):
    _type_name = 'Ceil'

    def __init__(self, operand, parents=None):
        super(CeilOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_ceil(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass  # Nondifferentiable?

    def simplify(self):
        pass


class ProdOP(exprgraph.Operator):
    pass


class SumOP(exprgraph.Operator):
    pass


class NansumOP(exprgraph.Operator):
    pass


class CumprodOP(exprgraph.Operator):
    pass


class CumsumOP(exprgraph.Operator):
    pass


class ExpOP(UnaryOP):
    _type_name = "Exp"

    def __init__(self, operand, parents=None):
        super(ExpOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return ExpOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return ExpOP(self.operands[0])

    def simplify(self):
        simp_op = self.operands[0].simplify()
        # TODO wtf?
        if isinstance(simp_op, ExpOP):
            return simp_op.operands[0]

        raise NotImplementedError


class Exp2OP(UnaryOP):
    _type_name = 'Exp2'

    def __init__(self, operand, parents=None):
        super(Exp2OP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return Exp2OP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError

    def simplify(self):
        raise NotImplementedError


class LogOP(UnaryOP):
    _type_name = "Log"

    def __init__(self, operand, parents=None):
        super(LogOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return LogOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(ReciprocalOP(self.operands[0]), self.operands[0].gradient(wrt))

    def simplify(self):
        raise NotImplementedError   # TODO


class Log10OP(UnaryOP):
    _type_name = 'Log10'

    def __init__(self, operand, parents=None):
        super(Log10OP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return Log10OP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError   # TODO

    def simplify(self):
        raise NotImplementedError   # TODO


class Log1pOP(UnaryOP):
    _type_name = 'Log1p'

    def __init__(self, operand, parents=None):
        super(Log1pOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return Log1pOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError   # TODO

    def simplify(self):
        raise NotImplementedError   # TODO


class AddOP(BinaryOP):
    _type_name = 'Add'
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2, parents=None):
        super(AddOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s + %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return AddOP(self.operands[0].substitute(a, b),
                         self.operands[1].substitute(a, b),
                         self.parents)

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return AddOP(self.operands[0].gradient(wrt),
                     self.operands[1].gradient(wrt),
                     self.parents)

    def simplify(self):
        raise NotImplementedError   # TODO


class ReciprocalOP(UnaryOP):
    _type_name = 'Reciprocal'

    def __init__(self, op, parents=None):
        super(ReciprocalOP, self).__init__(1, [op], parents)

    def __str__(self):
        return '1 / (%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return ReciprocalOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(DivOP(exprgraph.Constant(-1), SqrOP(self.operands[0])), self.operands[0].gradient(wrt))

    def simplify(self):
        raise NotImplementedError    # TODO


class NegOP(UnaryOP):
    _type_name = "Neg"

    def __init__(self, operand, parents=None):
        super(NegOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return '-(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return NegOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return NegOP(self.operands[0].gradient(wrt))

    def simplify(self):
        simp_op = self.operands[0].simplify()
        if type(simp_op) == type(self):
            return simp_op.operands[0]
        return NegOP(simp_op)


class MulOP(BinaryOP):
    _type_name = 'Mul'
    isCommutative = True
    isAssociative = False

    def __init__(self, op1, op2, parents=None):
        super(MulOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s * %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return MulOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return AddOP(MulOP(self.operands[0].gradient(wrt), self.operands[1]),
                     MulOP(self.operands[0], self.operands[1].gradient(wrt)))

    def simplify(self):
        raise NotImplementedError


class DivOP(BinaryOP):
    _type_name = 'Div'
    isCommutative = False
    isAssociative = True

    def __init__(self, op1, op2, parents=None):
        super(DivOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s / %s)' % map(str, self.operands)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return DivOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return DivOP(SubOP(MulOP(self.operands[0].gradient(wrt), self.operands[1]),
                           MulOP(self.operands[0], self.operands[1].gradient(wrt))),
                     SqrOP(self.operands[1]))

    def simplify(self):
        raise NotImplementedError


class PowOP(BinaryOP):
    _type_name = 'Pow'
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2, parents=None):
        super(PowOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s ^ %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return PowOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError   # TODO finish this

    def simplify(self):
        raise NotImplementedError   # TODO


class SubOP(BinaryOP):
    _type_name = 'Sub'
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2, parents=None):
        super(SubOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s - %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return SubOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return SubOP(self.operands[0].gradient(wrt), self.operands[1].gradient(wrt))

    def simplify(self):
        raise NotImplementedError   # TODO


class ModOP(BinaryOP):
    _type_name = 'Mod'

    def __init__(self, op1, op2, parents=None):
        super(ModOP, self).__init__(2, [op1, op2], parents)

    def __str__(self):
        return '(%s %% %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return ModOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError   # TODO

    def simplify(self):
        raise NotImplementedError   # TODO


class AbsOP(UnaryOP):
    _type_name = 'Abs'

    def __init__(self, op, parents=None):
        super(AbsOP, self).__init__(1, [op], parents)

    def simplify(self):
        # if operand is also abs, colapse it
        # if operand is a constant, colapse this into a constant
        pass

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        pass  # TODO <0 : -1, 0: 0, >0: 1

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return AbsOP(self.operands[0].substitute(a, b))


class SignOP(UnaryOP):
    _type_name = 'Sign'

    def __init__(self, operand, parents=None):
        super(SignOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return "sign(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return SignOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NotImplementedError   # TODO

    def simplify(self):
        pass


class SqrOP(UnaryOP):
    _type_name = 'Sqr'

    def __init__(self, op, parents=None):
        super(SqrOP, self).__init__(1, [op], parents)

    def __str__(self):
        return '(%s ^ 2)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return SqrOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        return MulOP(MulOP(exprgraph.Constant(2.0), self.operands[0]), self.operands[0].gradient(wrt))

    def simplify(self):
        raise NotImplementedError


class SqrtOP(UnaryOP):
    _type_name = 'Sqrt'

    def __init__(self, operand, parents=None):
        super(SqrtOP, self).__init__(1, [operand], parents)

    def __str__(self):
        return 'sqrt(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return SqrtOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        pass

    def simplify(self):
        pass


class MaximumOP(BinaryOP):
    _type_name = 'Maximum'

    def __init__(self, op1, op2, parents=None):
        super(MaximumOP, self).__init__(2, [op1, op2], parents)

    def substitute(self, a, b):
        pass

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NondifferentiableOpError

    def simplify(self):
        pass


class MinimumOP(BinaryOP):
    _type_name = 'Minimum'

    def __init__(self, op1, op2, parents=None):
        super(MinimumOP, self).__init__(2, [op1, op2], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return MinimumOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NondifferentiableOpError

    def simplify(self):
        pass


# STATISTICS OPS ######################################################

class MedianOP(UnaryOP):
    _type_name = 'Median'

    def __init__(self, operand, parents=None):
        super(MedianOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return MedianOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def simplify(self):
        pass


class AverageOP(UnaryOP):
    _type_name = 'Avg'

    def __init__(self, operand, parents=None):
        super(AverageOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return AverageOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def simplify(self):
        pass


class MeanOP(UnaryOP):
    _type_name = 'Mean'

    def __init__(self, operand, parents=None):
        super(MeanOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return MeanOP(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NondifferentiableOpError

    def simplify(self):
        pass


class StdOP(UnaryOP):
    _type_name = 'STD'

    def __init__(self, operand, parents=None):
        super(StdOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return StdOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def simplify(self):
        pass


class VarOP(UnaryOP):
    _type_name = 'Var'

    def __init__(self, operand, parents=None):
        super(VarOP, self).__init__(1, [operand], parents)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            return VarOP(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        if self == wrt:
            return exprgraph.Constant(1.0)
        raise NondifferentiableOpError

    def simplify(self):
        pass


class CorrelateOP(exprgraph.Operator):
    _type_name = 'Corr'
    pass    # TODO this shoud probably be a binary op


class CovOP(exprgraph.Operator):
    _type_name = 'Cov'
    pass    # TODO binary op probably


# ELEMENTWISE OP ######################################################

class ElementwiseOP(exprgraph.Operator):
    _type_name = 'Elementwise'

    def __init__(self, expr, ops, parents=None):
        super(ElementwiseOP, self).__init__(len(ops), ops, parents=parents)
        self._expr = expr

    def substitute(self, a, b):
        # TODO should this be supported?
        pass

    def get_shape(self):
        return self.operands[0].get_shape()  # does output have to have this dimension?

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass  # isn't it an error to call this for elementwise?


# SCAN OPS ############################################################

# sum, max, min, mean, std

# mogao bih da imam map expression?
# mogao bih da imam i expression koji se koristi za redukovanje?
# ali mogao bih i samo da blize pratim pyopencl pristup i da prosledim konkretne parametre
# koji bi direktno konstruisali operator?
# Da li se moze naci gradijent toga?
class ReduceOP(exprgraph.Operator):
    _type_name = 'Reduce'

    def __init__(self, operands, neutral, reduce_expr, map_expr=None, parents=None):
        super(ReduceOP, self).__init__(len(operands), operands, parents)
        self._neutral = neutral
        self._reduce_expr = reduce_expr
        self._map_expr = map_expr

    def get_shape(self):
        pass

    def substitute(self, a, b):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class ScanOP(exprgraph.Operator):
    _type_name = 'Scan'

    def __init__(self, operands, input_expr, scan_expr, output_statement=None, parents=None):
        super(ScanOP, self).__init__(len(operands), operands, parents)
        self._input_expr = input_expr
        self._scan_expr = scan_expr
        self._output_statement = output_statement

    def get_shape(self):
        pass

    def substitute(self, a, b):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


# BLAS OPS ############################################################

class GemmOP(exprgraph.Operator):
    _type_name = "Gemm"

    def __init__(self, A, B, C, alpha=exprgraph.Constant(1.0), beta=exprgraph.Constant(0.0),
                 transA=exprgraph.Constant(False), transB=exprgraph.Constant(False), parents=None):
        super(GemmOP, self).__init__(7, [A, B, C, alpha, beta, transA, transB], parents)

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
                 transA=exprgraph.Constant(False), parents=None):
        super(GemvOP, self).__init__(6, [A, X, Y, alpha, beta, transA], parents)
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

    def __init__(self, alpha, X, Y, A, parents=None):
        super(GerOP, self).__init__(4, [alpha, X, Y, A], parents)
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

class ConvOP(exprgraph.Operator):
    _type_name = 'Conv'

    def __init__(self, input, filters, image_shape=None, filter_shape=None,
                 border_mode='valid', parents=None):
        super(ConvOP, self).__init__(1, [input], parents)
        self._filters = filters
        self._image_shape = image_shape
        self._filter_shape = filter_shape
        self._border_mode = border_mode

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            pass    # TODO
            # ff = config.get_formula_factory()
            # return ff.create_conv(self.operands[0].substitute(a, b), self._filters,
            #                       self._image_shape, self._filter_shape, self._border_mode,
            #                       self.parents)

    def get_shape(self):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class PoolOP(exprgraph.Operator):
    _type_name = 'Pool'

    def __init__(self, input, ds, mode, parents=None):
        super(PoolOP, self).__init__()


class DownsampleOP(exprgraph.Operator):
    pass
