"""
All the operator types.

WRITEME
"""
import abc

import shaderrider.configuration as config
from shaderrider.core import IncompatibleDimensionsError, NondifferentiableOpError
from shaderrider.generator.codegen import FormulaFactory  # TODO gradient and substitutions need to use this
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

        self._shape = shape if isinstance(shape, tuple) else None  # TODO extract shape
        # TODO check transformation compatibility
        # multiply shape components and see if the lengths match current length

    def gradient(self, wrt):
        raise NondifferentiableOpError

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_reshape(self.operands[0].substitute(a, b), self._shape)

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
        pass  # TODO calculate size and shape of the result if possible?


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
            return ff.create_diagonal(self.operands[0].substitute(a, b))  # TODO parent?

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

    def __init__(self, op, parent=None):
        super(AbsOP, self).__init__(1, [op], parent)

    def simplify(self):
        # if operand is also abs, colapse it
        # if operand is a constant, colapse this into a constant
        pass

    def gradient(self, wrt):
        pass  # <0 : -1, 0: 0, >0: 1

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_abs(self.operands[0].substitute(a, b))  # parent?


class NegOP(UnaryOP):
    _type_name = "Neg"

    def __init__(self, operand, parent=None):
        super(NegOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return '-(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_neg(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        return NegOP(self.operands[0].gradient(wrt))  # FIXME use factory!!

    def simplify(self):
        simp_op = self.operands[0].simplify()
        if type(simp_op) == type(self):
            return simp_op.operands[0]
        return NegOP(simp_op)  # FIXME use factory!!


class ExpOP(UnaryOP):
    _type_name = "Exp"

    def __init__(self, operand, parent=None):
        super(ExpOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'exp(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_exp(self.operands[0].substitute(a, b))

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

    def __init__(self, operand, parent=None):
        super(LogOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'log(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_log(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class SinOP(UnaryOP):
    _type_name = "Sin"

    def __init__(self, operand, parent=None):
        super(SinOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'sin(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_sin(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class CosOP(UnaryOP):
    _type_name = "Cos"

    def __init__(self, operand, parent=None):
        super(CosOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return "cos(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_cos(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class CoshOP(UnaryOP):
    _type_name = 'Cosh'

    def __init__(self, operand, parent=None):
        super(CoshOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return "cos(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_cosh(self.operands[0].substitute(a, b))

    def simplify(self):
        pass

    def gradient(self, wrt):
        pass


class TanOP(UnaryOP):
    _type_name = "Tan"

    def __init__(self, operand, parent=None):
        super(TanOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return "tan(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_tan(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        # TODO
        raise NotImplementedError

    def simplify(self):
        # TODO
        raise NotImplementedError


class SignOP(UnaryOP):
    _type_name = 'Sign'

    def __init__(self, operand, parent=None):
        super(SignOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return "sign(%s)" % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_sign(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class CeilOP(UnaryOP):
    _type_name = 'Ceil'

    def __init__(self, operand, parent=None):
        super(CeilOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'ceil(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_ceil(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass  # Nondifferentiable?

    def simplify(self):
        pass


class FloorOP(UnaryOP):
    _type_name = 'Floor'

    def __init__(self, operand, parent=None):
        super(FloorOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'floor(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_floor(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class RoundOP(UnaryOP):
    _type_name = 'Round'

    def __init__(self, operand, parent=None):
        super(RoundOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'round(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_round(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class SqrOP(UnaryOP):
    _type_name = 'Sqr'

    def __init__(self, op, parent=None):
        super(SqrOP, self).__init__(1, [op], parent)

    def __str__(self):
        return '(%s ^ 2)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_sqr(self.operands[0].substitute(a, b), self.parent)

    def gradient(self, wrt):
        pass

    def simplify(self):
        raise NotImplementedError


class SqrtOP(UnaryOP):
    _type_name = 'Sqrt'

    def __init__(self, operand, parent=None):
        super(SqrtOP, self).__init__(1, [operand], parent)

    def __str__(self):
        return 'sqrt(%s)' % str(self.operands[0])

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_sqrt(self.operands[0].substitute(a, b))

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class MaximumOP(BinaryOP):
    _type_name = 'Maximum'

    def __init__(self, op1, op2, parent=None):
        super(MaximumOP, self).__init__(2, [op1, op2], parent)

    def substitute(self, a, b):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class MinimumOP(BinaryOP):
    _type_name = 'Minimum'

    def __init__(self, op1, op2, parent=None):
        super(MinimumOP, self).__init__(2, [op1, op2], parent)

    def substitute(self, a, b):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass


class AddOP(BinaryOP):
    _type_name = 'Add'
    isCommutative = True
    isAssociative = True

    def __init__(self, op1, op2, parent=None):
        super(AddOP, self).__init__(2, [op1, op2], parent)

    def __str__(self):
        return '(%s + %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_add(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b), self.parent)

    def gradient(self, wrt):
        ff = config.get_formula_factory()
        return ff.create_add(self.operands[0].gradient(wrt), self.operands[1].gradient(wrt))

    def simplify(self):
        raise NotImplementedError


class SubOP(BinaryOP):
    _type_name = 'Sub'
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2, parent=None):
        super(SubOP, self).__init__(2, [op1, op2], parent)

    def __str__(self):
        return '(%s - %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_sub(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b), self.parent)

    def gradient(self, wrt):
        ff = config.get_formula_factory()
        return ff.create_sub(self.operands[0].gradient(wrt), self.operands[1].gradient(wrt))

    def simplify(self):
        raise NotImplementedError


class MulOP(BinaryOP):
    _type_name = 'Mul'
    isCommutative = True
    isAssociative = False

    def __init__(self, op1, op2, parent=None):
        super(MulOP, self).__init__(2, [op1, op2], parent)

    def __str__(self):
        return '(%s * %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_mul(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b), self.parent)

    def gradient(self, wrt):
        ff = config.get_formula_factory()
        return ff.create_add(ff.create_mul(self.operands[0].gradient(wrt), self.operands[1]),
                             ff.create_mul(self.operands[0], self.operands[1].gradient(wrt)))

    def simplify(self):
        raise NotImplementedError


class DivOP(BinaryOP):
    _type_name = 'Div'
    isCommutative = False
    isAssociative = True

    def __init__(self, op1, op2, parent=None):
        super(DivOP, self).__init__(2, [op1, op2], parent)

    def __str__(self):
        return '(%s / %s)' % map(str, self.operands)

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_div(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b), self.parent)

    def gradient(self, wrt):
        ff = config.get_formula_factory()
        return ff.create_div(ff.create_sub(ff.create_mul(self.operands[0].gradient(wrt), self.operands[1]),
                                           ff.create_mul(self.operands[0], self.operands[1].gradient(wrt))),
                             ff.create_pow(self.operands[1], exprgraph.Constant(2)))

    def simplify(self):
        raise NotImplementedError


class PowOP(BinaryOP):
    _type_name = 'Pow'
    isCommutative = False
    isAssociative = False

    def __init__(self, op1, op2, parent=None):
        super(PowOP, self).__init__(2, [op1, op2], parent)

    def __str__(self):
        return '(%s ^ %s)' % (str(self.operands[0]), str(self.operands[1]))

    def substitute(self, a, b):
        if self == a:
            return b
        else:
            ff = config.get_formula_factory()
            return ff.create_pow(self.operands[0].substitute(a, b), self.operands[1].substitute(a, b), self.parent)

    def gradient(self, wrt):
        pass

    def simplify(self):
        raise NotImplementedError


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

    def __init__(self, operands, neutral, reduce_expr, map_expr=None, parent=None):
        super(ReduceOP, self).__init__(len(operands), operands, parent)
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

    def __init__(self, operands, input_expr, scan_expr, neutral, parent=None):
        super(ScanOP, self).__init__(len(operands), operands, parent)
        self._input_expr = input_expr
        self._scan_expr = scan_expr
        self._neutral = neutral

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

class Conv(exprgraph.Operator):
    _type_name = 'Conv'
