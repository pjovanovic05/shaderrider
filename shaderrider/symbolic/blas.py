"""
WRITEME
"""

import shaderrider.symbolic.vast as ast
from shaderrider.core import IncompatibleDimensionsError


class GemmOP(ast.Operator):
    _type_name = "Gemm"

    def __init__(self, A, B, C, alpha=ast.Constant(1.0), beta=ast.Constant(0.0),
                 transA=ast.Constant(False), transB=ast.Constant(False)):
        super(GemmOP, self).__init__(7, [A, B, C, alpha, beta, transA, transB])

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


class GemvOP(ast.Operator):
    _type_name = "Gemv"

    def __init__(self, A, X, Y, alpha=ast.Constant(1.0), beta=ast.Constant(0.0), transA=ast.Constant(False)):
        super(GemvOP, self).__init__(6, [A, X, Y, alpha, beta, transA])
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


class GerOP(ast.Operator):
    _type_name = "Ger"

    def __init__(self, alpha, X, Y, A):
        super(GerOP, self).__init__(4, [alpha, X, Y, A])
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
