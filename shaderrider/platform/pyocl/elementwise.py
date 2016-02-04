"""
WRITEME
"""

from pyopencl.elementwise import ElementwiseKernel
from pyopencl import array
import pyopencl as cl

from shaderrider.symbolic import exprgraph as ast
from shaderrider.symbolic import elementwise
from shaderrider.symbolic import basic


class ElementwiseOP(elementwise.ElementwiseOP):
    def __init__(self, expr, ctx=None, device=0):
        """
        :type expr: Formula
        """
        super(ElementwiseOP, self).__init__(expr)
        self._ctx = ctx
        self._expr = expr
        self._fn = self.generate_eval()

    def evaluate(self, valuation=None):
        return self._fn(valuation)

    def generate_eval(self):
        atoms = self._expr.get_atoms()
        args = []
        for a in atoms:
            if a.is_array():
                args.append('%s *%s' % (a.dtype, a.name))
            else:
                args.append('%s %s' % (a.dtype, a.name))
        argstr = ', '.join(args)
        argstr += ', %s *%s' % (self.dtype, self.fid)
        cexpr = _c_expr(self._expr)
        ewk = ElementwiseKernel(self._ctx, argstr, cexpr)

        def evaluator(valuation, events=None, device=0):
            params = []
            waits = []
            for a in atoms:
                if a.fid in valuation:
                    params.append(valuation[a.fid].data if a.is_array() else valuation[a.fid])
                else:
                    raise ValueError('Valuation missing parameter ' + a.fid)
                if a.fid in events:
                    waits.append(events[a.fid])
            out = valuation[self.fid] if self.fid in valuation else array.zeros(self._ctx, self.get_shape(), self.dtype)
            params.append(out)
            return ewk(wait_for=waits, *params)

        return evaluator


def _c_expr(formula):
    if isinstance(formula, ast.Atom):
        if formula.is_array():
            return formula.name + '[i]'
        return formula.name
    if isinstance(formula, ast.Constant):
        if formula.is_array():   # TODO check this
            return formula.fid + '[i]'
        return formula.fid
    if isinstance(formula, basic.NegOP):
        return '-(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.ExpOP):
        return 'exp(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.LogOP):
        return 'log(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.SinOP):
        return 'sin(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.CosOP):
        return 'cos(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.TanOP):
        return 'tan(%s)' % _c_expr(formula.operands[0])
    if isinstance(formula, basic.AddOP):
        return '(%s + %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.SubOP):
        return '(%s - %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.MulOP):
        return '(%s * %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.DivOP):
        return '(%s / %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.PowOP):
        return 'pow(%s, %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.EqOP):
        return '(%s == %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.GtOP):
        return '(%s > %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.LtOP):
        return '(%s < %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.GeOP):
        return '(%s >= %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.LeOP):
        return '(%s <= %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))
    if isinstance(formula, basic.NeOP):
        return '(%s != %s)' % (_c_expr(formula.operands[0]), _c_expr(formula.operands[1]))

    # TODO handle blas and other more complex functions which behave as atoms in this context

    raise ValueError('Unable to convert formula to c expression: %s' % formula)
