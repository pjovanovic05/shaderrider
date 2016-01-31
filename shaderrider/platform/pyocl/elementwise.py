"""
WRITEME
"""

from operator import itemgetter as _itemgetter
from collections import OrderedDict

import sys
from mako.template import Template

from pyopencl.elementwise import ElementwiseKernel

from shaderrider.symbolic import vast as ast
from shaderrider.symbolic import basic
from shaderrider.generator import codegen


class ElementwiseEval(codegen.OpEvaluator):
    def setup(self):
        pass

    def teardown(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


# TODO posto imam problem sa nalazenjem parametara za elementwise (koji nisu samo atomi):
# 1) out je resen dole
# 2) parametri koji nisu atomi ce biti pronadjeni u nekom skenu drveta pri proveri da li
#    drvo moze da se elementwiseuje

class ElementwiseGenerator(codegen.OpEvalGenerator):
    def generate(self, op, ctx):
        # TODO get atomics (inputs), and figure out the output type and dimension
        atoms = op.get_atoms()   # TODO what about constants?
        args = []
        for a in atoms:
            if a.is_array():
                args.append("%s *%s" % (a.dtype, a.name))
            else:   # scalar
                args.append("%s %s" % (a.dtype, a.name))
        argstr = ', '.join(args)

        cexpr = _c_expr(op)
        ewk = ElementwiseKernel(ctx, argstr, cexpr)

        template = Template('''
class ${class_name}(codegen.OpEvaluator):
    def __init__(self, ewk, atoms):
        self._ewk = ewk
        self._atoms = atoms

    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation):
        pass    #TODO alloc output buffer if needed!

    def after(self, op, valuation):
        pass

    def evaluate(self, op, valuation=None, events=None):
        % for a in atoms:
        % if a.isArray():
        arg_${a.fid} = valuation['${a.fid}'].data
        % else:
        arg_${a.fid} = valuaion['${a.fid}']
        % endif
        % endfor
        out = valuation[op.fid]

        if events is not None:
            % for a in atoms:
            % if a.isArray():
            # TODO ovo mozda nisu samo atomi - sigmoid je elementwise ali je njegov jedan "atom" u stvari rezultat gemm-a
            # TODO i ne bi trebalo samo o "atomima" pricati, treba i output array nekako uglaviti u parametre.
            % endif
            % endfor

        myev = self._ewk(
                        % for a in atoms:
                        arg_${a.fid},
                        % endfor
                        out.data, wait_for=wait_for
                        )
        if events is not None:
            events[op.fid] = myev
        return myev
        ''')
        # TODO collect atoms taking care of those subexpressions which should behave atomically
        klasstemp = template.render(class_name='elementwise_'+op.fid, atoms=atoms)

        # taken from namedtuple : ~334  (https://hg.python.org/cpython/file/8527427914a2/Lib/collections.py)
        namespace = dict(_itemgetter=_itemgetter, __name__='elementwise_%s' % op.fid,
                         OrderedDict=OrderedDict, _property=property, _tuple=tuple)
        try:
            exec klasstemp in namespace
        except SyntaxError, e:
            raise SyntaxError(e.message + ':\n' + klasstemp)

        ElemwiseKlass = namespace['elementwise_%s' % op.fid]
        try:
            ElemwiseKlass.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass

        ewOpEval = ElemwiseKlass(ewk, atoms)
        return ewOpEval


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

    raise ValueError('Unable to convert formula to c expression: %s' % formula)
