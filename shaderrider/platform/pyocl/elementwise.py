"""
WRITEME
"""

from mako.template import  Template

from pyopencl.elementwise import ElementwiseKernel

from shaderrider.symbolic import vast as ast
from shaderrider.symbolic import basic
from shaderrider.generator import codegen


class ElementwiseEval(codegen.OpEvaluator):
    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation=None):
        pass

    def evaluate(self, op, valuation=None):
        pass

    def after(self, op, valuation=None):
        pass


class ElementwiseGenerator(codegen.OpEvalGenerator):
    def generate(self, op, ctx):
        # TODO get atomics (inputs), and figure out the output type and dimension
        atoms = op.getAtoms()   # TODO what about constants?
        args = []
        for a in atoms:
            if a.isArray():
                args.append("%s *%s" % (a.dtype, a.name))
            else:   # scalar
                args.append("%s %s" %(a.dtype, a.name))
        argstr = ', '.join(args)
        # TODO generate C code for the operation (what is the PYOPENCL_ELWISE_CONTINUE exactly?)
        cexpr = _c_expr(op)
        ewk = ElementwiseKernel(ctx, argstr, cexpr)
        # TODO create the evaluator class (how?)

        klasstemp = '''
class ${eval_class_name}(codegen.OpEvaluator):
    def __init__(self, ewk, atoms):
        self._ewk = ewk
        self._atoms = atoms

    def init(self):
        pass

    def finalize(self):
        pass

    def before(self, op, valuation):
        pass

    def after(self, op, valuation):
        pass

    def evaluate(self, op, valuation=None):
        % for a in ${atoms}:
        arg_${a.fid} = valuation['${a.fid}']
        % endfor
'''

        def eval(self, myop, valuation):
            for a in self._atoms:
                pass
        return type(op.fid+'_elementwise_evaluator', (codegen.OpEvaluator,), {
            '_ewk': ewk,
            '_atoms': atoms,
            'init': lambda self: 1,
            'finalize': lambda self: 1,
            'before': lambda self, op, valuation: 1,
            'after': lambda self, op, valuation: 1,
            'eval': lambda self, op, valuation: 1
        })


def _c_expr(formula):
    if isinstance(formula, ast.Atom):
        if formula.isArray():
            return formula.name + '[i]'
        return formula.name
    if isinstance(formula, ast.Constant):
        if formula.isArray():   # TODO check this
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
