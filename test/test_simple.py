import unittest
import sys
import numpy as np

from shaderrider import expr
from shaderrider import operators as op
from shaderrider import clplatf as plat


class SimpleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plat.init_cl(1)

    def test_addition(self):
        x = expr.Variable("x")
        y = expr.Constant(2)

        valuation = plat.valuation()

        valuation['x'] = np.eye(2, dtype=np.float32)

        z = op.Add(x,y).evaluate(valuation)
        zz = z.get()
        self.assertEqual(zz[0,0], 3.0)
        self.assertEqual(zz[0,1], 2.0)
        self.assertEqual(zz[1,0], 2.0)
        self.assertEqual(zz[1,1], 3.0)

    def test_fwd_grad(self):
        x = expr.Variable('x')
        wrt = {'x': 1}

        valuation = plat.valuation()

        e1 = op.Add(x, x)
        valuation['x'] = np.eye(2, dtype=np.float32)
        d1 = e1.fwd_grad(wrt, valuation)
        self.assertEqual(d1, 2)

    def test_rev_grad(self):
        x = expr.Variable('x')
        y = expr.Variable('y')
        z = expr.Variable('z')

        valuation = plat.valuation()
        valuation['x'] = np.ones((2,2), dtype=np.float32)*5
        valuation['y'] = np.eye(2, dtype=np.float32)*2
        valuation['z'] = np.ones((2,2), dtype=np.float32)*3

        ex = op.Add(op.Mul(x, y), z)
        # rg(X.*Y+Z) = [
        rg = ex.rev_grad(valuation)

        # dex/dx = y
        dx = rg['x'].get()
        yy = valuation['y'].get()
        self.assertTrue(np.all(dx == yy))

    def test_sigmoid_grads(self):
        # TODO Testiraj da li AD gradijent 1/(1+exp(-x)) daje priblizno iste rezultate
        # kao i sigmoid(x)*(1-sigmoid(x)), tj. grad sigmoida.
        x = expr.Variable('x')
        valuation = plat.valuation()
        valuation['x'] = np.ones((3,3), dtype=np.float32)*2

        sigm1 = op.Div(expr.Constant(1.0), op.Add(expr.Constant(1.0), op.Exp(op.Neg(x))))
        sigm2 = op.Sigmoid(x)

        e1 = sigm1.evaluate(valuation)
        e2 = sigm2.evaluate(valuation)

        rg1 = sigm1.rev_grad(valuation)
        rg2 = sigm2.rev_grad(valuation)

        xg1 = rg1['x'].get()
        xg2 = rg2['x'].get()
        self.assertTrue(np.all((xg1 - xg2) < 0.001))
