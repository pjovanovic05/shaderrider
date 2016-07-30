import unittest
import numpy as np

from shaderrider import expr
from shaderrider import operators as op
from shaderrider import clplatf as plat


class SimpleTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_addition(self):
        x = expr.Variable("x")
        y = expr.Constant(2)

        plat.init_cl(1)
        valuation = plat.valuation()

        valuation['x'] = np.zeros((2,2), dtype=np.float32)

        z = op.Add(x,y).evaluate(valuation)
        zz = z.get()
        self.assertEqual(zz[0,0], 2.0)
        self.assertEqual(zz[0,1], 2.0)

    def test_fwd_grad(self):
        pass

    def test_rev_grad(self):
        pass
