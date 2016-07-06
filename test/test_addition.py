import unittest
import numpy as np
from shaderrider.symbolic import exprgraph
from shaderrider.symbolic.operators import AddOP
from shaderrider.generator import function


class FirstStep(unittest.TestCase):
    def setUp(self):
        pass

    def testAdd(self):
        a = exprgraph.Variable(name='a', array=np.eye(3,3))
        b = exprgraph.Variable(name='b', array=np.ones((3,3), np.float32))
        expr = AddOP(a, b)
        self.assertEqual(1, 1)
