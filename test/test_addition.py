import unittest
import numpy as np
from shaderrider.symbolic import exprgraph
from shaderrider.symbolic.operators import AddOP
from shaderrider.generator import function


class FirstStep(unittest.TestCase):
    def setUp(self):
        pass

    def test_np_add(self):
        a = exprgraph.Variable(name='a', array=np.eye(3,3))
        b = exprgraph.Variable(name='b', array=np.ones((3,3), np.float32))
        expr = AddOP(a, b)
        f1 = function.function([expr], name='allonsy', platform='numpy')
        self.assertEqual(f1._name, 'allonsy')
        v = function.valuation(platform='numpy')
        v.add('a', a)
        v.add('b', b)
        f1(valuation=v)
        ret = v.get('Add2')
        self.assertEqual(ret[0,0], 2)
        self.assertEqual(ret[0,1], 1)
        self.assertEqual(ret[1,1], 2)

    def test_pyocl_add(self):
        a = exprgraph.Variable(name='a', array=np.eye(3, 3))
        b = exprgraph.Variable(name='b', array=np.ones((3, 3), np.float32))
        expr = AddOP(a, b)
        f1 = function.function([expr], name='allonsy', platform='pyopencl')
        self.assertEqual(f1._name, 'allonsy')
        v = function.valuation(platform='pyopencl')
        v.add('a', a)
        v.add('b', b)
        f1(valuation=v)
        ret = v.get('Add2')
        self.assertEqual(ret[0, 0], 2)
        self.assertEqual(ret[0, 1], 1)
        self.assertEqual(ret[1, 1], 2)
