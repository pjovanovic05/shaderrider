import unittest
import numpy as np
from shaderrider import configuration
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
        ret = v.get(expr.fid)
        self.assertEqual(ret[0,0], 2)
        self.assertEqual(ret[0,1], 1)
        self.assertEqual(ret[1,1], 2)

    def test_pyocl_add(self):
        configuration.set_platform('pyopencl', ngpus=1)
        a = exprgraph.Variable(name='a', array=np.eye(3, 3, dtype=np.float32))
        b = exprgraph.Variable(name='b', array=np.ones((3, 3), np.float32))
        expr = AddOP(a, b)
        f1 = function.function([expr], name='allonsy', platform='pyopencl')
        self.assertEqual(f1._name, 'allonsy')
        v = function.valuation(platform='pyopencl')
        v.add('a', a)
        v.add('b', b)
        f1(valuation=v)
        ret = v.get(expr.fid)
        self.assertEqual(ret.value[0, 0], 2)
        self.assertEqual(ret.value[0, 1], 1)
        self.assertEqual(ret.value[1, 1], 2)

    def test_complexity(self):
        a = exprgraph.Variable(name='a', array=np.eye(3, 3, dtype=np.float32))
        b = exprgraph.Variable(name='b', array=np.ones((3, 3), np.float32))
        expr = AddOP(a, b)
        self.assertEqual(expr.complexity(), 1)
