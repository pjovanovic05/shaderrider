import unittest
import numpy as np
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider import linalg

import sys


class LinalgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clplatf.init_cl(1)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dot(self):
        q = clplatf.qs[0]
        X = np.random.uniform(0, 1, (50000,)).astype(np.float32)
        Y = np.random.uniform(0, 1, (50000,)).astype(np.float32)
        expected = np.dot(X, Y)
        gX = clarray.to_device(q, X)
        gY = clarray.to_device(q, Y)

        gR = linalg.dot(q, gX, gY)
        R = gR.get()
        self.assertTrue(np.allclose(R, expected))

        A = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
        B = np.random.uniform(0, 1, (512, 512)).astype(np.float32)

        expected = np.dot(A, B)
        gA = clarray.to_device(q, A)
        gB = clarray.to_device(q, B)
        gC = linalg.dot(q, gA, gB)
        C = gC.get()
        self.assertTrue(np.allclose(C, expected))

    def test_dot_again(self):
        q = clplatf.qs[0]
        X = np.random.uniform(0, 1, (128, 64, 1024)).astype(np.float32)
        Y = np.random.uniform(0, 1, (128, 27, 1024)).astype(np.float32)

        gX = clarray.to_device(q, X)
        gY = clarray.to_device(q, Y)

        for i in range(128):
            expected = X[i].dot(Y[i].T)
            gR = linalg.dot(q, gX[i], gY[i], transB=True)
            R = gR.get()
            if not np.allclose(R, expected):
                print >>sys.stderr, '\nReal:\n', R
                print >>sys.stderr, 'expected:\n', expected
                print >>sys.stderr, 'shapes: r:', R.shape, 'e:', expected.shape
                print >>sys.stderr, 'mean diff:', np.mean(R-expected)
                break
            self.assertTrue(np.allclose(R, expected))

    def test_dot_offentdingvectors(self):
        q = clplatf.qs[0]
        X = np.loadtxt(open('test/gymat.txt', 'r'), delimiter=',').astype(np.float32)
        Y = np.loadtxt(open('test/colmat.txt', 'r'), delimiter=',').astype(np.float32)
        gX = clarray.to_device(q, X)
        gY = clarray.to_device(q, Y)

        expected = X.dot(Y.T)
        gR = linalg.dot(q, gX, gY, transB=True)
        R = gR.get()
        print >>sys.stderr, '\nReal:\n', R
        print >>sys.stderr, 'expected:\n', expected
        print >>sys.stderr, 'shapes: r:', R.shape, 'e:', expected.shape
        print >>sys.stderr, 'mean diff:', np.mean(R-expected)
        self.assertTrue(np.allclose(R, expected))

    def test_batch_dot(self):
        pass
