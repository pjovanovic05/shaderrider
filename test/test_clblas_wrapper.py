import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
import pyximport
pyximport.install()
from shaderrider.aux import clblaswrap


class ClBlasWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = None
        cls.q = None
        ps = cl.get_platforms()
        for p in ps:
            ds = list(p.get_devices(device_type=cl.device_type.GPU))
            if len(ds) < 1:
                continue
            cls.ctx = cl.Context(ds)
            cls.q = cl.CommandQueue(cls.ctx, device=ds[0])
            break
        clblaswrap.setup()

    @classmethod
    def tearDownClass(cls):
        clblaswrap.teardown()

    def test_dot(self):
        X = np.random.uniform(0, 1, (10,)).astype(np.float32)
        Y = np.random.uniform(0, 1, (10,)).astype(np.float32)
        expected = np.dot(X, Y)
        gX = clarray.to_device(self.q, X)
        gY = clarray.to_device(self.q, Y)
        gD = clarray.to_device(self.q, np.empty((1,1), dtype=np.float32))
        scratch = clarray.empty(self.q, (10,), dtype=np.float32)
        ev = clblaswrap.dot(self.q, gX, gY, gD, scratch)
        ev.wait()
        aD = gD.get()
        self.assertTrue(np.allclose(aD, expected))

    def test_ger(self):
        X = np.random.uniform(0, 1, (10,)).astype(np.float32)
        Y = np.random.uniform(0, 1, (20,)).astype(np.float32)
        eA = np.outer(X, Y)
        gX = clarray.to_device(self.q, X)
        gY = clarray.to_device(self.q, Y)
        gA = clarray.zeros(self.q, (10, 20), np.float32)
        ev = clblaswrap.ger(self.q, gA, gX, gY)
        ev.wait()
        A = gA.get()

        self.assertEqual(A.shape, eA.shape)
        self.assertTrue(np.allclose(A, eA))

    def test_gemv(self):
        A = np.random.uniform(0,1,(500,5)).astype(np.float32)
        X = np.random.uniform(0,1, (5)).astype(np.float32)
        gA = clarray.to_device(self.q, A)
        gX = clarray.to_device(self.q, X)
        gY = clarray.zeros(self.q, (500,), np.float32)
        eY = np.dot(A,X)
        ev = clblaswrap.gemv(self.q, gA, gX, gY, False)
        ev.wait()
        Y = gY.get()

        self.assertEqual(Y.shape, eY.shape)
        self.assertEqual(Y.shape, (500,))
        self.assertTrue(np.allclose(Y, eY))

    def test_gemm(self):
        A = np.random.uniform(0, 1, (100, 30)).astype(np.float32)
        B = np.random.uniform(0, 1, (30, 50)).astype(np.float32)
        eC = np.dot(A, B)
        gA = clarray.to_device(self.q, A)
        gB = clarray.to_device(self.q, B)
        gC = clarray.zeros(self.q, (100, 50), np.float32)
        ev = clblaswrap.gemm(self.q, gA, gB, gC)
        ev.wait()
        C = gC.get()

        self.assertEqual(C.shape, eC.shape)
        self.assertTrue(np.allclose(C, eC))

    def test_gemm_batch(self):
        pass
