"""
Linear algebra functions.

Uses clBLAS as optimization when possible
"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider.aux import clblaswrap


def dot(queue, a, b, out=None, wait_for=None):
    lsa, lsb = 0, 0
    try:
        lsa = len(a.shape)
    except AttributeError:
        lsa = 0     # scalar
    try:
        lsb = len(b.shape)
    except AttributeError:
        lsb = 0     # scalar

    if lsa == 0 or lsb == 0:
        return a*b, None

    if lsa > 2:
        raise TypeError('A is not a matrix')
    if lsb > 2:
        raise TypeError('B is not a matrix')

    M, K = a.shape if len(a.shape) == 2 else (a.shape[0], 1)
    N = b.shape[1] if len(b.shape) == 2 else 1
    ev = None

    if (lsa == 1 and lsb == 1) or (M == 1 and N == 1) or (K == 1 and M == N):
        # dot
        if out is None:
            out = clarray.empty(queue, (1, 1), a.dtype)
        scratch = clarray.empty_like(a, queue=queue)
        ev = clblaswrap.dot(queue, a, b, out, scratch)
    elif (lsa > 1 and lsb == 1):
        # gemv
        if out is None:
            out = clarray.empty(queue, (M, N), a.dtype)
        ev = clblaswrap.gemv(queue, a, b, out)
    elif (lsa == 2 and lsb == 2):
        # gemv
        if out is None:
            out = clarray.empty(queue, (M, N), a.dtype)
        ev = clblaswrap.gemm(queue, a, b, out)

    return out, ev


def batch_dot(queue, a, b, out=None, wait_for=None):
    pass


def outer(queue, a, b, out=None, wait_for=None):
    pass

# TODO batch gemm
