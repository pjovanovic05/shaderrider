"""
Linear algebra functions.

Uses clBLAS as optimization when possible
"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider.aux import clblaswrap


def dot(queue, a, b, out=None, transA=False, transB=False, wait_for=None):
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
        return a*b, []

    # if lsa > 2:
    #     raise TypeError('A is not a matrix')
    # if lsb > 2:
    #     raise TypeError('B is not a matrix')
    M, N, K = 1, 1, 1
    if a.ndim >= 2:
        M, K = a.shape[-2:]
    elif a.ndim == 1:
        M, K = a.shape[-1], 1
    if transA:
        M, K = K, M
    if b.ndim >= 2:
        N = b.shape[-1]
        if transB:
            N = b.shape[-2]
    elif b.ndim == 1:
        N = b.shape[-1]

    # M, K = a.shape if len(a.shape) == 2 else (a.shape[0], 1)
    # N = b.shape[1] if len(b.shape) == 2 else 1
    evl = []

    if (lsa == 1 and lsb == 1) or (M == 1 and N == 1) or (K == 1 and M == N):
        # dot
        if out is None:
            out = clarray.zeros(queue, (1, 1), a.dtype)
        scratch = clarray.zeros_like(a)
        ev = clblaswrap.dot(queue, a, b, out, scratch)
        evl = [ev]
    elif (lsa > 1 and lsb == 1):
        # gemv
        if out is None:
            out = clarray.zeros(queue, (M, N), a.dtype)
        ev = clblaswrap.gemv(queue, a, b, out, transA=transA)
        evl = [ev]
    elif (lsa == 2 and lsb == 2):
        # gemm
        if out is None:
            out = clarray.zeros(queue, (M, N), a.dtype)
        ev = clblaswrap.gemm(queue, a, b, out, transA=transA, transB=transB)
        evl = [ev]
    elif (lsa > 2) and (lsb >= 2):
        d = np.prod(a.shape[:-2])
        if out is None:
            out = clarray.zeros(queue, (d, M, N), a.dtype)
        evl = clblaswrap.gemm_batch(queue, a, b, out, transA=transA, transB=transB)   # TODO transpositions?
    for ev in evl:
        ev.wait()
    return out


def batch_dot(queue, a, b, out=None, wait_for=None):
    lsa, lsb = len(a.shape), len(b.shape)
    M, K = a.shape if len(a.shape) == 2 else (a.shape[0], 1)
    N = b.shape[1] if len(b.shape) == 2 else 1
    evl = []
    if (lsa > 2) and (lsb >= 2):
        d = np.prod(a.shape[:-2])
        if out is None:
            out = clarray.zeros(queue, (d, M, N), a.dtype)
        evl = clblaswrap.gemm_batch(queue, a, b, out)   # TODO transpositions?

    return out, evl


def outer(queue, a, b, out=None, wait_for=None):
    pass
