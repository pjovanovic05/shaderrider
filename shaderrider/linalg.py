"""
Linear algebra functions.

Uses clBLAS as optimization when possible
"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider.aux import clblaswrap
from shaderrider import clplatf


def dot(a, b, axes=2, out=None):
    # TODO a moz biti skalar, vektor ili matrica
    if len(a.shape) == 1:
        # TODO
        if len(b.shape) == 1:
            # vector-vector (inner or outer) product
            pass
        elif len(b.shape) == 2:
            # TODO ?
            pass
    elif len(a.shape) == 2:
        M, K = a.shape
        # TODO b moze biti skalar, vektor ili matrica
        if len(b.shape) == 1:
            pass
        elif len(b.shape) == 2:
            N = b.shape[1]
            if out is not None:
                # TODO check if out is large enough and has correct shape
                pass
            else:
                # TODO allocate out array
                pass
            ev = clblaswrap.gemm(clplatf.qs[0], a, b, c)
    else:
        # TODO batch matrix multiplication... last 2 dims are mm
        pass
    return out  # TODO return i event
