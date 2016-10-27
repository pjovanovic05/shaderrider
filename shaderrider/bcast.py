"""Broadcast-able operations."""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf as pl
from shaderrider.utils import yaptu


_kernel_template = r"""
__kernel void {{kname}}(
        const int n
{@ for dim in range(nd):
        , const int dim{{dim}}
@}
{@ for i in range(nargs):
        , __global {{dtype}} *arg{{i}}_data
        , int arg{{i}}_off
{@ for dim in range(nd):
        , int arg{{i}}_stride{{dim}}
@}
@}
        ) {
    const int idx = get_global_id(0);

    // rewind data pointers to offsets
    __global char *tmp;
{@ for i in range(nargs):
    tmp = (__global char*) arg{{i}}_data;
    tmp += arg{{i}}_off;
    arg{{i}}_data = (__global {{dtype}}*) tmp;
@}

    // rewind args to the element position
    int ii = idx;
    int pos;
{@ for i in range(nargs):
    __global char *arg{{i}}_p = (__global char*)arg{{i}}_data;
@}

{@ for d in range(nd-1, -1, -1):
{@ if d > 0:
    pos = ii % dim{{d}};
    ii /= dim{{d}};
@@ else:
    pos = ii;
@}
{@ for i in range(nargs):
    arg{{i}}_p += pos * arg{{i}}_stride{{d}};
@}
@}

{@ for i in range(nargs):
    __global {{dtype}} *arg{{i}} = (__global {{dtype}} *)arg{{i}}_p;
@}

    {{expression}}
}
"""


_kernel_cache = {}


def bcast_add(A, B, out=None):
    q = pl.qs[0]
    nda, ndb = A.ndim, B.ndim
    a_shape, b_shape = list(A.shape), list(B.shape)
    a_strides, b_strides = list(A.strides), list(B.strides)
    ndim = max(nda, ndb)
    dtype = 'float' if A.dtype == np.float32 else 'double'

    if nda != ndb:
        # TODO extend smaller shape with 1s
        # extend smaller strides with 0s
        if nda > ndb:
            b_shape = [1] * (nda - ndb) + b_shape
            b_strides = [0] * (nda - ndb) + b_strides
        if nda < ndb:
            a_shape = [1] * (ndb - nda) + a_shape
            b_strides = [0] * (ndb - nda) + a_strides

    # check broadcasting compatibility
    c_shape = []
    for i in range(ndim):
        # TODO dimenzije jednake ili je jedna odnjih == 1
        if a_shape[i] != b_shape[i]:
            if a_shape[i] == 1:
                a_strides[i] = 0
            if b_shape[i] == 1:
                b_strides[i] = 0
            if a_shape[i] != 1 and b_shape[i] != 1:
                raise ValueError('Incompatible broadcasting shapes: ' +
                                 str(a_shape) + ' & ' + str(b_shape))
        c_shape.append(max(a_shape[i], b_shape[i]))

    # generate kernel
    kname = 'ewk_add_%s_%d' % (dtype, ndim)
    if kname not in _kernel_cache:
        pass    # TODO
        kdesc = {
            'nargs': 3,
            'nd': ndim,
            'kname': kname,
            'expression': '*arg2 = *arg0 + *arg1;',
            'dtype': dtype
        }
        ksource = yaptu.generate_kernel(_kernel_template, kdesc)
        prg = cl.Program(pl.ctx, ksource).build()
        _kernel_cache[kname] = eval('prg.' + kname)
    # krnl = eval('prg.'+kdesc['kname'])

    krnl = _kernel_cache[kname]

    if out is None:
        # c_shape = a_shape if nda > ndb else b_shape
        out = clarray.empty(q, tuple(c_shape), dtype=A.dtype)

    launch_sz = int(np.prod(out.shape))
    evt = krnl(q, (launch_sz,), None,
               np.int32(launch_sz),
               np.int32(out.shape[0]),
               np.int32(out.shape[1]),
               A.base_data,
               np.int32(A.offset),
               np.int32(a_strides[0]),
               np.int32(a_strides[1]),
               B.base_data,
               np.int32(B.offset),
               np.int32(b_strides[0]),
               np.int32(b_strides[1]),
               out.base_data,
               np.int32(out.offset),
               np.int32(out.strides[0]),
               np.int32(out.strides[1]))

    return out, evt


def bcast_mul(A, B, out=None):
    q = pl.qs[0]
    nda, ndb = A.ndim, B.ndim
    a_shape, b_shape = list(A.shape), list(B.shape)
    a_strides, b_strides = list(A.strides), list(B.strides)
    ndim = max(nda, ndb)
    dtype = 'float' if A.dtype == np.float32 else 'double'

    if nda != ndb:
        # TODO extend smaller shape with 1s
        # extend smaller strides with 0s
        if nda > ndb:
            b_shape = [1]*(nda-ndb) + b_shape
            b_strides = [0]*(nda-ndb) + b_strides
        if nda < ndb:
            a_shape = [1]*(ndb-nda) + a_shape
            b_strides = [0]*(ndb-nda) + a_strides

    # check broadcasting compatibility
    c_shape = []
    for i in range(ndim):
        # TODO dimenzije jednake ili je jedna odnjih == 1
        if a_shape[i] != b_shape[i]:
            if a_shape[i] == 1:
                a_strides[i] = 0
            if b_shape[i] == 1:
                b_strides[i] = 0
            if a_shape[i] != 1 and b_shape[i] != 1:
                raise ValueError('Incompatible broadcasting shapes: ' +
                                 str(a_shape)+' & '+str(b_shape))
        c_shape.append(max(a_shape[i], b_shape[i]))

    # generate kernel
    kdesc = {
        'nargs': 3,
        'nd': ndim,
        'kname': 'ewk_mul_%s_%d' % (dtype, ndim),
        'expression': '*arg2 = *arg0 * *arg1;',
        'dtype': dtype
    }
    ksource = yaptu.generate_kernel(_kernel_template, kdesc)
    prg = cl.Program(pl.ctx, ksource).build()
    krnl = eval('prg.'+kdesc['kname'])

    if out is None:
        # c_shape = a_shape if nda > ndb else b_shape
        out = clarray.empty(q, tuple(c_shape), dtype=A.dtype)

    launch_sz = int(np.prod(out.shape))
    evt = krnl(q, (launch_sz,), None,
               np.int32(launch_sz),
               np.int32(out.shape[0]),
               np.int32(out.shape[1]),
               A.base_data,
               np.int32(A.offset),
               np.int32(a_strides[0]),
               np.int32(a_strides[1]),
               B.base_data,
               np.int32(B.offset),
               np.int32(b_strides[0]),
               np.int32(b_strides[1]),
               out.base_data,
               np.int32(out.offset),
               np.int32(out.strides[0]),
               np.int32(out.strides[1]))
    evt.wait()
    return out


def bcast_sub(A, B, out=None):
    q = pl.qs[0]
    nda, ndb = A.ndim, B.ndim
    a_shape, b_shape = list(A.shape), list(B.shape)
    a_strides, b_strides = list(A.strides), list(B.strides)
    ndim = max(nda, ndb)
    dtype = 'float' if A.dtype == np.float32 else 'double'

    if nda != ndb:
        # TODO extend smaller shape with 1s
        # extend smaller strides with 0s
        if nda > ndb:
            b_shape = [1] * (nda - ndb) + b_shape
            b_strides = [0] * (nda - ndb) + b_strides
        if nda < ndb:
            a_shape = [1] * (ndb - nda) + a_shape
            b_strides = [0] * (ndb - nda) + a_strides

    # check broadcasting compatibility
    c_shape = []
    for i in range(ndim):
        # TODO dimenzije jednake ili je jedna odnjih == 1
        if a_shape[i] != b_shape[i]:
            if a_shape[i] == 1:
                a_strides[i] = 0
            if b_shape[i] == 1:
                b_strides[i] = 0
            if a_shape[i] != 1 and b_shape[i] != 1:
                raise ValueError('Incompatible broadcasting shapes: ' +
                                 str(a_shape) + ' & ' + str(b_shape))
        c_shape.append(max(a_shape[i], b_shape[i]))

    # generate kernel
    kdesc = {
        'nargs': 3,
        'nd': ndim,
        'kname': 'ewk_sub_%s_%d' % (dtype, ndim),
        'expression': '*arg2 = *arg0 - *arg1;',
        'dtype': dtype
    }
    ksource = yaptu.generate_kernel(_kernel_template, kdesc)
    prg = cl.Program(pl.ctx, ksource).build()
    krnl = eval('prg.'+kdesc['kname'])

    if out is None:
        # c_shape = a_shape if nda > ndb else b_shape
        out = clarray.empty(q, tuple(c_shape), dtype=A.dtype)

    launch_sz = int(np.prod(out.shape))
    evt = krnl(q, (launch_sz,), None,
               np.int32(launch_sz),
               np.int32(out.shape[0]),
               np.int32(out.shape[1]),
               A.base_data,
               np.int32(A.offset),
               np.int32(a_strides[0]),
               np.int32(a_strides[1]),
               B.base_data,
               np.int32(B.offset),
               np.int32(b_strides[0]),
               np.int32(b_strides[1]),
               out.base_data,
               np.int32(out.offset),
               np.int32(out.strides[0]),
               np.int32(out.strides[1]))
    evt.wait()
    return out


def bcast_div(A, B, out=None):
    q = pl.qs[0]
    nda, ndb = A.ndim, B.ndim
    a_shape, b_shape = list(A.shape), list(B.shape)
    a_strides, b_strides = list(A.strides), list(B.strides)
    ndim = max(nda, ndb)
    dtype = 'float' if A.dtype == np.float32 else 'double'

    if nda != ndb:
        # TODO extend smaller shape with 1s
        # extend smaller strides with 0s
        if nda > ndb:
            b_shape = [1] * (nda - ndb) + b_shape
            b_strides = [0] * (nda - ndb) + b_strides
        if nda < ndb:
            a_shape = [1] * (ndb - nda) + a_shape
            b_strides = [0] * (ndb - nda) + a_strides

    # check broadcasting compatibility
    c_shape = []
    for i in range(ndim):
        # TODO dimenzije jednake ili je jedna odnjih == 1
        if a_shape[i] != b_shape[i]:
            if a_shape[i] == 1:
                a_strides[i] = 0
            if b_shape[i] == 1:
                b_strides[i] = 0
            if a_shape[i] != 1 and b_shape[i] != 1:
                raise ValueError('Incompatible broadcasting shapes: ' +
                                 str(a_shape) + ' & ' + str(b_shape))
        c_shape.append(max(a_shape[i], b_shape[i]))

    # generate kernel
    kdesc = {
        'nargs': 3,
        'nd': ndim,
        'kname': 'ewk_div_%s_%d' % (dtype, ndim),
        'expression': '*arg2 = *arg0 / *arg1;',
        'dtype': dtype
    }
    ksource = yaptu.generate_kernel(_kernel_template, kdesc)
    prg = cl.Program(pl.ctx, ksource).build()
    krnl = eval('prg.'+kdesc['kname'])

    if out is None:
        # c_shape = a_shape if nda > ndb else b_shape
        out = clarray.empty(q, tuple(c_shape), dtype=A.dtype)

    launch_sz = int(np.prod(out.shape))
    evt = krnl(q, (launch_sz,), None,
               np.int32(launch_sz),
               np.int32(out.shape[0]),
               np.int32(out.shape[1]),
               A.base_data,
               np.int32(A.offset),
               np.int32(a_strides[0]),
               np.int32(a_strides[1]),
               B.base_data,
               np.int32(B.offset),
               np.int32(b_strides[0]),
               np.int32(b_strides[1]),
               out.base_data,
               np.int32(out.offset),
               np.int32(out.strides[0]),
               np.int32(out.strides[1]))
    evt.wait()
    return out
