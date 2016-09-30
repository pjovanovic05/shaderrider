"""PyOpenCL platform specific stuff."""

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray

from shaderrider.aux import clblaswrap


def setup_context(ngpus=0):
    """

    :param ngpus:
    :return:
    """
    ctx = None
    qs = None
    if ngpus > 0:
        ps = cl.get_platforms()
        for p in ps:
            ds = p.get_devices(device_type=cl.device_type.GPU)
            if len(ds) < ngpus:
                continue  # insufficient number of gpus on the platform
            ctx = cl.Context(ds[:ngpus])
            qs = [cl.CommandQueue(ctx, device=d) for d in ds]
            break  # found our platform (probably)
    else:
        ctx = cl.create_some_context()
        qs = [cl.CommandQueue(ctx)]
    clblaswrap.setup()
    return ctx, qs


ctx = None
qs = []


# TODO za kesiranje kernela, u svakom modulu koji pravi svoje kernele pozvati ovako nesto
def precompile_kernels(ctx, q):
    pass


def init_cl(ngpus=0):
    global ctx, qs
    ctx, qs = setup_context(ngpus)


class PyOCLValuation(dict):
    def __init__(self, queue,  *args, **kwargs):
        self._queue = queue
        self.update(*args, **kwargs)

    # def __getitem__(self, key):
    #     val = dict.__getitem__(self, key)
    #     return val

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            val = clarray.to_device(self._queue, value)
        elif isinstance(value, clarray.Array):
            val = value
        else:
            raise TypeError('Unsupported value type')
        dict.__setitem__(self, key, val)

    def set_async(self, key, value):
        if isinstance(value, np.ndarray):
            val = clarray.to_device(self._queue, value, async=True)
            dict.__setitem__(self, key, val)

    def get_async(self, key):
        return self[key].get(async=True)

    def read_value(self, key):
        return self[key].get()

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k,v in dict(*args, **kwargs).iteritems():
            self[k] = v


def valuation(*args, **kwargs):
    return PyOCLValuation(qs[0], *args, **kwargs)
