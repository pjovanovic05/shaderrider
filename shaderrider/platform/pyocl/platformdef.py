"""
Defines PyOpenCL platform.
"""

import pyopencl as cl

from shaderrider.symbolic import exprgraph
from shaderrider.symbolic import basic as sbo
from shaderrider.symbolic import blas as sblas
from shaderrider.symbolic import elementwise as sew

from shaderrider.platform.pyocl import basic as bo
from shaderrider.platform.pyocl import blas
from shaderrider.platform.pyocl import elementwise


class PyOCLPlatform(object):
    def __init__(self, ngpus=0):
        self._ngpus = ngpus
        self._setup_context(ngpus)

        self._opevals = {
            # unary
            sbo.NegOP.get_type_name(): bo.NegOP(),
            sbo.ExpOP.get_type_name(): bo.ExpEval(),
            sbo.LogOP.get_type_name(): bo.LogEval(),
            sbo.SinOP.get_type_name(): bo.SinEval(),
            sbo.CosOP.get_type_name(): bo.CosEval(),
            sbo.TanOP.get_type_name(): bo.TanEval(),
            # binary
            sbo.AddOP.get_type_name(): bo.TanEval(),
            sbo.SubOP.get_type_name(): bo.SubEval(),
            sbo.MulOP.get_type_name(): bo.MulEval(),
            sbo.DivOP.get_type_name(): bo.DivEval(),
            sbo.PowOP.get_type_name(): bo.PowEval(),
            # comparisons
            sbo.EqOP.get_type_name(): bo.EqEval(),
            sbo.GtOP.get_type_name(): bo.GtEval(),
            sbo.LtOP.get_type_name(): bo.LtEval(),
            sbo.GeOP.get_type_name(): bo.GeEval(),
            sbo.LeOP.get_type_name(): bo.LeEval(),
            sbo.NeOP.get_type_name(): bo.NeEval(),
            # blas
            sblas.GemmOP.get_type_name(): blas.GemmEval(),
            sblas.GemvOP.get_type_name(): blas.GemvEval(),
            sblas.GerOP.get_type_name(): blas.GerEval()
        }
        # these should include elementwise, reduce, scan...
        self._opgens = {
            # elementwise
            sew.ElementwiseOP.get_type_name(): elementwise.ElementwiseGenerator()
        }

    def get_op_evaluators(self):
        return self._opevals

def setup_context(ngpus=0):
    ctx = None
    if ngpus>0:
        ps = cl.get_platforms()
        for p in ps:
            ds = p.get_devices(device_type=cl.device_type.GPU)
            if len(ds) < ngpus:
                continue    # insufficient number of gpus on the platform
            ctx = cl.Context(ds[:ngpus])
            queues = [cl.CommandQueue(ctx, device=d) for d in ds]
            break   # found our platform (probably)
    else:
        ctx = cl.create_some_context()
        queues = [cl.CommandQueue(ctx)]
    return (ctx, queues)


default_ctx, queues = setup_context(1)
default_queue = queues[0]
