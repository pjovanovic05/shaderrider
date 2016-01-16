"""
Defines PyOpenCL platform.
"""

import pyopencl as cl

from shaderrider.symbolic import vast
from shaderrider.symbolic import basic as sbo
from shaderrider.symbolic import blas as sblas
from shaderrider.symbolic import elementwise as sew

from shaderrider.platform.pyocl import basic as bo
from shaderrider.platform.pyocl import blas
from shaderrider.platform.pyocl import elementwise


class PyOCLPlatform(object):
    def __init__(self, ngpus=0):
        self._ngpus = ngpus

        self._opevals = {
            # unary
            sbo.NegOP.getTypeName(): bo.NegEval(),
            sbo.ExpOP.getTypeName(): bo.ExpEval(),
            sbo.LogOP.getTypeName(): bo.LogEval(),
            sbo.SinOP.getTypeName(): bo.SinEval(),
            sbo.CosOP.getTypeName(): bo.CosEval(),
            sbo.TanOP.getTypeName(): bo.TanEval(),
            # binary
            sbo.AddOP.getTypeName(): bo.TanEval(),
            sbo.SubOP.getTypeName(): bo.SubEval(),
            sbo.MulOP.getTypeName(): bo.MulEval(),
            sbo.DivOP.getTypeName(): bo.DivEval(),
            sbo.PowOP.getTypeName(): bo.PowEval(),
            # comparisons
            sbo.EqOP.getTypeName(): bo.EqEval(),
            sbo.GtOP.getTypeName(): bo.GtEval(),
            sbo.LtOP.getTypeName(): bo.LtEval(),
            sbo.GeOP.getTypeName(): bo.GeEval(),
            sbo.LeOP.getTypeName(): bo.LeEval(),
            sbo.NeOP.getTypeName(): bo.NeEval(),
            # blas
            sblas.GemmOP.getTypeName(): blas.GemmEval(),
            sblas.GemvOP.getTypeName(): blas.GemvEval(),
            sblas.GerOP.getTypeName(): blas.GerEval()
            # elementwise
        }

    def get_op_evaluators(self):
        return self._opevals

    def _setup_context(self, ngpus=0):
        if ngpus>0:
            ps = cl.get_platforms()
            for p in ps:
                ds = p.get_devices(device_type=cl.device_type.GPU)
                if len(ds) < ngpus:
                    continue    # insufficient number of gpus on the platform
                self._ctx = cl.Context(ds[:ngpus])
                self._queues = [cl.CommandQueue(self._ctx, device=d) for d in ds]
                break   # found our platform (probably)
        else:
            self._ctx = cl.create_some_context()
            self._queues = [cl.CommandQueue(self._ctx)]
        # TODO import and setup blas (first compile it though)
