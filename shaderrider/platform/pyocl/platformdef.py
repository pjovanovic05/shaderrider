"""
Defines PyOpenCL platform.
"""

from shaderrider.symbolic import vast
from shaderrider.symbolic import basic as sbo
from shaderrider.symbolic import blas as sblas
from shaderrider.symbolic import elementwise as sew

from shaderrider.platform.pyocl import basic as bo
from shaderrider.platform.pyocl import blas
from shaderrider.platform.pyocl import elementwise


class PyOCLPlatform(object):
    def __init__(self):
        self._opevals = {
            # unary
            sbo.NegOP.getTypeName(): bo.NegEval(),
            sbo.ExpOP.getTypeName(): bo.ExpEval(),
            sbo.LogOP.getTypeName(): bo.LogEval()
            # TODO stao ovde!! Treba da napisem evaluatore za sve ostale operatore i da ih ovde instanciram

            # binary
            # comparisons
            # blas
            # elementwise
        }

    def get_op_evaluators(self):
        return self._opevals
