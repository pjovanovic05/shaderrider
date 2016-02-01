"""
WRITEME
"""

from shaderrider.symbolic import exprgraph


class ElementwiseOP(exprgraph.Operator):
    def substitute(self, a, b):
        pass

    def get_shape(self):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass
