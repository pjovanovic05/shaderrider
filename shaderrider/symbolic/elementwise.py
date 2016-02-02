"""
WRITEME
"""

from shaderrider.symbolic import exprgraph


class ElementwiseOP(exprgraph.Operator):
    _type_name = 'Elwise'

    def __init__(self, expr):
        super(ElementwiseOP, self).__init__('Elwise', expr.get_atoms())

    def substitute(self, a, b):
        # TODO should this be supported?
        pass

    def get_shape(self):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass
