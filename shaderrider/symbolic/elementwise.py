"""
WRITEME
"""

from shaderrider.symbolic import exprgraph


class ElementwiseOP(exprgraph.Operator):
    _type_name = 'Elwise'

    def __init__(self, expr, ops, parent=None):
        super(ElementwiseOP, self).__init__('Elwise', ops, parent=parent)

    def substitute(self, a, b):
        # TODO should this be supported?
        pass

    def get_shape(self):
        pass

    def gradient(self, wrt):
        pass

    def simplify(self):
        pass
