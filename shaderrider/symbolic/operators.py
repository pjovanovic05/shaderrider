"""
All the operator types.

WRITEME
"""
import abc


from shaderrider.core import IncompatibleDimensionsError, NondifferentiableOpError
from shaderrider.symbolic import exprgraph


# ABSTRACT OPS ########################################################

class UnaryOP(exprgraph.Operator):
    """docstring for UnaryOP"""
    __metaclass__ = abc.ABCMeta

    def __eq__(self, other):
        return type(self) == type(other) and \
               self.operands[0] == other.operands[0]

    def get_shape(self):
        return self.operands[0].get_shape()


class BinaryOP(exprgraph.Operator):
    """docstring for BinaryOP"""
    __metaclass__ = abc.ABCMeta
    isCommutative = False
    isAssociative = False

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.operands[0] == other.operands[0]) and \
               (self.operands[1] == other.operands[1])

    def get_shape(self):  # TODO move this to broadcastable or elementwise
        ds = []
        for d1, d2 in zip(self.operands[0].get_shape(), self.operands[1].get_shape()):
            if d1 == 1 or d2 == 1 or d1 == d2:
                ds.append(max(d1, d2))
            else:
                # TODO print offending dimensions to exception message
                raise IncompatibleDimensionsError
        return tuple(ds)



class Add(BinaryOP):
    def __init__(self, op1, op2, parents=None):
        super(Add, self).__init__(parents)
        self.ops = [op1, op2]
