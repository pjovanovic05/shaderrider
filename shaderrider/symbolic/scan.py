"""
scan operator
"""

from shaderrider.symbolic import exprgraph


# prvo cu napraviti simple scan op koji ce direktno pozivati scan iz pypencl-a
# za posle cemo videti da li treba generalniji sken ili su for petlje dovoljne

class ScanOP(exprgraph.Operator):
    _type_name = 'Scan'
    def __init__(self, inputs, ):
        pass