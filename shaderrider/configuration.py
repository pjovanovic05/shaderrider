"""
Holds platform loading and configuration logic
"""

# from shaderrider.platform.pyocl.platformdef import PyOCLFactory
from shaderrider.platform.numpy.platformdef import NPFactory


class Configuration(object):
    pass


def get_platform():
    raise NotImplementedError


def get_formula_factory():
    pass


def get_platform_factory():
    pass
