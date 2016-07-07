"""
Holds platform loading and configuration logic
"""

from shaderrider.platform.pyocl.platformdef import PyOCLFactory
from shaderrider.platform.numpy.platformdef import NPFactory


class Configuration(object):
    pass


_platforms = {
    'numpy': NPFactory(),
    'pyopencl': PyOCLFactory()
}

def get_platform_factory(platform='numpy'):
    return _platforms[platform]
