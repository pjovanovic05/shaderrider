"""
Holds platform loading and configuration logic
"""

from shaderrider.platform.pyocl.platformdef import PyOCLFactory
from shaderrider.platform.numpy.platformdef import NPFactory


_platforms = {
    'numpy': NPFactory(),
    'pyopencl': PyOCLFactory()
}
default_platform = 'numpy'


def get_platform_factory(platform=None):
    if platform is None:
        platform = default_platform
    return _platforms[platform]


def set_platform(platform, **options):
    global  default_platform
    default_platform = platform
    if platform == 'pyopencl':
        ngpus = 0
        if 'ngpus' in options:
            ngpus = options['ngpus']
        _platforms[platform].init_platform(ngpus)


def init_platform(platform, **options):
    _platforms[platform].init_platform(options)
