"""Root of the KerPy module"""
from . import (diff, processing, shapes, objs)
from .objs.Kernel import Kernel

from . import _version
__version__ = _version.get_versions()['version']
