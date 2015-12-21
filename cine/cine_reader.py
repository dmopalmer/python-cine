"""
Extension to the pims library to read Phantom cine files
based on python-cine
https://github.com/cbenhagen/python-cine
"""

# Copy from pims/norpix_reader.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import range

from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
from pims.utils.misc import FileLocker
import os, struct, itertools
from warnings import warn
import datetime
import numpy as np
from threading import Lock

from . import cine
from . import linLUT

__all__ = ['CineSequence']

class CineSequence(FramesSequence):
    """Read Cine files from Phantom
    """


    @classmethod
    def class_exts(cls):
        return {'cine'} | super(CineSequence, cls).class_exts()

    propagate_attrs = ['frame_shape', 'pixel_type', 'get_time',
                       'get_time_float', 'filename', 'width', 'height',
                       'frame_rate']
