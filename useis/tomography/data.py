#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
#
#
__doc__ = \
"""
TODO : BIG DESCRIPTION OF EXCHANGE DATA
"""

import numpy as np
import pickle
import copy
from functools import wraps


def memoize(fct):
    """
    This is a decorator which cache the result of a function based on the
    given parameter.
    """
    return_dict = {}

    @wraps(fct)
    def wrapper(*args, **kwargs):
        if args not in return_dict:
            return_dict[args] = fct(*args, **kwargs)
        return return_dict[args]
    return wrapper

# This is the data description for the input array describing the event
#
ev_dtype = [('name',       'str'),
            ('id',          'int'),
            ('position',    'float',    (3,)),
            ('delta_t',     'float')]

st_dtype = ev_dtype

tt_dtype = [('id',          'int'),
            ('event_id',    'int'),
            ('traveltime',  'float')]


class EKTTTable(object):
    """
    This object represent a
    :param data: u
    :param staid: u
    :param evnfile: u
    :param stafile: u
    """
    dtype = tt_dtype

    def __init__(self, data, staid, evnfile=None, stafile=None):
        try:
            for tname, ttype in tt_dtype:
                data[tname]
            import sys
            sys.stderr.write(str(data.size))
            sys.stderr.flush()
            data = np.array(data.__getitem__([tname for tname, ttype in 
                                              tt_dtype]), dtype = self.dtype)
        except ValueError as e:
            data = np.asarray(data, dtype = self.dtype)

        self.data               = data
        self.data.dtype         = self.dtype
        self.station_id         = staid

        self.__evn_file__   = evnfile
        self.__sta_file__   = stafile

    def __get_event_rows__(self):
        return self.event_table.data[self.data['event_id']]
    event_rows = property(__get_event_rows__)

    @memoize
    def __get_event_table__(self):
        return pickle.load(open(self.__evn_file__))
    event_table = property(__get_event_table__)

    def __get_station_row__(self):
        return self.station_table.data[self.station_id]
    station_row = property(__get_station_row__)

    @memoize
    def __get_station_table__(self):
        return pickle.load(open(self.__sta_file__))
    station_table = property(__get_station_table__)


class EKPunctualData(object):
    dtype = None

    def __init__(self, data, origin=None, scale=1):
        try:
            for tname, ttype in self.dtype:
                data[tname]
            self.data = data
        except ValueError as e:
            self.data = np.asarray(data, dtype=self.dtype)

        self.origin = tuple([0] * data['position'].shape[-1]) \
            if origin is None else origin
        self.scale = scale

    def __get_position_zyx__(self):
        return self.data['position']

    def __set_position_zyx__(self, pos):
        self.data['position'] = pos
    position_zyx = property(__get_position_zyx__, __set_position_zyx__)

    def __get_position_xyz__(self):
        position_zyx = self.data['position']
        position_xyz = np.empty_like(position_zyx)
        for i in range(position_zyx.shape[1]):
            position_xyz[:, i] = position_zyx[:, -(i + 1)]
        return position_xyz

    position_xyz = property(__get_position_xyz__)

    def __get_size__(self):
        return self.data.size
    size = property(__get_size__)

    def add_gaussian_noise(self, position=0, time=0):
        pass


class EKEventTable(EKPunctualData):
    dtype = ev_dtype


class EKStationTable(EKPunctualData):
    dtype = st_dtype


class EKImageData(object):
    """
    This object represent a geometric structure that representing a regular
    array of points positionned in space.

    :param shape_or_data: The numpy array or the shape of the underlying data
    :param spacing: The spacing of the grid
    :param origin: A Tuple representing the position of the lower left corner \
            of the grid
    """
    def __init__(self, shape_or_data, spacing = 1, origin = None):
        if isinstance(shape_or_data, np.ndarray):
            self.data = shape_or_data
        else:
            self.data = np.zeros(shape_or_data)

        self.origin = tuple([0] * self.data.ndim) if origin is None else origin
        self.spacing = spacing

    def transform_to(self, values):
        return (values - self.origin) / self.spacing

    def transform_from(self, values):
        return (values + self.origin) * self.spacing

    def check_compatibility(self, other):
        return (self.shape == other.shape) and \
                (self.spacing == other.spacing) and \
                (self.origin == other.origin)

    def __get_shape__(self):
        return self.data.shape
    shape = property(__get_shape__)

    def homogenous_like(self, value):
        data = np.empty_like(self.data)
        data.fill(value)
        return EKImageData(data, self.spacing, origin=self.origin)

    def copy(self):
        cp = copy.deepcopy(self)
        return cp

    def SaveNLL(self):
        pass

    def LoadNLL(self):
        pass

    ### ADDING method to save and load on data from/to NLL




