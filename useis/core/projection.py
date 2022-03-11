import pyproj
from pydantic import BaseModel, validator, conlist


class Latitude(BaseModel):
    latitude: float

    @validator('latitude')
    def range(cls, v):
        assert -90 <= v <= 90, 'Latitude must be between -180 and 180'
        return v


class Longitude(BaseModel):
    longitude: float

    @validator('longitude')
    def range(cls, v):
        assert -180 <= v <= 180, 'Latitude must be between -90 and 90'
        return v


class Projection(object):
    def __init__(self, global_epsg_code: int = 4326,
                 local_epsg_code: int = 32613,
                 offset: conlist(float, min_items=3, max_items=3) = [0, 0, 0]):
        self.global_epsg_code = global_epsg_code
        self.local_epsg_code = local_epsg_code
        self.offset = offset

    def __repr__(self):
        return f"""[projection]
global.epsg = {self.global_epsg_code}
local.epsg = {self.local_epsg_code}
local.offset = {self.offset} 
"""

    def transform_to_local(self, latitude: float = 0, longitude: float = 0,
                           z: float = 0):
        Latitude(latitude=latitude)
        Longitude(longitude=longitude)
        transformer = pyproj.Transformer.from_crs(self.global_epsg_code,
                                                  self.local_epsg_code,
                                                  always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)

        easting += self.offset[0]
        northing += self.offset[1]
        z += self.offset[2]

        return easting, northing, z

    def transform_to_global(self, easting: float = 0, northing: float = 0,
                            z: float = 0):
        transformer = pyproj.Transformer.from_crs(self.local_epsg_code,
                                                  self.global_epsg_code,
                                                  always_xy=True)

        easting -= self.offset[0]
        northing -= self.offset[1]
        z -= self.offset[2]

        longitude, latitude = transformer.transform(easting, northing)

        return longitude, latitude, z

    def write(self, filename):
        with open(filename, 'w') as fout:
            fout.write(str(self))


class UnitConversion(object):
    pass
