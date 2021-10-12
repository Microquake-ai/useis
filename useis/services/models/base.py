import numpy as np
from pydantic import BaseModel, validator, ValidationError
from enum import Enum


class Phase(str, Enum):
    P = "P"
    S = "S"


class Angle(BaseModel):
    angle: float

    @validator('angle')
    def range(cls, v):
        if -180 >= v > 180:
            raise ValidationError


class Coordinates3D(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_array(cls, array: np.array):
        return cls(x=array[0], y=array[1], z=array[2])

