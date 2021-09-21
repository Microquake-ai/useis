from pydantic import BaseModel
from typing import Optional, List
from .nlloc import Coordinates3D
from uquake.grid.nlloc import VelocityGrid3D as UQVelocityGrid3D
from uquake.core.event import ResourceIdentifier
import numpy as np


class VelocityGrid3D(BaseModel):
    data: List[float]
    origin: List[float]
    spacing: List[float]
    shape: List[int]
    phase: str
    model_id: str
    float_type: Optional[str] = 'FLOAT'

    class Config:
        orm_mode = True

    @classmethod
    def from_uquake(cls, uq_velocity_grid_3d: UQVelocityGrid3D):
        return cls(data=list(uq_velocity_grid_3d.data.ravel()),
                   origin=list(uq_velocity_grid_3d.origin),
                   spacing=list(uq_velocity_grid_3d.spacing),
                   shape=list(uq_velocity_grid_3d.shape),
                   phase=uq_velocity_grid_3d.phase,
                   model_id=str(uq_velocity_grid_3d.model_id),
                   float_type=uq_velocity_grid_3d.float_type)

    def to_uquake(self, network_code):
        return UQVelocityGrid3D(network_code, self.reshaped_data, self.origin,
                                self.spacing, phase=self.phase,
                                model_id=self.model_id,
                                float_type=self.float_type)

    def to_dict(self):
        return self.__dict__

    @property
    def reshaped_data(self):
        return np.array(self.data).reshape(self.shape)


