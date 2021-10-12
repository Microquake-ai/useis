from .event import SimpleArrival
from pydantic import BaseModel
from typing import Optional, List
from uquake.nlloc import nlloc
from enum import Enum
from datetime import datetime
import numpy as np
from uquake.core import UTCDateTime
from .base import Coordinates3D
from .event import Ray
import useis
from uquake.core.event import Event


class FloatType(str, Enum):
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"


class Site(BaseModel):
    label: str
    location: Coordinates3D


class Srces(BaseModel):
    sites: List[Site]


class Observations(BaseModel):
    picks: List[SimpleArrival]
    p_pick_error: Optional[float] = 1e-3
    s_pick_error: Optional[float] = 1e-3

    class Config:
        orm_mode = True

    @classmethod
    def from_uquake(cls, observations: nlloc.Observations):
        simple_arrivals = []
        for pick in observations.picks:
            simple_arrivals.append(SimpleArrival.from_uquake_pick(pick))

        return cls(picks=simple_arrivals,
                   p_pick_error=observations.p_pick_error,
                   s_pick_error=observations.s_pick_error)

    def to_uquake(self):
        picks = []
        for pick in self.picks:
            picks.append(pick.to_uquake_pick())
        return nlloc.Observations(picks=picks,
                                  p_pick_error=self.p_pick_error,
                                  s_pick_error=self.s_pick_error)

    def to_dict(self):
        picks = []
        for pick in self.picks:
            picks.append(pick.__dict__)

        observations_dict = self.__dict__
        observations_dict['picks'] = picks
        return observations_dict


class NLLOCResults(BaseModel):
    hypocenter: Coordinates3D
    event_time: datetime
    scatter_cloud: List[Coordinates3D]
    rays: List[Ray]
    observations: Observations
    uncertainty: float
    hypocenter_file: str

    @classmethod
    def from_nlloc_results(cls, nlloc_results: useis.procesors.NLLocResults):
        hypocenter = Coordinates3D.from_array(nlloc_results.hypocenter)

        scatter_cloud = []
        for scatter in nlloc_results.scatter_cloud:
            scatter_cloud.append(Coordinates3D.from_array(scatter))

        rays = []
        for ray in nlloc_results.rays:
            rays.append(Ray.from_uquake(ray))

        observations = Observations.from_uquake(nlloc_results.observations)

        return cls(hypocenter=hypocenter, event_time=nlloc_results.event_time,
                   scatter_cloud=scatter_cloud, rays=rays,
                   observations=observations,
                   uncertainty=nlloc_results.uncertainty,
                   hypocenter_file=nlloc_results.hypocenter_file)
