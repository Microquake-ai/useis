from .event import SimpleArrival
from pydantic import BaseModel
from typing import Optional, List
from uquake.nlloc import nlloc
from enum import Enum
from datetime import datetime
import numpy as np
from uquake.core import UTCDateTime


class Coordinates3D(BaseModel):
    x: float
    y: float
    z: float


class FloatType(str, Enum):
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"


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
    hypocenter: List[float]
    event_time: datetime

    def __init__(self, hypocenter: np.array, event_time: UTCDateTime,
                 scatter_cloud: np.ndarray, rays: list,
                 observations: Observations, evaluation_mode: str,
                 evaluation_status: str):
        self.hypocenter = hypocenter
        self.event_time = event_time
        self.scatter_cloud = scatter_cloud
        self.rays = rays
        self.observations = observations
        self.evaluation_mode = evaluation_mode
        self.evaluation_status = evaluation_status

        self.uncertainty_ellipsoid = calculate_uncertainty(
            self.scatter_cloud[:, :-1])

        self.creation_info = CreationInfo(author='uQuake-nlloc',
                                          creation_time=UTCDateTime.now())


    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    # @classmethod
    # def from_uq_observations(cls, observations: nlloc.Observations):
    #     """
    #     create observation model from uquake observations
    #     :param observations: observation
    #     :type observations: uquake.nlloc.nlloc.Observations
    #     :return:
    #     """
    #     picks = []
    #     for pick in observations.picks:
    #         picks.append(Pick.from_uq_pick(pick))
    #     return cls(picks, observations.p_pick_error, observations.s_pick_error)
