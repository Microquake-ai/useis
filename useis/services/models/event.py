from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime
import uquake
from uquake.core.event import Pick as UQPick


# class Phase(BaseModel, Enum):
#     P = "P"
#     S = "S"


class WaveformStreamID(BaseModel):
    station_code: Optional[str]
    channel_code: Optional[str]
    location_code: Optional[str]


class QuantityError(BaseModel):
    uncertainty: Optional[float]
    lower_uncertainty: Optional[float]
    upper_uncertainty: Optional[float]
    confidence_level: Optional[float]


class PickOnset(str, Enum):
    emergent = "emergent"
    impulsive = "impulsive"
    questionable = "questionable"


class Phase(str, Enum):
    P = 'P'
    S = 'S'


class PickPolarity(str, Enum):
    positive = "positive"
    negative = "negative"
    undecidable = "undecidable"


class EvaluationMode(str, Enum):
    manual = "manual"
    automatic = "automatic"


class EvaluationStatus(str, Enum):
    preliminary = "preliminary"
    confirmed = "confirmed"
    reviewed = "reviewed"
    final = "final"
    rejected = "rejected"
    reported = "reported"


class CreationInfo(BaseModel):
    agency_id: Optional[str]
    agency_uri: Optional[str]
    author: Optional[str]
    author_uri: Optional[str]
    creation_time: Optional[datetime]
    version: Optional[str]


class ResourceIdentifier(BaseModel):
    fixed: bool = False
    _prefix: str = 'smi:local'
    _uuid: str

    class Config:
        orm_mode = True


class SimpleArrival(BaseModel):
    time: datetime
    station_code: str
    location_code: str
    channel_code: str
    phase: Phase
    onset: PickOnset
    evaluation_mode: EvaluationMode
    evaluation_status: EvaluationStatus

    class Config:
        orm_mode = True

    @classmethod
    def from_uquake_pick(cls, pick: uquake.core.event.Pick):
        """
        convert a uquake Pick object into a SimpleArrival
        :param pick: a pick object
        :type pick: uquake.core.event.Pick
        :return:  a SimpleArrival
        :rtype: SimpleArrival
        """
        return cls(time=pick.time, station_code=pick.waveform_id.station_code,
                   location_code=pick.waveform_id.location_code,
                   channel_code=pick.waveform_id.channel_code,
                   phase=pick.phase_hint, onset=pick.onset,
                   evaluation_mode=pick.evaluation_mode,
                   evaluation_status=pick.evaluation_status)

    def to_uquake_pick(self) -> uquake.core.event.Pick:
        """
        convert to uquake Pick
        :return: uquake pick
        :rtype: uquake.core.event.Pick
        """
        waveform_id = WaveformStreamID(station_code=self.station_code,
                                       channel_code=self.channel_code,
                                       location_code=self.location_code)
        return UQPick(time=self.time, waveform_id=waveform_id,
                      phase_hint=self.phase, onset=self.onset,
                      evaluation_mode=self.evaluation_mode,
                      evaluation_status=self.evaluation_status)


# class Pick(BaseModel):
#     resource_id: ResourceIdentifier
#     time: datetime
#     waveform_id: WaveformStreamID
#     filter_id: Optional[str]
#     method_id: Optional[str]
#     horizontal_slowness: Optional[float]
#     horizontal_slowness_error: Optional[QuantityError]
#     backazimuth: Optional[float]
#     backazimuth_error: Optional[QuantityError]
#     slowness_method_id: Optional[str]
#     onset: Optional[PickOnset]
#     phase_hint: str
#     polarity: Optional[PickPolarity]
#     evaluation_mode: EvaluationMode
#     evaluation_status: EvaluationStatus
#     creation_info: Optional[CreationInfo]
#
#     class Config:
#         orm_mode = True

