from pydantic import BaseModel
from typing import Optional, List
from uquake.core.event import Pick as UQPick
# from obspy.core.event.header import (PickOnset, PickPolarity, EvaluationMode,
#                                      EvaluationStatus)
from enum import Enum
#
#


class WaveformStreamID(BaseModel):
    station_code: str
    channel_code: str
    location_code: str


class QuantityError(BaseModel):
    uncertainty: Optional[float]
    lower_uncertainty: Optional[float]
    upper_uncertainty: Optional[float]
    confidence_level: Optional[float]


class PickOnset(str, Enum):
    emergent = "emergent"
    impulsive = "impulsive"
    questionable = "questionable"


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
    creation_time: Optional[str]
    version: Optional[str]


class Pick(BaseModel):
    resource_id: str
    time: str
    waveform_id: WaveformStreamID
    filter_id: Optional[str]
    method_id: Optional[str]
    horizontal_slowness: Optional[float]
    horizontal_slowness_error: Optional[QuantityError]
    backazimuth: Optional[float]
    backazimuth_error: Optional[QuantityError]
    slowness_method_id: Optional[str]
    onset: Optional[PickOnset]
    phase_hint: str
    polarity: Optional[PickPolarity]
    evaluation_mode: EvaluationMode
    evaluation_status: EvaluationStatus
    creation_info: Optional[CreationInfo]

# class Pick(BaseModel):
#     pass

