from pydantic import BaseModel
from typing import Optional, List


class ClassifierResults(BaseModel):
    networks: List[str]
    stations: List[str]
    channels: List[str]
    raw_outputs: List[List[float]]
    predicted_classes: List[str]
    probabilities: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "raw_outputs": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]
                ],
                "labels": ["class1", "class2", "class3"],
                "probabilities": [
                    [0.5, 0.3, 0.2],
                    [0.2, 0.6, 0.2],
                    [0.1, 0.3, 0.6]
                ]
            }
        }
