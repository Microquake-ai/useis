from .event import Pick

class Observations(BaseModel):
    picks: List[Pick]
    p_pick_error: Optional[float]
    s_pick_error: Optional[float]