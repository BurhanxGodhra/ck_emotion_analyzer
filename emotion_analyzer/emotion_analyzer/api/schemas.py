from pydantic import BaseModel


class PredictRequest(BaseModel):
    image: str  # base64 data URL: "data:image/jpeg;base64,...."


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FacePrediction(BaseModel):
    label: str
    confidence: float
    all_probs: dict[str, float]
    box: BoundingBox


class PredictResponse(BaseModel):
    faces: list[FacePrediction]
