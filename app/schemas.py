# app/schemas.py

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    class APIModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

except Exception:
    class APIModel(BaseModel):
        class Config:
            extra = "forbid"

class Customer(APIModel):

    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = Field(default=None, ge=0, le=1)
    Partner: Optional[str] = Field(default="No")
    Dependents: Optional[str] = None
    PhoneService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    Contract: Optional[str] = None
    PaymentMethod: Optional[str] = None

    tenure: Optional[int] = Field(default=None, ge=0)
    MonthlyCharges: Optional[float] = Field(default=None, ge=0)
    TotalCharges: Optional[Union[float, str]] = None


class PredictRequest(APIModel):
    customer: Customer


class PredictResponse(APIModel):
    churn_probability: float


class DriftRequest(APIModel):
    customers: List[Customer]


class DriftResponse(APIModel):
    drift: Dict[str, Any] = Field(default_factory=dict)


class ABSelectRequest(APIModel):
    customers: List[Customer]
    k: int = Field(..., ge=1)


class ABSelectResponse(APIModel):
    control_top_k_idx: List[int]
    treatment_top_k_idx: List[int]


class UpliftRequest(APIModel):
    customer: Customer


class UpliftResponse(APIModel):
    p_treated: float
    p_control: float
    uplift: float