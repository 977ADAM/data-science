# app/schemas.py

from typing import List, Optional, Union
from pydantic import BaseModel, Field

class Customer(BaseModel):

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


class PredictRequest(BaseModel):
    customer: Customer


class PredictResponse(BaseModel):
    churn_probability: float


class DriftRequest(BaseModel):
    customers: List[Customer]


class DriftResponse(BaseModel):
    # структура свободная (metrics per feature + prediction drift)
    drift: dict


class ABSelectRequest(BaseModel):
    customers: List[Customer]
    k: int


class ABSelectResponse(BaseModel):
    control_top_k_idx: List[int]
    treatment_top_k_idx: List[int]


class UpliftRequest(BaseModel):
    customer: Customer


class UpliftResponse(BaseModel):
    p_treated: float
    p_control: float
    uplift: float